# Muon optimizer from https://github.com/KellerJordan/modded-nanogpt
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        # Check if we're in distributed mode and initialize properly
        self.distributed = dist.is_available() and dist.is_initialized()
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.Tensor) for p in params)
        
        # Handle parameter grouping for both single and multi-GPU scenarios
        sizes = {p.numel() for p in params}
        param_groups = []
        
        for size in sizes:
            size_params = [p for p in params if p.numel() == size]
            # Ensure params are evenly divisible by world_size in distributed setting
            if self.distributed:
                # Pad parameter list if needed
                remainder = len(size_params) % self.world_size
                if remainder:
                    padding = self.world_size - remainder
                    size_params.extend([size_params[-1]] * padding)
            
            param_groups.append({
                'params': size_params,
                'update_buffer': [
                    torch.empty(size, device='cuda', dtype=torch.bfloat16)
                    for _ in range(max(1, self.world_size))
                ] if self.distributed else None,
            })
        
        super().__init__(param_groups, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            params = group['params']
            update_buffers = group['update_buffer']

            if self.distributed:
                # Distributed training logic
                assert len(params) % self.world_size == 0
                handle = None
                params_world = None

                def update_prev():
                    if params_world is None:
                        return
                    assert handle is not None
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffers):
                        p_world.data.add_(
                            g_world.view_as(p_world),
                            alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                        )

                for base_i in range(len(params))[::self.world_size]:
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        continue
                        
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                    
                    update_prev()
                    handle = dist.all_gather(update_buffers, g, async_op=True)
                    params_world = params[base_i : base_i + self.world_size]
                
                update_prev()
                
            else:
                # Single GPU logic
                for p in params:
                    if p.grad is None:
                        continue
                        
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    
                    # Apply update directly in single GPU case
                    p.data.add_(
                        g.view_as(p),
                        alpha=-lr * max(1, p.size(0) / p.size(1)) ** 0.5,
                    )