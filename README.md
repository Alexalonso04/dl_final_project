# GPT-X

## Quick Start:

Run the following commands to begin a training run on a cloud instance running pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel 
It is available as a preconfigured template on vast.ai. Alternatively, you may download it here: https://hub.docker.com/r/pytorch/pytorch/

The test config is setup to automatically download a chunk of a cached version of the dataset and begin training immediately.
```bash
pip install -r requirements.txt

# Standard training
python main.py configs/test_config.yaml

# Multi-GPU training
torchrun --nproc_per_node=8 main.py configs/test_config.yaml # Replace 8 with the number of GPUs available


```

## Requirements

```
torch>=2.0
huggingface-hub
tiktoken
transformers
triton
```

## Citation

@misc{karpathy2023nanoGPT,
  title={nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs},
  author={Andrej Karpathy},
  year={2023},
  url={https://github.com/karpathy/nanoGPT}
}

@article{chen2023difftransformer,
  title={Diff-Transformer: A General Framework for Zero-Shot Transformers Quantization},
  author={Chen, Xiao and Du, Xiaohan and Ding, Mingyu and Xu, Lin},
  journal={arXiv preprint arXiv:2401.08541},
  year={2024}
}

@article{jordan2023muon,
  title={MUON: Memory-Efficient Orthogonal Optimization},
  author={Jordan, Keller},
  year={2023},
  url={https://github.com/KellerJordan/modded-nanogpt}
}

## License

MIT