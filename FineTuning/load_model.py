import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# we are going to need to 


model_saved = ''
model_name = ''

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(model_saved, weights_only=True))

config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(config=config)

model.push_to_hub(model_name, config=config)