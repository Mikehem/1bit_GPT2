import os
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        binary_weights = self.binarize(self.linear.weight)
        binary_input = self.binarize(x)
        return nn.functional.linear(binary_input, binary_weights)

    @staticmethod
    def binarize(tensor):
        return tensor.sign()

def binarize_model(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BinaryLinear(module.in_features, module.out_features))
        else:
            binarize_model(module)

def create_1bit_gpt_model(model_config, config_path):
    config = GPT2Config(**model_config)
    
    # Get the directory of the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Construct the full path to the model file
    model_path = os.path.join(config_dir, model_config.get('model_save_path', ''))
    
    if os.path.exists(model_path) and any(f.endswith(('.bin', '.safetensors', '.h5', '.ckpt.index', '.msgpack')) for f in os.listdir(model_path)):
        print(f"Loading existing model from {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        print("Initializing new model for pre-training from scratch")
        model = GPT2LMHeadModel(config)
    
    binarize_model(model)
    return model
