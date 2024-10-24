import os
import yaml
import torch
from transformers import GPT2Tokenizer
from model import create_1bit_gpt_model
from generate import generate_claim_summary
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.ERROR)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(config, config_path):
    model = create_1bit_gpt_model(config['model'])
    
    # Get the directory of the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Construct the full path to the model file
    model_path = os.path.join(config_dir, config['paths']['model_save_path'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main(config_path):
    print("Loading configuration and model...")
    config = load_config(config_path)
    try:
        model = load_model(config, config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the model file exists and the path in the config file is correct.")
        return

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"\nModel loaded successfully. Using device: {device}")
    print("\nEnter your prompts below. Type 'exit' to quit.")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'exit':
            break

        print("\nGenerating summary...")
        summary = generate_claim_summary(model, tokenizer, prompt, config['generation']['max_length'])
        
        print("\n" + "="*50)
        print("Generated Summary:")
        print("-"*50)
        print(summary)
        print("="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
