import os
import yaml
import torch
from transformers import GPT2Tokenizer
from ocr_utils import extract_text_from_image
from preprocessing import preprocess_text
from model import create_1bit_gpt_model
from dataset import HealthClaimsDataset
from train import train_model
from generate import generate_claim_summary
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_sequences(tokens, seq_length, stride=None):
    if stride is None:
        stride = seq_length // 2  # 50% overlap by default

    sequences = []
    for i in range(0, len(tokens), stride):
        end = min(i + seq_length, len(tokens))
        seq = tokens[i:end]
        if len(seq) >= seq_length // 2:  # Only keep sequences at least half the desired length
            sequences.append(seq)
        if end == len(tokens):
            break
    return sequences

def process_image(image_path, processed_folder):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_file = os.path.join(processed_folder, f"{base_name}.txt")
    
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        text = extract_text_from_image(image_path)
        processed_text = preprocess_text(text)
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        return processed_text

def process_images(image_folder, processed_folder, file_extensions):
    all_text = ""
    for image_file in tqdm(os.listdir(image_folder), desc="Processing images"):
        if image_file.endswith(file_extensions):
            image_path = os.path.join(image_folder, image_file)
            text = process_image(image_path, processed_folder)
            all_text += text + " "
    return all_text

def main(config_path):
    logging.info("Loading configuration...")
    config = load_config(config_path)

    image_folder = config['dataset']['image_folder']
    processed_folder = config['dataset']['processed_folder']
    file_extensions = tuple(config['dataset']['file_extensions'])

    if processed_folder is None:
        processed_folder = os.path.join(os.path.dirname(image_folder), 'processed')

    os.makedirs(processed_folder, exist_ok=True)

    logging.info("Processing images...")
    all_text = process_images(image_folder, processed_folder, file_extensions)

    logging.info(f"Total text length: {len(all_text)} characters")

    logging.info("Tokenizing the text...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(all_text)
    logging.info(f"Number of tokens after encoding: {len(tokens)}")

    logging.info("Creating sequences...")
    seq_length = config['training']['seq_length']
    sequences = create_sequences(tokens, seq_length)
    logging.info(f"Number of sequences created: {len(sequences)}")

    if len(sequences) == 0:
        logging.error("No sequences were created. The dataset is empty.")
        return

    logging.info("Creating dataset and dataloader...")
    dataset = HealthClaimsDataset(sequences, seq_length)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    logging.info("Initializing the model...")
    try:
        model = create_1bit_gpt_model(config['model'], config_path)
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        return

    logging.info("Training the model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    trained_model = train_model(model, dataloader, config, device, config_path)

    logging.info("Generating a summary...")
    summary = generate_claim_summary(trained_model, tokenizer, config['generation']['prompt'], config['generation']['max_length'])
    logging.info(f"Generated summary: {summary}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
