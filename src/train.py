import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

def train_model(model, dataloader, config, device, config_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    num_epochs = config['training']['num_epochs']
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config['training']['max_norm']))
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Get the directory of the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Construct the full path to save the model
    model_save_path = os.path.join(config_dir, config['paths']['model_save_path'])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logging.info(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    return model
