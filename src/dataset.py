import torch
from torch.utils.data import Dataset

class HealthClaimsDataset(Dataset):
    def __init__(self, sequences, max_length):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate the sequence to max_length
        if len(seq) < self.max_length:
            padding = [0] * (self.max_length - len(seq))
            seq = seq + padding
        else:
            seq = seq[:self.max_length]
        
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': labels}
