from torch.utils.data import Dataset, DataLoader
import torch
class TextDataset(Dataset):
    """A custom Dataset class for handling tokenized texts."""
    def __init__(self, tokenized_inputs):
        self.tokenized_inputs = tokenized_inputs

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.tokenized_inputs.items()}

