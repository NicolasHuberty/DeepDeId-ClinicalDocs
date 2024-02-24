from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    """
    Custom dataset for PyTorch, handling encoded tokens and labels for model training.
    Parameters:
    - encodings (dict): Encoded input tokens.
    - labels (list): Corresponding labels for inputs.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
