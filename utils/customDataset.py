from torch.utils.data import Dataset
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_inputs, labels=None):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        # Assuming offset_mapping is needed elsewhere, keep it, but it's not used for the model directly
        self.offset_mapping = tokenized_inputs.get("offset_mapping", None)
        self.labels = labels  # This can be None for prediction

    def __len__(self):
        # Use the length of input_ids as the dataset size since labels might be None
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        # Include offset_mapping in the item if it exists
        if self.offset_mapping is not None:
            item["offset_mapping"] = self.offset_mapping[idx]
        # Only add labels to the item if they are not None
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)