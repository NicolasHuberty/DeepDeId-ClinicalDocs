from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs, labels=None):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.offset_mapping = tokenized_inputs.get("offset_mapping", None)
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        if self.offset_mapping is not None:
            item["offset_mapping"] = self.offset_mapping[idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
class TextDataset(Dataset):
    """A custom Dataset class for handling tokenized texts."""
    def __init__(self, tokenized_inputs):
        self.tokenized_inputs = tokenized_inputs

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.tokenized_inputs.items()}

