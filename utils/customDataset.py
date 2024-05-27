# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    # Initialize the CustomDataset class, which is a subclass of PyTorch's Dataset.
    def __init__(self, tokenized_inputs, labels=None):
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_inputs["input_ids"]]
        self.attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in tokenized_inputs["attention_mask"]]
        self.offset_mapping = tokenized_inputs.get("offset_mapping", None)
        if labels is not None:
            self.labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        else:
            self.labels = None

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