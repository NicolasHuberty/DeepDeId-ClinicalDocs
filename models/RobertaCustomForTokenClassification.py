# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
from transformers import RobertaConfig, RobertaModel, RobertaForTokenClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

"""
Custom Token classification model based on RoBERTa.

Parameters:
- num_labels (int): Number of labels for the classification task.
- config (RobertaConfig, optional): Configuration object for the RoBERTa model. If None, defaults are loaded.
"""
class RobertaCustomForTokenClassification(nn.Module):
    def __init__(self, num_labels, config=None):
        super(RobertaCustomForTokenClassification, self).__init__()
        
        if isinstance(num_labels, RobertaConfig):
            self.config = num_labels
        else:
            self.config = RobertaConfig.from_pretrained("Jean-Baptiste/roberta-large-ner-english", num_labels=num_labels)
        
        self.roberta = RobertaModel.from_pretrained("Jean-Baptiste/roberta-large-ner-english", config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, threshold=0.5):
        """
        Receives the tokenized informations and do all the computations
        
        Parameters:
        - input_ids (Tensor): Indices of input sequence tokens in the vocabulary.
        - attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
        - labels (Tensor, optional): Labels for computing the loss when training.
        - threshold (float): Threshold for converting probabilities to binary predictions.
        
        Returns:
        - dict: Containing the loss, logits, and predicted labels.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0])
        
        probabilities = F.softmax(logits, dim=-1)
        label_predictions = (probabilities > threshold).long()
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else torch.tensor([True]*logits.numel())
            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        return {'loss': loss, 'logits': logits, 'label_predictions': label_predictions}
    
    @classmethod
    def from_pretrained(cls, pretrained_dir, *model_args, **kwargs):
        # Load a model and its configuration from a predefined directory
        config = RobertaConfig.from_pretrained(pretrained_dir)
        model = cls(config.num_labels, config=config)
        model_path = os.path.join(pretrained_dir, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(self, save_directory):
        # Save the model and its configuration to a specified directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        config_path = os.path.join(save_directory, 'config.json')
        self.config.to_json_file(config_path)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
