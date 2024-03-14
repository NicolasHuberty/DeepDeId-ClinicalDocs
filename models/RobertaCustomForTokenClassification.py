from transformers import RobertaConfig, RobertaModel, RobertaForTokenClassification
import torch.nn as nn
import torch.nn.functional as F


class RobertaCustomForTokenClassification(nn.Module):
    """Custom class of Roberta where the last layer is updated to allow all number of labels

    Args:
        num_labels (int): The number of unique labels for the token classification task.
    """
    def __init__(self, num_labels):
        super(RobertaCustomForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.config = RobertaConfig.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        self.config.num_labels = num_labels

        self.roberta = RobertaModel.from_pretrained("Jean-Baptiste/roberta-large-ner-english", config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None,offset_mapping=None, labels=None, threshold=0.5):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0])
        
        probabilities = F.softmax(logits, dim=-1)
        label_predictions = (probabilities > threshold).long()
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Returning a dictionary for consistency
        return {
            'loss': loss,  # This will be None if no labels are provided
            'logits': logits,
            'label_predictions': label_predictions
        }