import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support,precision_score, recall_score, f1_score
from tqdm import tqdm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_model(model, eval_dataset, device='cuda', tokenizer=None):
    """
    Evaluate a model on a given evaluation dataset.
    Parameters:
    - model: The model to evaluate.
    - eval_dataset: The dataset for evaluation.
    - device (str): The device to run the evaluation on ('cuda' or 'cpu').
    - tokenizer: Tokenizer to decode the tokens for mismatch analysis.
    Returns:
    - dict: A dictionary containing various evaluation metrics and information.
    """
    id2label = {v: k for k, v in model.config.label2id.items()}
    # Pass the model on evaluation mode
    model.eval()
    model.to(device)
    # Create a DataLoader to process the evaluation dataset
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    all_predictions, all_true_labels = [], []
    mismatched_tokens_info = []  # All badly predicted labels

    for batch in tqdm(eval_loader, desc="Evaluating the model"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device) if 'labels' in batch else None
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1)
        all_predictions.append(batch_predictions.cpu().numpy())
        all_true_labels.append(labels.cpu().numpy())

        # Detect mismatched labels
        input_ids = inputs['input_ids'].cpu().numpy()
        for input_id, prediction, true_label in zip(input_ids, batch_predictions.cpu().numpy(), labels.cpu().numpy()):
            for idx, (pred, true) in enumerate(zip(prediction, true_label)):
                if pred != true and true != -100: 
                    token_id = input_id[idx]
                    token = tokenizer.decode([token_id], skip_special_tokens=True)
                    true_label_name = id2label.get(true, 'Unknown Label')
                    pred_label_name = id2label.get(pred, 'Unknown Label')
                    mismatched_tokens_info.append({'token': token, 'true_label': true_label_name, 'predicted_label': pred_label_name})
    
    # Format all the results
    flat_predictions = np.concatenate(all_predictions, axis=None)
    flat_true_labels = np.concatenate(all_true_labels, axis=None)
    valid_indices = flat_true_labels != -100 
    filtered_predictions = flat_predictions[valid_indices]
    filtered_true_labels = flat_true_labels[valid_indices]
    unique, counts = np.unique(flat_true_labels[flat_true_labels != -100], return_counts=True)
    support_per_label = dict(zip(unique, counts))

    # Calculate TP, FP, FN, TN
    cm = confusion_matrix(filtered_true_labels, filtered_predictions, labels=list(model.config.label2id.values()))
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    metrics = {
        'precision_per_label': precision_score(filtered_true_labels, filtered_predictions, labels=list(model.config.label2id.values()), average=None),
        'recall_per_label': recall_score(filtered_true_labels, filtered_predictions, labels=list(model.config.label2id.values()), average=None),
        'f1_per_label': f1_score(filtered_true_labels, filtered_predictions, labels=list(model.config.label2id.values()), average=None),
        'global_precision': precision_score(filtered_true_labels, filtered_predictions, average='macro'),
        'global_recall': recall_score(filtered_true_labels, filtered_predictions, average='macro'),
        'global_f1': f1_score(filtered_true_labels, filtered_predictions, average='macro'),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'support_per_label': support_per_label,
        'mismatched_tokens_info': mismatched_tokens_info
    }

    return metrics
