from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,precision_score, recall_score, f1_score
import numpy as np
from transformers import EvalPrediction
def compute_metrics(p: EvalPrediction):
    predictions = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids

    # Filter out the ignored token predictions (e.g., -100 used for padding in some models)
    true_predictions = [
        [p for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, true_labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in true_labels
    ]

    # Flatten the lists
    true_predictions = [p for sublist in true_predictions for p in sublist]
    true_labels = [l for sublist in true_labels for l in sublist]

    # Calculate metrics for each label
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average=None, labels=np.unique(true_labels)
    )

    # Structure results in a dictionary
    label_metrics = {}
    for label_index, label in enumerate(np.unique(true_labels)):
        label_metrics[f'label_{label}'] = {
            'precision': precision[label_index],
            'recall': recall[label_index],
            'f1': f1[label_index],
        }

    return label_metrics