import sys
from pathlib import Path
import logging
import argparse
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, CamembertTokenizerFast, CamembertForTokenClassification
from transformers import BertTokenizerFast, BertForTokenClassification,RobertaTokenizerFast, RobertaForTokenClassification, DistilBertTokenizerFast, DistilBertForTokenClassification, AlbertTokenizerFast, AlbertForTokenClassification,AutoTokenizer, RobertaForTokenClassification
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from models import tokenize_and_encode_labels, compute_metrics,RobertaCustomForTokenClassification
from utils import readFormattedFile, CustomDataset, evaluate_model,dataset_statistics,plot_f1
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,precision_score, recall_score, f1_score
from transformers import EvalPrediction, TrainerCallback
import matplotlib.pyplot as plt
logging.getLogger('tensorflow').setLevel(logging.ERROR)
class MetricsCallback(TrainerCallback):
    """A callback that logs the evaluation results to a list."""
    def __init__(self):
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.metrics.append(metrics['eval_f1'])

def compute_metrics(p: EvalPrediction):
    predictions = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids
    true_predictions = [[p for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, true_labels)]
    true_labels = [[l for l in label if l != -100] for label in true_labels]
    true_predictions = [p for sublist in true_predictions for p in sublist]
    true_labels = [l for sublist in true_labels for l in sublist]
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='macro')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Define the dataset sizes
f1_scores_df = pd.DataFrame()

# Retrieve all arguments
parser = argparse.ArgumentParser(description='DeIdentidication of clinical documents using deep learning')
parser.add_argument("--train_set",nargs="+",default=["n2c2_2014/training1.tsv","n2c2_2014/training2.tsv"],help="List of datasets for training")
parser.add_argument('--eval_set', type=str, default="n2c2_2014/training3.tsv", help='Dataset for testing')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
parser.add_argument('--mapping',type=str, default="None", help="None for no mapping, name of the file otherwise")
parser.add_argument('--dataset_size',type=str,default=str(790))
args = parser.parse_args()

print(f"Launch training with bert-base-uncased with: {args.train_set} eval set: {args.eval_set} epochs: {args.epochs} batch size: {args.batch_size}")

# Retrieve the correct mapping dictionnary
train_set = ",".join(args.train_set)

# Process training set
tokens, labels, unique_labels_train = [], [], set()
train_dataset_string = ""
for dataset in args.train_set:
    t, l, unique_labels = readFormattedFile(dataset,args.mapping)
    tokens.extend(t)
    labels.extend(l)
    unique_labels_train.update(unique_labels)
    train_dataset_string += dataset.split("/")[0]
    print(f"Retrieve training dataset {dataset} Labels: {unique_labels_train} with mapping: {args.mapping}")

print(f"Number of records to train: {len(tokens)}")
# Initialize tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = RobertaCustomForTokenClassification(num_labels=len(unique_labels_train))
model.config.name = f"roberta-{train_dataset_string}_{args.dataset_size}-mapping_{args.mapping}-epochs_{args.epochs}-batch_size_{args.batch_size}"
model.config.label2id = {label: id for id, label in enumerate(unique_labels_train)}
model.config.id2label = {id: label for label, id in model.config.label2id.items()}

# Tokenize and encode labels for training and evaluation set
print(f"Tokenize the tokens and process labels...")
tokenized_train_inputs = tokenizer(tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
encoded_train_labels = tokenize_and_encode_labels(labels, tokenized_train_inputs,model.config.label2id)

eval_tokens, eval_labels, eval_unique_labels = readFormattedFile(args.eval_set,args.mapping)
tokenized_eval_inputs = tokenizer(eval_tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
encoded_eval_labels = tokenize_and_encode_labels(eval_labels, tokenized_eval_inputs,model.config.label2id)
eval_dataset = CustomDataset(tokenized_eval_inputs, encoded_eval_labels)

# Create dataset for training and evaluation
print(f"Transform datasets tokenized in customDataset...")
train_dataset = CustomDataset(tokenized_train_inputs, encoded_train_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=3,
    save_strategy="no",
)
metrics_callback = MetricsCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback],
)

# Launch the training of the model
print(f"Launch the training for {args.epochs} epochs...")
trainer.train()

    

f1_scores = metrics_callback.metrics

# Optionally, plot or save the F1 scores
plt.figure(figsize=(10, 6))
plt.plot(range(len(f1_scores)), f1_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Evaluation Step')
plt.ylabel('F1 Score')
plt.title('F1 Score Over Training')
plt.grid(True)
plt.show()