import logging
import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AutoModelForSequenceClassification
from models import tokenize_and_encode_labels, compute_metrics
from utils import readFormattedFile, CustomDataset, evaluate_model, complete_plot,dataset_statistics,plot_f1
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Retrieve all arguments
parser = argparse.ArgumentParser(description='DeIdentidication of clinical documents using deep learning')
parser.add_argument("--train_set",nargs="+",default=["n2c2_2014/training1.tsv","n2c2_2014/training2.tsv"],help="List of datasets for training")
parser.add_argument('--eval_set', type=str, default="n2c2_2014/training3.tsv", help='Dataset for testing')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
parser.add_argument('--mapping',type=str, default="n2c2_removeBIO", help="None for no mapping, name of the file otherwise")
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

# Process evaluation set
print(f"Retrieve eval dataset {args.eval_set}")
eval_tokens, eval_labels, eval_unique_labels = readFormattedFile(args.eval_set,args.mapping)
dataset_statistics(tokens,labels,unique_labels_train)


# Initialize tokenizer and model
#tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-uncased")
#model = BertForTokenClassification.from_pretrained("google-bert/bert-base-multilingual-uncased",num_labels=len(unique_labels_train))
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
model = BertForTokenClassification.from_pretrained('bert-large-uncased', num_labels=len(unique_labels_train))
model.config.name = f"bert-{train_dataset_string}-mapping_{args.mapping}-epochs_{args.epochs}"
model.config.label2id = {label: id for id, label in enumerate(unique_labels_train)}
model.config.id2label = {id: label for label, id in model.config.label2id.items()}

# Tokenize and encode labels for training and evaluation set
print(f"Tokenize the tokens and process labels...")
tokenized_train_inputs = tokenizer(tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
tokenized_eval_inputs = tokenizer(eval_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
encoded_train_labels = tokenize_and_encode_labels(labels, tokenized_train_inputs,model.config.label2id)
encoded_eval_labels = tokenize_and_encode_labels(eval_labels, tokenized_eval_inputs,model.config.label2id)

# Create dataset for training and evaluation
print(f"Transform datasets tokenized in customDataset...")
train_dataset = CustomDataset(tokenized_train_inputs, encoded_train_labels)
eval_dataset = CustomDataset(tokenized_eval_inputs, encoded_eval_labels)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./.trainingLogs",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    #evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    #compute_metrics=compute_metrics,
)

# Launch the training of the model
print(f"Launch the training for {args.epochs} epochs...")
trainer.train()

# Save the model
print(f"Save the model {model.config.name}")
trainer.save_model(f"model_save/{model.config.name}")

# Evaluate the performance of the model
print(f"Evalue the performance of the model...")
metrics = evaluate_model(model, eval_dataset,'cuda',tokenizer)

# Format performances to be fit in csv file
df_metrics = pd.DataFrame({
    'Support': pd.Series(metrics['support_per_label']),
    'True Positives': pd.Series(metrics['TP']),
    'False Positives': pd.Series(metrics['FP']),
    'False Negatives': pd.Series(metrics['FN']),
    'True Negatives': pd.Series(metrics['TN']),
    'Precision': pd.Series(metrics['precision_per_label']),
    'Recall': pd.Series(metrics['recall_per_label']),
    'F1 Score': pd.Series(metrics['f1_per_label']),
})

# Save results
df_metrics.index = [label for _, label in sorted(model.config.id2label.items())]
print(f"df_metrics.index: {df_metrics.index}")
df_metrics.to_csv(f"./results/{model.config.name}.csv", index_label='Label ID')

# Plot results
print(f"Plot performance metrics...")
plot_f1(metrics, [label for _, label in sorted(model.config.id2label.items())],model.config.name)