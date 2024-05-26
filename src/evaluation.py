import argparse
import pandas as pd
from pathlib import Path
import sys
from transformers import BertTokenizerFast, BertForTokenClassification,AutoTokenizer,RobertaTokenizerFast,AutoModelForTokenClassification
import torch
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
from utils import readFormattedFile, CustomDataset, evaluate_model

# Check availability of a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluation will run on {device}")
# Retrieve all arguments
parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('--eval_set', nargs="+", default=["n2c2/train_set.tsv"], help='Path to the evaluations set')
parser.add_argument('--batch_size', type=int, default=4, help='Testing batch size')
parser.add_argument('--mapping',type=str, default="None", help="None for no mapping, name of the file otherwise")
parser.add_argument('--model_path', type=str, default="modelGood", help='Model name for the result files')
parser.add_argument('--variant_name', type=str, default="roberta", help='Name of the model')
args = parser.parse_args()

# Select model and tokenizer for the evaluation
print(f"Load model: {args.model_path} and tokenizer...")
if(args.variant_name == "roberta"):
    tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = RobertaCustomForTokenClassification.from_pretrained(f"models_save/{args.model_path}")
elif(args.variant_name == "mbert"):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
        model = BertForTokenClassification.from_pretrained(f"models_save/{args.model_path}")
elif(args.variant_name == "xlm"):
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForTokenClassification.from_pretrained(f"models_save/{args.model_path}")
else:
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model = AutoModelForTokenClassification.from_pretrained(f"models_save/{args.model_path}")

# Process the evaluation set
eval_tokens, eval_labels, unique_labels = [], [], set()
train_dataset_string = ""
for dataset in args.eval_set:
    t, l, u = readFormattedFile(dataset,args.mapping)
    eval_tokens.extend(t)
    eval_labels.extend(l)
    unique_labels.update(u)
    train_dataset_string += dataset.split("/")[0]
    print(f"Retrieve testing dataset {dataset} Labels: {unique_labels} with mapping: {args.mapping}")
print(f"Process {args.eval_set} with mapping {args.mapping} \n Retrieved Labels: {unique_labels} and model labels: {model.config.label2id}")


# Tokenize, encode and create evaluation dataset
tokenized_val_inputs = tokenizer(eval_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
encoded_val_labels = tokenize_and_encode_labels(eval_labels, tokenized_val_inputs,model.config.label2id)
eval_dataset = CustomDataset(tokenized_val_inputs, encoded_val_labels)

# Launch the evaluation
print(f"Launch evaluation...")
metrics = evaluate_model(model, eval_dataset, device, tokenizer)
# Format performances to fit in csv file
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
# Save results in results folder
df_metrics.index = [label for _, label in sorted(model.config.id2label.items())]
df_metrics.to_csv(f"./results/{args.model_path}_{train_dataset_string}.csv", index_label='Label ID')