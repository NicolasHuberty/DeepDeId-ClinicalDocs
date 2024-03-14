import argparse
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification,AutoTokenizer, RobertaForTokenClassification,RobertaTokenizerFast
from models import tokenize_and_encode_labels
from utils import readFormattedFile, CustomDataset, evaluate_model

# Retrieve all arguments
parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('--eval_set', type=str, default="n2c2_2014/training3.tsv", help='Path to the validation set')
parser.add_argument('--batch_size', type=int, default=4, help='Testing batch size')
parser.add_argument('--mapping',type=str, default="n2c2_removeBIO", help="None for no mapping, name of the file otherwise")
parser.add_argument('--model_name', type=str, default="roberta-n2c2_2014n2c2_2014_-mapping_n2c2_removeBIO-epochs_5-batch_size_4", help='Path to the validation set')
args = parser.parse_args()

# Define model and tokenizer for the evaluation
print(f"Load model: {args.model_name} and tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = RobertaForTokenClassification.from_pretrained(f"./model_save/{args.model_name}")

# Process the evaluation set
eval_tokens, eval_labels, unique_labels = readFormattedFile(args.eval_set,mapping=args.mapping)
print(f"Process {args.eval_set} with mapping {args.mapping} \n Retrieved Labels: {unique_labels} and model labels: {model.config.label2id}")

# Tokenize, encode and create evaluation dataset
tokenized_val_inputs = tokenizer(eval_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
encoded_val_labels = tokenize_and_encode_labels(eval_labels, tokenized_val_inputs,model.config.label2id)
eval_dataset = CustomDataset(tokenized_val_inputs, encoded_val_labels)

# Launch the evaluation
print(f"Launch evaluation...")
metrics = evaluate_model(model, eval_dataset, 'cuda',tokenizer)
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
df_metrics.to_csv(f"./results/{model.config.name}.csv", index_label='Label ID')