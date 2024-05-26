import sys
from pathlib import Path
import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments, RobertaTokenizerFast,  BertTokenizerFast, BertForTokenClassification,CamembertTokenizerFast, CamembertForTokenClassification
from transformers import AutoTokenizer, XLMRobertaForTokenClassification, AutoModelForTokenClassification, XLMRobertaConfig, FlaubertForTokenClassification, FlaubertTokenizer, CamembertConfig
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils import readFormattedFile, CustomDataset, evaluate_model, plot_f1
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
import matplotlib.pyplot as plt

# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DeIdentidication of clinical documents using deep learning')
    parser.add_argument("--train_set",nargs="+",default=["n2c2_2014/training1.tsv","n2c2_2014/training2.tsv"],help="List of datasets for training")
    parser.add_argument('--eval_set', type=str, default="n2c2_2014/training3.tsv", help='Dataset for testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--mapping',type=str, default="n2c2_removeBIO", help="None for no mapping, name of the file otherwise")
    parser.add_argument('--dataset_size',type=str,default=-1)
    parser.add_argument('--variant_name',type=str,default="roberta")
    parser.add_argument('--transfer_learning_path',type=str,default="None",help="Enable transfer Learning")
    args = parser.parse_args()
    return args

def process_training_set(args):
    """Process the training set"""
    tokens, labels, unique_labels_train = [], [], set()
    train_dataset_string = ""
    for dataset in args.train_set:
        t, l, unique_labels = readFormattedFile(dataset,args.mapping)
        tokens.extend(t)
        labels.extend(l)
        unique_labels_train.update(unique_labels)
        train_dataset_string += dataset.split("/")[0]
        print(f"Retrieve training dataset {dataset} Labels: {unique_labels_train} with mapping: {args.mapping}")
    tokens = tokens[0:int(args.dataset_size)]
    labels = labels[0:int(args.dataset_size)]
    print(f"Number of records to train: {len(tokens)}")
    return tokens, labels, unique_labels_train, train_dataset_string

def initialize_model_and_tokenizer(unique_labels_train, train_dataset_string, args):
    """Initialize the tokenizer and the model"""
    if(args.variant_name == "roberta"):
        tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        if(args.transfer_learning_path != "None"):
            model = RobertaCustomForTokenClassification(args.transfer_learning_path)
        else:
            model = RobertaCustomForTokenClassification(num_labels=len(unique_labels_train))
    elif(args.variant_name == "mbert"):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
        if(args.transfer_learning_path) != "None":
            model = BertForTokenClassification.from_pretrained(args.transfer_learning_path)
        else:
            model = BertForTokenClassification.from_pretrained("bert-base-multilingual-uncased",num_labels=len(unique_labels_train))
    elif(args.variant_name == "xlm"):
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        if(args.transfer_learning_path) != "None":
            model = AutoModelForTokenClassification.from_pretrained(args.transfer_learning_path)
        else:
            model = AutoModelForTokenClassification.from_pretrained('xlm-roberta-large',num_labels=len(unique_labels_train))
    else:
        tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        if(args.transfer_learning_path) != "None":
            model = AutoModelForTokenClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", num_labels=len(unique_labels_train))
        else:
            model = AutoModelForTokenClassification.from_pretrained(args.transfer_learning_path)
    model.config.name = f"{args.variant_name}-{train_dataset_string}_{args.dataset_size}-mapping_{args.mapping}-epochs_{args.epochs}-batch_size_{args.batch_size}"
    model.config.label2id = {label: id for id, label in enumerate(unique_labels_train)}
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    return model, tokenizer

def tokenize_and_process_labels(tokens, labels, tokenizer, model):
    """Tokenize and process labels for training and evaluation set"""
    print(f"Tokenize the tokens and process labels...")
    tokenized_train_inputs = tokenizer(tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
    encoded_train_labels = tokenize_and_encode_labels(labels, tokenized_train_inputs,model.config.label2id)
    return tokenized_train_inputs, encoded_train_labels

def define_training_parameters(args):
    """Define training parameters."""
    training_args = TrainingArguments(
        output_dir="./.trainingLogs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="no",
        logging_dir="./logs",  # Path to store the log files
        logging_strategy="steps",  # Log at regular step intervals
        logging_steps=10  # Log metrics every 10 steps
    )
    return training_args


def evaluate_and_save(model,tokenizer, args):
    """Evaluate the performance of the model and save the results."""
    print(f"Retrieve eval dataset {args.eval_set}")
    eval_tokens, eval_labels, eval_unique_labels = readFormattedFile(args.eval_set,args.mapping)
    tokenized_eval_inputs = tokenizer(eval_tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
    encoded_eval_labels = tokenize_and_encode_labels(eval_labels, tokenized_eval_inputs,model.config.label2id)
    eval_dataset = CustomDataset(tokenized_eval_inputs, encoded_eval_labels)

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
    df_metrics.to_csv(f"results/{model.config.name}.csv", index_label='Label ID')

    # Plot results
    #print(f"Plot performance metrics...")
    #plot_f1(metrics, [label for _, label in sorted(model.config.id2label.items())],model.config.name)

def main():
    args = parse_arguments()
    print(f"Launch training with args.variant_name with: {args.train_set} eval set: {args.eval_set} epochs: {args.epochs} batch size: {args.batch_size}")
    tokens, labels, unique_labels_train, train_dataset_string = process_training_set(args)
    model, tokenizer = initialize_model_and_tokenizer(unique_labels_train, train_dataset_string, args)
    tokenized_train_inputs, encoded_train_labels = tokenize_and_process_labels(tokens, labels, tokenizer, model)
    train_dataset = CustomDataset(tokenized_train_inputs, encoded_train_labels)
    training_args = define_training_parameters(args)

    # Create the trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print(f"Launch the training for {args.epochs} epochs...")
    trainer.train()
    # Save the model
    print(f"Save the model {model.config.name}")
    if(args.variant_name == "roberta"):
        model.save_pretrained(f"models_save/{model.config.name}")
    else:
        trainer.save_model(f"models_save/{model.config.name}")
    if(args.eval_set != "None"):
        evaluate_and_save(model,tokenizer, args)


if __name__ == "__main__":
        main()