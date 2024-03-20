



import sys
from pathlib import Path
import argparse
import os, shutil
import numpy as np
import json
import sqlite3
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from transformers import Trainer, TrainingArguments, RobertaTokenizerFast,  BertTokenizerFast, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from sklearn.metrics import classification_report
from utils import CustomDataset, evaluate_model,TextDataset,load_records_manual_process,store_predicted_labels, load_config_field,save_config_field
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
from src import predict_and_align
from load_dataset import load_txt_dataset,load_dataset
from datetime import datetime

# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from torch.nn.functional import softmax

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeIdentification of clinical documents using deep learning')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of records to eval")
    args = parser.parse_args()
    return args

def evaluate_performance(model, tokenizer, texts_eval, labels_eval, label2id, id2label):
    predictions, true_labels = [], []
    for text, labels in zip(texts_eval, labels_eval):
        aligned_predicted_labels = predict_and_align(model, tokenizer, text, label2id, id2label)
        predictions.extend([label2id[label] for label in aligned_predicted_labels])
        true_labels.extend([label2id[label] for label in labels])
    return(classification_report(true_labels, predictions, target_names=list(label2id.keys())))

def evaluate_model(project_name,records,labels):
    num_labels = len(load_config_field(project_name,"labels"))
    label2id = load_config_field(project_name,"label2id")
    id2label = load_config_field(project_name,"id2label")
    model_path = load_config_field(project_name,"model_path")
    tokenizer_path = load_config_field(project_name,"tokenizer_path")        
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels)  
    evaluation = evaluate_performance(model,tokenizer,records,labels,label2id,id2label)  
    print(evaluation)
    return json.dumps(evaluation)

def main():
    args = parse_arguments()
    project_name = args.project_name
    texts,manual_labels,predicted_labels,_ = load_records_manual_process(project_name,1)
    original_size = len(texts)
    print(f"Size of extracted texts: {len(texts)}")
    last_train = load_config_field(project_name,"lastProjectTrain")
    print(f"Start at pos {last_train} and end at {len(texts)}")
    texts_train, texts_eval, manual_labels_train, manual_labels_eval = train_test_split(texts, manual_labels, test_size=0.2, random_state=42)

    labels = load_config_field(project_name,"labels")
    label2id = load_config_field(project_name,"label2id")
    id2label = load_config_field(project_name,"id2label")
    save_config_field(project_name,"lastProjectTrain",original_size)
    
    model_path = load_config_field(project_name,"model_path")
    tokenizer_path = load_config_field(project_name,"tokenizer_path")
    if (model_path != None) and (tokenizer_path != None):
        #tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        #model = RobertaCustomForTokenClassification(num_labels=len(labels))
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
    else:
        print(f"Num of labels: {len(labels)}")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(labels))
   
    # Tokenize and encode labels for both training and evaluation datasets
    tokenized_inputs_train = tokenizer(texts_train, max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels_train = tokenize_and_encode_labels(manual_labels_train, tokenized_inputs_train, label2id)

    tokenized_inputs_eval = tokenizer(texts_eval, max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels_eval = tokenize_and_encode_labels(manual_labels_eval, tokenized_inputs_eval, label2id)
    
    train_dataset = CustomDataset(tokenized_inputs_train, encoded_labels_train)
    eval_dataset = CustomDataset(tokenized_inputs_eval, encoded_labels_eval)




if __name__ == '__main__':
    main()


