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
sys.path.append("datasets")
from utils import CustomDataset, evaluate_model,TextDataset,load_records_manual_process,store_predicted_labels, load_config_field,save_config_field
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
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

def predict_and_align(model, tokenizer, eval_text, label2id, id2label):
    """
    Evaluate the model on a document (eval_text), predict labels for each token,
    and align these predictions with the original text tokens. Handles long texts by automatically
    splitting them into manageable parts with tokenizer capabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the text, automatically handling long texts
    inputs = tokenizer(
        eval_text,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        stride=50,
        return_overflowing_tokens=True,
        is_split_into_words=True
    )

    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    word_ids_per_chunk = [inputs.word_ids(batch_index=i) for i in range(inputs['input_ids'].shape[0])]
    idsShape = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Process each chunk
    all_predictions = []
    for i in range(idsShape.shape[0]):
        # Prepare inputs for the model
        input_ids = inputs_device['input_ids'][i].unsqueeze(0)
        attention_mask = inputs_device['attention_mask'][i].unsqueeze(0)
        offset_mapping = inputs_device['offset_mapping'][i].unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.squeeze().tolist()

        # Align predictions with original text
        word_ids = word_ids_per_chunk[i]
        previous_word_idx = None
        for j, (word_idx, prediction_idx) in enumerate(zip(word_ids, predictions)):
            # Skip special tokens
            if word_idx is None:
                continue
            
            # Check for new word (ignore subword predictions)
            if word_idx != previous_word_idx:
                # Adjust prediction alignment according to offset_mapping
                offset_start = offset_mapping[0][j][0].item()
                offset_end = offset_mapping[0][j][1].item()                
                if offset_start == 0 and offset_end != 0:
                    # Convert prediction index to label and append
                    all_predictions.append((word_idx, id2label[str(prediction_idx)]))
                previous_word_idx = word_idx
    seen = set()
    aligned_labels = [label for idx, label in sorted(set(all_predictions)) if not (idx in seen or seen.add(idx))]

    return aligned_labels

def train_model(project_name,records,manual_labels,new_training_size):
    labels = load_config_field(project_name,"labels")
    label2id = load_config_field(project_name,"label2id")
    save_config_field(project_name,"lastProjectTrain",new_training_size)
    
    model_path = load_config_field(project_name,"model_path")
    tokenizer_path = load_config_field(project_name,"tokenizer_path")
    if (model_path != None) and (tokenizer_path != None):
        #tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        #model = RobertaCustomForTokenClassification(num_labels=len(labels))
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(labels))
   
    # Tokenize and encode labels for both training and evaluation datasets
    tokenized_inputs_train = tokenizer(records, max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels_train = tokenize_and_encode_labels(manual_labels, tokenized_inputs_train, label2id)
    print(f"Number of training data:{len(records)}")
    train_dataset = CustomDataset(tokenized_inputs_train, encoded_labels_train)
    training_args = TrainingArguments(
        save_strategy="no",
        output_dir="./.trainingLogs",
        num_train_epochs=5,
        per_device_train_batch_size=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    print(f"Launch the training...")
    trainer.train()
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f"projects/{project_name}/modelSave-{date}"
    tokenizer_name = f"projects/{project_name}/tokenizerSave-{date}"
    trainer.save_model(model_name)
    tokenizer.save_pretrained(tokenizer_name)
    save_config_field(project_name,"model_path",model_name)
    save_config_field(project_name, "tokenizer_path", tokenizer_name)
    save_config_field(project_name,"lastProjectTrain",new_training_size)
    return new_training_size


def main():
    args = parse_arguments()
    project_name = args.project_name
    texts,manual_labels,predicted_labels,_ = load_records_manual_process(project_name,1)

    new_training_size = len(texts)
    last_train = load_config_field(project_name,"lastProjectTrain")
    texts = texts[last_train:]
    manual_labels = manual_labels[last_train:]
    texts_train, texts_eval, manual_labels_train, manual_labels_eval = train_test_split(texts, manual_labels, test_size=0.2, random_state=42)
    print(f"Size of training documents: {len(texts)} Starting at pos: {last_train} and end at {new_training_size}")
    train_model(project_name,texts_train,manual_labels_train,new_training_size)

if __name__ == '__main__':
    main()


