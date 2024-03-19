import sys
from pathlib import Path
import argparse
import os, shutil
import numpy as np
import json
import sqlite3
from tqdm import tqdm
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
    Evaluate the model on a single document (eval_text), predict labels for each token,
    and align these predictions with the original text tokens.

    Parameters:
    - model: The pre-trained model used for token classification.
    - tokenizer: Tokenizer used to tokenize the input text.
    - eval_text: A list of strings, each representing a word in the evaluation text.
    - label2id: A dictionary mapping label names to their corresponding IDs.
    - id2label: A dictionary mapping label IDs back to their corresponding names.

    Returns:
    - aligned_labels: A list of labels aligned with the input eval_text.
    """ 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # Tokenize the input text
    inputs = tokenizer(eval_text, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, return_tensors="pt")
    word_ids = inputs.word_ids(batch_index=0)  # Get word_ids to align subtokens with original tokens
    inputs = {k: v.to(device) for k, v in inputs.items()}

    offset_mapping = inputs.pop("offset_mapping")  # Remove offset_mapping from inputs
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.squeeze().tolist()  # Convert to list

    # Initialize variables for alignment
    aligned_labels = []

    previous_word_idx = None
    for word_idx, prediction_idx in zip(word_ids, predictions):
        if word_idx is None:
            continue  # Special tokens have no associated word
        if word_idx != previous_word_idx:  # Start of a new word
            aligned_labels.append(id2label[str(prediction_idx)])
        previous_word_idx = word_idx

    return aligned_labels


def main():
    args = parse_arguments()
    project_name = args.project_name
    texts,manual_labels,predicted_labels,_ = load_records_manual_process(project_name,1)
    print(f"Size of training documents: {len(texts)}")
    save_config_field(project_name,"manual_annotations",len(texts))
    labels = load_config_field(project_name,"labels")
    label2id = load_config_field(project_name,"label2id")
    id2label = load_config_field(project_name,"id2label")
    #tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    #model = RobertaCustomForTokenClassification(num_labels=len(labels))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(labels))
    tokenized_inputs = tokenizer(texts,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels = tokenize_and_encode_labels(manual_labels, tokenized_inputs,label2id)

    train_dataset = CustomDataset(tokenized_inputs, encoded_labels)
    training_args = TrainingArguments(
        save_strategy="no",
        output_dir="./.trainingLogs",
        num_train_epochs=5,
        per_device_train_batch_size=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print(f"Launch the training...")
    trainer.train()
    trainer.save_model("backup")
    eval_texts,_,_,_ = load_records_manual_process(project_name,0)
    eval_texts = eval_texts[0:args.num_evals]
    print(f"Number of eval texts: {len(eval_texts)}")
    all_aligned_predicted_labels = []
    for i, text in enumerate(eval_texts):
            aligned_predicted_labels = predict_and_align(model, tokenizer, text, label2id, id2label)
            if(len(text) != len(aligned_predicted_labels)):
                print(f"Record {i} size misamatched: \n {text} \n {aligned_predicted_labels}")
            all_aligned_predicted_labels.append(aligned_predicted_labels)
    from_predicted = load_config_field(project_name,"manual_annotations")
    end_predicted = load_config_field(project_name,"num_records")
    records_ids = np.arange(from_predicted+1,from_predicted+1+len(eval_texts),1)
    print(aligned_predicted_labels)
    print(f"Number of label list: {len(all_aligned_predicted_labels)}")
    store_predicted_labels(project_name,records_ids,all_aligned_predicted_labels)
if __name__ == '__main__':
    main()


