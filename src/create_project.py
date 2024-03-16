import sys
from pathlib import Path
import argparse
import os, shutil
import json
import sqlite3
from tqdm import tqdm
import torch
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import CustomDataset, evaluate_model
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
from tokenizer import tokenize_text
from load_dataset import load_txt_dataset,load_dataset
# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeIdentification of clinical documents using deep learning')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    parser.add_argument('--dataset', type=str, default="datasets/formatted/n2c2_2014/training1.txt", help='Dataset to label with .tsv file supported')
    parser.add_argument('--labels', nargs="+", default=["PERSON", "LOCATION","ORG","B-DATE"], help='Labels to recognize on the dataset')
    args = parser.parse_args()
    return args

def create_database(project_name,labels_to_predict):
    if(os.path.exists(f"{project_name}")):
        shutil.rmtree(f"{project_name}")
    os.makedirs(project_name, exist_ok=True)
    create_project_config(project_name,labels_to_predict)
    conn = sqlite3.connect(f"{project_name}/dataset.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY, text TEXT, manual_labels TEXT, predicted_labels TEXT,manual_process BOOL)''')
    conn.commit()
    conn.close()


def create_project_config(project_name, labels):
    labels = ["O"] + [label for label in labels if label != "O"]
    config = {
        "project_name": project_name,
        "manual_annotations": 0,
        "model_path": None,
        "tokenizer_path": None,
        "labels": labels,
        "label2id": {label: i for i, label in enumerate(labels)},
        "id2label": {i: label for i, label in enumerate(labels)}
        }
    with open(f"{project_name}/config_project.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

def store_record_with_labels(record_id, text, manual_labels, predicted_labels):
    conn = sqlite3.connect('customProject/dataset.db')
    c = conn.cursor()
    text_str = ' '.join(text)  # Reconstruire la phrase à partir de la liste des mots
    manual_labels_str = ','.join(map(str, manual_labels))  # Assumant que manual_labels est déjà une liste de labels correspondants
    predicted_labels_str = ','.join(map(str, predicted_labels))
    c.execute('''INSERT INTO records (id, text, manual_labels, predicted_labels,manual_process)
                 VALUES (?, ?, ?, ?, ?)''',
                 (record_id, text_str, manual_labels_str, predicted_labels_str,False))
    conn.commit()
    conn.close()


def main():
    args = parse_arguments()
    create_database(args.project_name,args.labels)
    records,labels = load_dataset(args.dataset)
    print(records[0])
    for i, record in enumerate(tqdm(records, desc="Processing records")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(i + 1, record, manual_labels, predicted_labels)

if __name__ == "__main__":
    main()