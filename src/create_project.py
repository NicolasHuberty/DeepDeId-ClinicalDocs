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
from utils import CustomDataset, evaluate_model,store_record_with_labels,save_config_field
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
from load_dataset import load_txt_dataset,load_dataset
# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeIdentification of clinical documents using deep learning')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    parser.add_argument('--dataset', type=str, default="datasets/formatted/wikiNER/test.tsv", help='Dataset to label with .tsv file supported')
    parser.add_argument('--labels', nargs="+", default=["O","DATE"])
    args = parser.parse_args()
    return args

def create_database(project_name, labels_to_predict):
    project_dir = f"projects/{project_name}"
    print(f"Create database here: {project_dir}")
    os.makedirs(project_dir, exist_ok=True) 
    create_project_config(project_dir, labels_to_predict)
    conn = sqlite3.connect(f"{project_dir}/dataset.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY, text TEXT, manual_labels TEXT, predicted_labels TEXT, manual_process BOOL)''')
    conn.commit()
    conn.close()

def create_project_config(project_dir, labels):
    labels = ["O"] + [label for label in labels if label != "O"]
    config = {
        "project_name": Path(project_dir).name,  # Use the directory name as the project name
        "manual_annotations": 0,
        "num_records": 0,
        "model_path": None,
        "tokenizer_path": None,
        "labels": labels,
        "label2id": {label: i for i, label in enumerate(labels)},
        "id2label": {i: label for i, label in enumerate(labels)}
        }
    with open(f"{project_dir}/config_project.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

def create_project_from_scratch(project_name, dataset, labels):
    # Adjusted to place projects within a 'projects' directory
    projects_root = Path(__file__).resolve().parent.parent / 'projects'
    project_dir = projects_root / project_name
    print(f"Project directory: {project_dir}")
    create_database(project_name,labels)
    #shutil.copy(dataset,f"projects/{project_name}")
    records, labels = load_dataset(dataset)
    for i, record in enumerate(tqdm(records, desc="Processing records")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(project_name, i + 1, record, manual_labels, predicted_labels)
    save_config_field(project_name, "num_records", len(records))

def main():
    args = parse_arguments()
    project_name = f"{args.project_name}"
    create_database(project_name,args.labels)
    shutil.copy(args.dataset, f"projects/{project_name}/original_dataset.tsv") 
    records,labels = load_dataset(args.dataset)
    for i, record in enumerate(tqdm(records, desc="Processing records")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(project_name,i + 1, record, manual_labels, predicted_labels)
    save_config_field(project_name,"num_records",len(records))


    
if __name__ == "__main__":
    main()