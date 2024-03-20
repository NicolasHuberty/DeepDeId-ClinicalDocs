import sys
from pathlib import Path
import argparse
import os, shutil
import json
import sqlite3
from tqdm import tqdm
import random
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
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    parser.add_argument('--dataset', type=str, default="datasets/formatted/n2c2_2014/training3.tsv", help='Dataset to label with .tsv file supported')
    parser.add_argument('--labels', nargs="+", default=['I-COUNTRY', 'B-COUNTRY', 'I-CITY', 'B-CITY', 'I-STREET', 'B-STREET', 'I-ZIP', 'B-ZIP', 'I-STATE', 'B-STATE', 'I-HOSPITAL', 'B-HOSPITAL', 'I-LOCATION-OTHER', 'B-LOCATION-OTHER', 'I-ORGANIZATION', 'B-ORGANIZATION', 'I-DATE', 'B-DATE', 'I-DOCTOR', 'B-DOCTOR', 'I-PATIENT', 'B-PATIENT', 'I-MEDICALRECORD', 'B-MEDICALRECORD', 'I-PHONE', 'B-PHONE', 'I-FAX', 'B-FAX', 'I-IDNUM', 'B-IDNUM', 'I-DEVICE', 'B-DEVICE', 'B-AGE', 'I-AGE', 'I-PROFESSION', 'B-PROFESSION', 'I-EMAIL', 'B-EMAIL', 'I-USERNAME', 'B-USERNAME', 'I-URL', 'B-URL', 'B-BIOID', 'I-BIOID', 'I-HEALTHPLAN', 'B-HEALTHPLAN', 'O'])
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
                 (id INTEGER PRIMARY KEY, text TEXT, manual_labels TEXT, predicted_labels TEXT,eval_record BOOL, manual_process BOOL)''')
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

def create_project_from_scratch(project_name, dataset, labels_name):
    # Adjusted to place projects within a 'projects' directory
    projects_root = Path(__file__).resolve().parent.parent / 'projects'
    project_dir = projects_root / project_name
    print(f"Project directory: {project_dir}")
    create_database(project_name,labels_name)
    records, labels = load_dataset(dataset)
    label_colors = {label: "#{:02x}{:02x}{:02x}".format(random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)) for label in labels_name}
    label_colors["O"] = "#FFFFFF" 
    for i, record in enumerate(tqdm(records, desc="Processing records")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(project_name, i + 1, record, manual_labels, predicted_labels)
    save_config_field(project_name, "num_records", len(records))
    save_config_field(project_name,"labelColors",label_colors)
    save_config_field(project_name,"lastProjectTrain",0)

def main():
    args = parse_arguments()
    project_name = f"{args.project_name}"
    create_database(project_name,args.labels)
    shutil.copy(args.dataset, f"projects/{project_name}/original_dataset.tsv") 
    records,labels = load_dataset(args.dataset)
    label_colors = {label: "#{:02x}{:02x}{:02x}".format(random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)) for label in args.labels}
    label_colors["O"] = "#FFFFFF" 
    for i, record in enumerate(tqdm(records, desc="Processing records")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(project_name,i + 1, record, manual_labels, predicted_labels)
    save_config_field(project_name,"num_records",len(records))
    save_config_field(project_name,"labelColors",label_colors)
    save_config_field(project_name,"lastProjectTrain",1)

    
if __name__ == "__main__":
    main()