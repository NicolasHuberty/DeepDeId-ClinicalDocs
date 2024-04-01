import sys
from pathlib import Path
import argparse
import os, shutil
import json
import sqlite3
from tqdm import tqdm
import random
import csv
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import  store_record_with_labels,save_config_field
import logging
from load_dataset import load_dataset

logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a Project from scratch to identify specific labels on documents')
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    parser.add_argument('--dataset', type=str, default="datasets/formatted/n2c2_2014/training1Map.tsv", help='Dataset to label with .tsv file supported')
    parser.add_argument('--labels', nargs="+", default=["PERSON","LOCATION","DATE","ID"],help="Label to identify on documents")
    args = parser.parse_args()
    return args

def create_database(project_name, labels_to_predict):
    # Create sqlite database containing all records with all needed informations
    project_dir = f"projects/{project_name}"
    create_project_config(project_dir, labels_to_predict)
    try:
        conn = sqlite3.connect(f"{project_dir}/dataset.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, text TEXT, manual_labels TEXT, predicted_labels TEXT,eval_record BOOL,confidence REAL, manual_process INTEGER)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"An error occrurred while creating the project database: {e}")

def create_project_config(project_dir, labels):
    # Create json file with all needed informations about the project
    labels = ["O"] + [label for label in labels if label != "O"]
    config = {
        "project_name": Path(project_dir).name, 
        "num_records": 0,
        "model_path": None,
        "tokenizer_path": None,
        "labels": labels,
        "label2id": {label: i for i, label in enumerate(labels)},
        "id2label": {i: label for i, label in enumerate(labels)}
        }
    with open(f"{project_dir}/config_project.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

def get_pastel_color():
    # Create the colors of each label used for the frontend
    hue = random.randint(0, 360) 
    saturation = random.randint(30, 60) 
    lightness = random.randint(70, 90)
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def create_project_from_scratch(project_name, dataset, labels_name):
    # Create all configurations and database for the project
    projects_root = Path(__file__).resolve().parent.parent / 'projects'
    project_dir = projects_root / project_name
    os.makedirs(project_dir, exist_ok=True) 
    
    # Create the database with empty records
    create_database(project_name,labels_name)

    # Insert the shuffled records and labels
    records, labels = load_dataset(dataset)
    label_colors = {label: get_pastel_color() for label in labels_name}
    label_colors["O"] = "#FFFFFF"
    paired = list(zip(records, labels))
    random.shuffle(paired)
    records,labels = zip(*paired)
    for i, record in enumerate(tqdm(records, desc="Insert records on the database")):
        manual_labels = labels[i]
        predicted_labels = ['O'] * len(record)
        store_record_with_labels(project_name, i + 1, record, manual_labels, predicted_labels)
    
    # Save on the configuration file of the project the colors of the labels and total number of records
    save_config_field(project_name, "num_records", len(records))
    save_config_field(project_name,"labelColors",label_colors)
    save_config_field(project_name,"totalTrain",0)
    save_config_field(project_name,"numRecordsToTrain",0)

def main():
    args = parse_arguments()
    project_name = f"{args.project_name}"
    create_project_from_scratch(project_name,args.dataset,args.labels)
    shutil.copy(args.dataset, f"projects/{project_name}/original_dataset.tsv") 

if __name__ == "__main__":
    main()