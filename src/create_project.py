# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
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
sys.path.append("dataset")
from utils import  store_record_with_labels,save_config_field
import logging
from all_datasets import load_dataset,load_txt_dataset
logging.getLogger('tensorflow').setLevel(logging.ERROR)
def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a Project from scratch to identify specific labels on documents')
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    parser.add_argument('--dataset', type=str, default="all_datasets/formatted/n2c2_2014/training1Map.tsv", help='Dataset to label with .tsv file supported')
    parser.add_argument('--labels', nargs="+", default=["PERSON","LOCATION","DATE","ID"],help="Label to identify on documents")
    parser.add_argument("--model_name", type=str, default="roberta", help="Model that will be used (roberta, mBERT,camembert, xml-roberta)")
    parser.add_argument("--eval_percentage", type=int, default=30, help="Percentage of the documents that will be reserved for evaluation")
    parser.add_argument("--training_steps", type=int, default=10, help="Number of new documents between each training")
    parser.add_argument("--num_predictions", type=int, default=50, help="Number of new predictions for future documents")
    parser.add_argument("--start_from", type=int, default=20, help="Start to reserve eval_percentage after this number of documents")    
    args = parser.parse_args()
    return args
def create_database(project_name, labels_to_predict,model_name,eval_percentage,training_steps,num_predictions,start_from):
    # Create sqlite database containing all records with all needed informations
    project_dir = f"projects/{project_name}"
    create_project_config(project_dir, labels_to_predict,model_name,eval_percentage,training_steps,num_predictions,start_from)
    try:
        conn = sqlite3.connect(f"{project_dir}/dataset.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, text TEXT, manual_labels TEXT, predicted_labels TEXT,eval_record BOOL,confidence REAL, manual_process INTEGER)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"An error occrurred while creating the project database: {e}")

def create_project_config(project_dir, labels,model_name,eval_percentage,training_steps,num_predictions,start_from):
    # Create json file with all needed informations about the project
    labels = ["O"] + [label for label in labels if label != "O"]
    config = {
        "project_name": Path(project_dir).name, 
        "num_records": 0,
        "model_path": None,
        "tokenizer_path": None,
        "labels": labels,
        "label2id": {label: i for i, label in enumerate(labels)},
        "id2label": {i: label for i, label in enumerate(labels)},
        "modelName": model_name,
        "evalPercentage": eval_percentage,
        "trainingSteps" : training_steps,
        "numPredictions": num_predictions,
        "startFrom": start_from
        }
    with open(f"{project_dir}/config_project.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

def get_pastel_color():
    # Create the colors of each label used for the frontend
    hue = random.randint(0, 360) 
    saturation = random.randint(30, 60) 
    lightness = random.randint(70, 90)
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def create_project_from_scratch(project_name, dataset, labels_name, model_name,eval_percentage,training_steps,num_predictions,start_from):
    # Create all configurations and database for the project
    projects_root = Path(__file__).resolve().parent.parent / 'projects'
    os.makedirs(projects_root, exist_ok=True)
    project_dir = projects_root / project_name
    os.makedirs(project_dir, exist_ok=True) 
    
    # Create the database with empty records
    create_database(project_name,labels_name,model_name,eval_percentage,training_steps,num_predictions,start_from)

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
    save_config_field(project_name,"onTraining",False)
def main():
    args = parse_arguments()
    project_name = f"{args.project_name}"
    create_project_from_scratch(project_name,args.dataset,args.labels,args.model_name,args.eval_percentage,args.training_steps,args.num_predictions,args.start_from)
    shutil.copy(args.dataset, f"projects/{project_name}/original_dataset.tsv") 

if __name__ == "__main__":
    main()