import sys
from pathlib import Path
import json
import numpy as np
import sqlite3
import argparse
from sklearn.model_selection import train_test_split
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import CustomDataset,TextDataset,load_records_manual_process,store_eval_records, load_config_field,save_config_field,should_allocate_to_evaluation,load_records_eval_set, load_records_in_range, set_manual_labels
from src import train_model, make_prediction,evaluate_model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeIdentification of clinical documents using deep learning')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    args = parser.parse_args()
    return args

def process_new_records(project_name,new_predictions=100):
    train_records_start = load_config_field(project_name,"lastProjectTrain")
    train_records_end = load_config_field(project_name,"manual_annotations")
    print(f"Process new records starting at pos {train_records_start} and ending at pos {train_records_end}")
    

    
    train_ids, train_records, train_labels,_,_ = load_records_eval_set(project_name,"NULL",1)
    print(f"Train ids: {train_ids}")
    evaluation_index = should_allocate_to_evaluation(train_records,train_ids)
    store_eval_records(project_name,train_ids,evaluation_index)

    last_record_train = train_model(project_name,train_records,train_labels,train_records_end)
    set_manual_labels(project_name,2,train_ids) #Do not need to be processed anymore

    new_records_ids,new_records,_,_,_ = load_records_eval_set(project_name,"NULL",0)
    new_records = new_records[:new_predictions]
    train_records_end = load_config_field(project_name,"manual_annotations")
    print(f"Last record train val: {last_record_train}")
    make_prediction(project_name,new_records,new_records_ids)


    print("Retrieve evaluation records...")
    eval_ids, eval_records, eval_manual_labels,_,_ = load_records_eval_set(project_name,1,2)
    if(len(eval_records)>1):
        print("Launch evaluation of the model...")
        print(evaluate_model(project_name,eval_records,eval_manual_labels))
    train_records_end = load_config_field(project_name,"manual_annotations")

if __name__ == "__main__":
    args = parse_arguments()
    process_new_records(args.project_name)