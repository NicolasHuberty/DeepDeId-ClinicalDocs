import sys
from pathlib import Path
import json
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import CustomDataset,TextDataset,load_records_manual_process,store_eval_records, load_config_field,save_config_field,should_allocate_to_evaluation,load_records_eval_set, load_records_in_range
from src import train_model, make_prediction,evaluate_model

def process_new_records(project_name,new_predictions=100):
    train_records_start = load_config_field(project_name,"lastProjectTrain")
    train_records_end = load_config_field(project_name,"manual_annotations")
    print(f"Process new records starting at pos {train_records_start} and ending at pos {train_records_end}")
    
    evaluation_index = should_allocate_to_evaluation((train_records_end-train_records_start),train_records_start)
    new_records_ids = np.arange(train_records_start+1,train_records_end+1,1)
    print(new_records_ids)
    store_eval_records(project_name,new_records_ids,evaluation_index)
    
    train_records, train_labels,_,_ = load_records_eval_set(project_name,0,train_records_start+1,train_records_end)
    last_record_train = train_model(project_name,train_records,train_labels,train_records_end)
    
    new_records,_,_,_ = load_records_in_range(project_name,last_record_train+1,last_record_train + new_predictions +1 )
    new_records = new_records[:new_predictions]
    train_records_end = load_config_field(project_name,"manual_annotations")
    make_prediction(project_name,new_records,last_record_train+1)
    print("Retrieve evaluation records...")
    eval_records, eval_manual_labels,_,_ = load_records_eval_set(project_name,1,0,train_records_end)
    if(len(eval_records)>1):
        print("Launch evaluation of the model...")
        print(evaluate_model(project_name,eval_records,eval_manual_labels))
    

if __name__ == "__main__":
    process_new_records("hugeTest")