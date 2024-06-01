# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.import sys
from pathlib import Path
import argparse
import random
import sys
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils import load_records_eval_set, load_config_field, save_config_field, manual_process
from src import train_model, make_prediction,evaluate_model
from all_datasets import load_txt_dataset,load_dataset

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch the automatic process of training, prediction and evaluation from a database")
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    args = parser.parse_args()
    return args

def handle_new_record(project_name,manual_labels,record_id):
        # Function called when a new record is processed (manually annotated)
        # This function decides if the document should be used for training or evaluation
        #ids,_,_,_,_ = load_records_eval_set(project_name,eval_set=0,manual_process=1)
        #on_training = load_config_field(project_name,"onTraining")
        num_records_trained = load_config_field(project_name,"numRecordsToTrain")
        eval_start = load_config_field(project_name,"startFrom")
        eval_percentage = load_config_field(project_name,"evalPercentage")
        training_steps = load_config_field(project_name,"trainingSteps")
        num_predictions = load_config_field(project_name,"numPredictions")
        print(f"Start eval at: {eval_start} eval Percentage: {eval_percentage}, trainingSteps: {training_steps}, numPred: {num_predictions} record id#: {record_id}")
        if record_id < eval_start:
            allocate_to_eval = 0
        else:
            allocate_to_eval =  (1 if random.randint(1, 100) <= eval_percentage else 0)
        if(allocate_to_eval == 0): 
            save_config_field(project_name,"numRecordsToTrain",num_records_trained+1)
        manual_process(project_name,manual_labels.split(','),record_id,allocate_to_eval) 
        #manual_process(project_name,manual_labels,record_id,allocate_to_eval) 


def process_new_records(project_name):
    # Launch training if required
    num_records_trained = load_config_field(project_name,"numRecordsToTrain")
    training_steps = load_config_field(project_name,"trainingSteps")
    on_training = load_config_field(project_name,"onTraining")
    if(num_records_trained  >= training_steps and not on_training):
        force_process(project_name)

def force_process(project_name):
    train_model(project_name)
    make_prediction(project_name,num_predictions=int(load_config_field(project_name,"numPredictions")))
    eval_set_path = load_config_field(project_name,"evalSetPath")
    if(eval_set_path):
        eval_records, eval_labels = load_dataset(eval_set_path)
        evaluate_model(project_name,eval_records,eval_labels,complete_evaluation=True)
    else:
        # Load all documents that need to be evaluate
        _, eval_records, eval_manual_labels,_,_ = load_records_eval_set(project_name,1,1)
        if(len(eval_records)>1):
            evaluate_model(project_name,eval_records,eval_manual_labels,complete_evaluation=False)

if __name__ == "__main__":
    args = parse_arguments()
    process_new_records(args.project_name,new_predictions=100)