# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import RobertaTokenizerFast,  BertTokenizerFast, BertForTokenClassification,AutoTokenizer,AutoModelForTokenClassification
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("dataset")
from sklearn.metrics import classification_report
from utils import load_config_field,save_config_field,load_records_eval_set,load_model_and_tokenizer,load_model_and_tokenizer
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification
import logging
from src import predict_and_align
from all_datasets import load_dataset
import pandas as pd
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import sqlite3
import shutil
def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model based on continual learning')
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    parser.add_argument("--supplementary_dataset", type=str, default="None", help="Name of the Project")
    args = parser.parse_args()
    return args


def store_evaluation_metrics(project_name, evaluation_metrics, trained_records_size, eval_set_type=False):
    conn = sqlite3.connect(f"projects/{project_name}/dataset.db") 
    cursor = conn.cursor()
    cursor.execute('SELECT AVG(confidence) FROM records WHERE confidence IS NOT NULL')
    avg_confidence = cursor.fetchone()[0]
    ids,_,_,_,_ = load_records_eval_set(project_name,eval_set=1,manual_process=1)
    dataframe_name = f"projects/{project_name}/{'eval_metrics_df.csv'}"
    try:
        df = pd.read_csv(dataframe_name, index_col='Trained Records Size')
    except FileNotFoundError:
        # Initialize a new DataFrame if no file found
        df = pd.DataFrame()  

    # Prepare new row
    new_row = pd.DataFrame({label: metrics['f1-score'] for label, metrics in evaluation_metrics.items() if label != 'accuracy'}, index=[trained_records_size])
    new_row['confidence'] = avg_confidence
    new_row['eval_size'] = len(ids)
    df = pd.concat([df, new_row], axis=0)

    # Save the updated DataFrame
    df.to_csv(dataframe_name, index=True, index_label='Trained Records Size')


def evaluate_performance(model, tokenizer, texts_eval, labels_eval, label2id, id2label):
    # Predict all documents and check the performance of the predictions compared to manual annotation
    predictions, true_labels = [], []
    for text, labels in tqdm(zip(texts_eval, labels_eval), total=len(texts_eval), desc="Predict documents for performance evaluation"):
        # Predict the document
        aligned_predicted_labels = predict_and_align(model, tokenizer, text, id2label)[0]

        predictions.extend([label2id.get(label, -1) for label in aligned_predicted_labels])
        true_labels.extend([label2id.get(label, -1) for label in labels])

    # Evaluate the performances
    unique_labels = sorted(set(true_labels + predictions))
    present_label2id = {label: idx for label, idx in label2id.items() if idx in unique_labels}
    labels = [label2id[label] for label in present_label2id.keys()]
    target_names = [label for label in present_label2id.keys()]
    return classification_report(true_labels, predictions, labels=labels, target_names=target_names, output_dict=True)


def evaluate_model(project_name,records,labels,complete_evaluation=False):
    # Retrieve all documents and configurations needed for the evaluation
    # complete_evaluation parameter allow to provide a supplementary file in tsv format with labels
    num_labels = len(load_config_field(project_name,"labels"))
    label2id = load_config_field(project_name,"label2id")
    id2label = load_config_field(project_name,"id2label")
    model,tokenizer = load_model_and_tokenizer(project_name)

    evaluation = evaluate_performance(model,tokenizer,records,labels,label2id,id2label)  
    # Retrieve the number of documents on which the model was train
    trained_records_size = len(load_records_eval_set(project_name,eval_set=0,manual_process=2)[0])
    store_evaluation_metrics(project_name, evaluation, trained_records_size,eval_set_type=complete_evaluation)
    save_config_field(project_name,"performances",evaluation)
    return evaluation

def main():
    args = parse_arguments()
    project_name = args.project_name

    # Retrieve documents that need to be evaluated and evaluate the model
    if(args.supplementary_dataset):
        records,labels = load_dataset(args.supplementary_dataset)
        evaluate_model(project_name,records,labels,True)
    else:
        _, records, labels,_,_ = load_records_eval_set(project_name,1,2)
        evaluate_model(project_name,records,labels,False)
    
    
if __name__ == '__main__':
    main()


