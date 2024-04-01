import sys
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast,  BertTokenizerFast, BertForTokenClassification
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import store_predicted_labels, load_config_field, load_records_eval_set
from models import RobertaCustomForTokenClassification
import logging
from load_dataset import load_txt_dataset,load_dataset
# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict documents to help manual annotations')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of records to predict")
    args = parser.parse_args()
    return args

def predict_and_align(model, tokenizer, eval_text, id2label):
    # Predict a document and map the labels to the original tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the text, automatically handling long texts with stride for overlap
    inputs = tokenizer(eval_text,return_offsets_mapping=True,padding=True,truncation=True,return_tensors="pt",max_length=512,stride=50,return_overflowing_tokens=True,is_split_into_words=True)
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    word_ids_per_chunk = [inputs.word_ids(batch_index=i) for i in range(inputs['input_ids'].shape[0])]
    all_predictions = []
    word_predictions_confidence = {}

    # Process each chunk of the tokenizer
    for i in range(inputs['input_ids'].shape[0]):
        input_ids = inputs_device['input_ids'][i].unsqueeze(0)
        attention_mask = inputs_device['attention_mask'][i].unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs["logits"]
        softmax_logits = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        confidences = torch.max(softmax_logits, dim=-1).values

        word_ids = word_ids_per_chunk[i]
        previous_word_idx = None
        for j, (word_idx, prediction_idx, confidence) in enumerate(zip(word_ids, predictions.squeeze().tolist(), confidences.squeeze().tolist())):
            if word_idx is None:  
                # Skip special tokens
                continue

            # Update predictions only if this is the first prediction for the word or if the confidence is higher than a previous prediction
            if word_idx != previous_word_idx and (word_idx not in word_predictions_confidence or confidence > word_predictions_confidence[word_idx][1]):
                word_predictions_confidence[word_idx] = (id2label[str(prediction_idx)], confidence)
            previous_word_idx = word_idx

    sorted_predictions = sorted(word_predictions_confidence.items(), key=lambda x: x[0])
    all_predictions = [pred[0] for _, pred in sorted_predictions]
    average_confidence = sum(pred[1] for _, pred in sorted_predictions) / len(sorted_predictions) if sorted_predictions else 0

    # Check if the labels correctyly match the original text
    if len(eval_text) != len(all_predictions):
        print(f"Warning: Mismatch in lengths. Text words: {len(eval_text)}, Predictions: {len(all_predictions)}")
    return all_predictions, average_confidence


def make_prediction(project_name,num_predictions=100):
    # Predict all documents that need to be predicted on a database
    new_records_ids,new_records,_,_,_ = load_records_eval_set(project_name,eval_set=0,manual_process=0)
    new_records = new_records[:num_predictions]
    labels = load_config_field(project_name,"labels")
    id2label = load_config_field(project_name,"id2label")
    model_path = load_config_field(project_name,"model_path")
    tokenizer_path = load_config_field(project_name,"tokenizer_path")  
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    model = RobertaCustomForTokenClassification.from_pretrained(model_path)
    #tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    #model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
    all_aligned_predicted_labels = []
    for document in new_records:
            # Add the prediction of a document
            all_aligned_predicted_labels.append(predict_and_align(model, tokenizer, document, id2label))
    # Store all predicted labels on the database
    store_predicted_labels(project_name,new_records_ids,all_aligned_predicted_labels)
    
def main():
    args = parse_arguments()
    make_prediction(args.project_name,num_predictions=args.num_records)
    
if __name__ == '__main__':
    main()


