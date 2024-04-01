import sys
from pathlib import Path
import argparse
import os
import shutil
from datetime import datetime
import torch
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
import logging
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import CustomDataset, load_config_field, save_config_field, set_manual_labels,load_records_eval_set
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification

# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for labels detection')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    return parser.parse_args()

def train_model(project_name):
    # Load the records and configuration to train the model

    save_config_field(project_name,"onTraining",True)
    ids, records, manual_labels,_,_ = load_records_eval_set(project_name,0,1)
    if(len(ids) < 1):
        return 0
    print(f"Train {ids}")
    labels = load_config_field(project_name, "labels")    
    label2id = load_config_field(project_name, "label2id")
    model_path = load_config_field(project_name, "model_path")
    tokenizer_path = load_config_field(project_name, "tokenizer_path")

    if model_path and tokenizer_path:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        model = RobertaCustomForTokenClassification.from_pretrained(model_path)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model = RobertaCustomForTokenClassification(num_labels=len(labels))

    tokenized_inputs_train = tokenizer(records, max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels_train = tokenize_and_encode_labels(manual_labels, tokenized_inputs_train, label2id)
    train_dataset = CustomDataset(tokenized_inputs_train, encoded_labels_train)


    training_args = TrainingArguments(save_strategy="no",output_dir=".trainingLogs",num_train_epochs=5,per_device_train_batch_size=4)
    trainer = Trainer( model=model,args=training_args,train_dataset=train_dataset)
    trainer.train()

    if(model_path):
        shutil.rmtree(model_path)
        shutil.rmtree(tokenizer_path)

    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f"projects/{project_name}/modelSave-{date}"
    tokenizer_name = f"projects/{project_name}/tokenizerSave-{date}"

    trainer.save_model(model_name)
    torch.save(model.state_dict(), os.path.join(model_name, 'pytorch_model.bin'))
    config_path = os.path.join(model_name, 'config.json')
    model.config.to_json_file(config_path)
    tokenizer.save_pretrained(tokenizer_name)
    save_config_field(project_name, "model_path", model_name)
    save_config_field(project_name, "tokenizer_path", tokenizer_name)
    save_config_field(project_name,"onTraining",False)
    total_train_records = load_config_field(project_name,"totalTrain")
    save_config_field(project_name,"totalTrain",total_train_records+len(ids))
    num_records_untrain = load_config_field(project_name,"numRecordsToTrain")
    save_config_field(project_name,"numRecordsToTrain",num_records_untrain-len(ids))
    set_manual_labels(project_name,2,ids)
    
def main():
    args = parse_arguments()
    project_name = args.project_name
    train_model(project_name)

if __name__ == '__main__':
    main()
