# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import sys
from pathlib import Path
import argparse
import os
import shutil
from datetime import datetime
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import logging
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from utils import CustomDataset, load_config_field, save_config_field, set_manual_labels,load_records_eval_set,load_model_and_tokenizer
from models import tokenize_and_encode_labels, RobertaCustomForTokenClassification

# Remove TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for labels detection')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    return parser.parse_args()

def save_model_optimizer_scheduler(model, optimizer, scheduler, save_directory):
    # Create directory and saves the model, optimizer, and scheduler states
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    torch.save(optimizer.state_dict(), os.path.join(save_directory, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(save_directory, 'scheduler.pt'))

def load_model_optimizer_scheduler(model_path, optimizer, scheduler):
    # Load a pretrained model and states of the optimizer and scheduler
    model = RobertaCustomForTokenClassification.from_pretrained(model_path)
    optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pt')))
    scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pt')))
    return model, optimizer, scheduler


def train_model(project_name):
    # Training logic including data preparation, model configuration, and training loop
    # Everything is handle by this function that will retrieve all available documents and update the project
    save_config_field(project_name, "onTraining", True)
    ids, records, manual_labels, _, _ = load_records_eval_set(project_name, 0, 1)
    print(len(ids))
    print(records)
    if len(ids) < 1:
        return 0
    
    labels = load_config_field(project_name, "labels")
    label2id = load_config_field(project_name, "label2id")
    model_path = load_config_field(project_name, "model_path")
    tokenizer_path = load_config_field(project_name, "tokenizer_path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,tokenizer = load_model_and_tokenizer(project_name)
    model.to(device)
    tokenized_inputs_train = tokenizer(records, max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    encoded_labels_train = tokenize_and_encode_labels(manual_labels, tokenized_inputs_train, label2id)
    train_dataset = CustomDataset(tokenized_inputs_train, encoded_labels_train)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))
    for epoch in range(5):
        model.train()
        for batch in tqdm(train_dataloader,desc="Train batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {'input_ids': batch['input_ids'],'attention_mask': batch['attention_mask']}
            if 'labels' in batch:
                inputs['labels'] = batch['labels']
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    # Post-training operations including model saving and configuration updates
    if model_path:
        shutil.rmtree(model_path)
        shutil.rmtree(tokenizer_path)
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_save_path = f"projects/{project_name}/modelSave-{date}"
    tokenizer_save_path = f"projects/{project_name}/tokenizerSave-{date}"
    save_model_optimizer_scheduler(model, optimizer, scheduler, model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    save_config_field(project_name, "model_path", model_save_path)
    save_config_field(project_name, "onTraining", False)
    total_train_records = load_config_field(project_name, "totalTrain")
    save_config_field(project_name, "totalTrain", total_train_records + len(ids))
    num_records_untrain = load_config_field(project_name, "numRecordsToTrain")
    save_config_field(project_name, "numRecordsToTrain", num_records_untrain - len(ids))
    save_config_field(project_name,"tokenizer_path",tokenizer_save_path)
    set_manual_labels(project_name, 2, ids)

    
def main():
    args = parse_arguments()
    project_name = args.project_name
    train_model(project_name)

if __name__ == '__main__':
    main()
