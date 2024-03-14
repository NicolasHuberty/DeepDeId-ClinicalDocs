import sys
from pathlib import Path
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, CamembertTokenizerFast, CamembertForTokenClassification
from transformers import BertTokenizerFast, BertForTokenClassification,RobertaTokenizerFast, RobertaForTokenClassification, DistilBertTokenizerFast, DistilBertForTokenClassification, AlbertTokenizerFast, AlbertForTokenClassification,AutoTokenizer, RobertaForTokenClassification
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from models import tokenize_and_encode_labels, compute_metrics,RobertaCustomForTokenClassification
from utils import readFormattedFile, CustomDataset, evaluate_model,dataset_statistics,plot_f1
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Define the dataset sizes to test
dataset_sizes = [2000,5000]
f1_scores_df = pd.DataFrame()

for dataset_size in dataset_sizes:
    # Retrieve all arguments
    parser = argparse.ArgumentParser(description='DeIdentidication of clinical documents using deep learning')
    parser.add_argument("--train_set",nargs="+",default=["wikiNER/train.tsv"],help="List of datasets for training")
    parser.add_argument('--eval_set', type=str, default="wikiNER/test.tsv", help='Dataset for testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--mapping',type=str, default="None", help="None for no mapping, name of the file otherwise")
    parser.add_argument('--dataset_size',type=str,default=str(dataset_size))
    args = parser.parse_args()

    print(f"Launch training with bert-base-uncased with: {args.train_set} eval set: {args.eval_set} epochs: {args.epochs} batch size: {args.batch_size}")

    # Retrieve the correct mapping dictionnary
    train_set = ",".join(args.train_set)

    # Process training set
    tokens, labels, unique_labels_train = [], [], set()
    train_dataset_string = ""
    for dataset in args.train_set:
        t, l, unique_labels = readFormattedFile(dataset,args.mapping)
        tokens.extend(t)
        labels.extend(l)
        unique_labels_train.update(unique_labels)
        train_dataset_string += dataset.split("/")[0]
        print(f"Retrieve training dataset {dataset} Labels: {unique_labels_train} with mapping: {args.mapping}")
    tokens = tokens[0:int(args.dataset_size)]
    labels = labels[0:int(args.dataset_size)]
    print(f"Retrieve training dataset {args.train_set} Labels: {unique_labels_train} with mapping: {args.mapping}")
    tokens = tokens[0:int(args.dataset_size)]
    labels = labels[0:int(args.dataset_size)]
    print(f"Number of records to train: {len(tokens)}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(unique_labels_train))
    model.config.name = f"roberta-{train_dataset_string}_{args.dataset_size}-mapping_{args.mapping}-epochs_{args.epochs}-batch_size_{args.batch_size}"
    model.config.label2id = {label: id for id, label in enumerate(unique_labels_train)}
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    # Tokenize and encode labels for training and evaluation set
    print(f"Tokenize the tokens and process labels...")
    tokenized_train_inputs = tokenizer(tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
    encoded_train_labels = tokenize_and_encode_labels(labels, tokenized_train_inputs,model.config.label2id)
    print(f"Transform datasets tokenized in customDataset...")
    train_dataset = CustomDataset(tokenized_train_inputs, encoded_train_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100000,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics,
    )

    # Launch the training of the model
    print(f"Launch the training for {args.epochs} epochs...")
    trainer.train()

    # Evaluate the performance of the model
    if(args.eval_set != "None"):
        print(f"Retrieve eval dataset {args.eval_set}")
        eval_tokens, eval_labels, eval_unique_labels = readFormattedFile(args.eval_set,args.mapping)
        tokenized_eval_inputs = tokenizer(eval_tokens,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_tensors="pt")
        encoded_eval_labels = tokenize_and_encode_labels(eval_labels, tokenized_eval_inputs,model.config.label2id)
        eval_dataset = CustomDataset(tokenized_eval_inputs, encoded_eval_labels)

        metrics = evaluate_model(model, eval_dataset,'cuda',tokenizer)
        label_names = [f"Label_{i}" for i in range(len(metrics['f1_per_label']))] 
        if f1_scores_df.empty:
            f1_scores_df = pd.DataFrame(columns=['Dataset Size'] + [f'F1_{label}' for label in label_names])

        current_row = {'Dataset Size': args.dataset_size}
        current_row.update({f'F1_{label}': metrics['f1_per_label'][i] for i, label in enumerate(label_names)})
        row_df = pd.DataFrame([current_row])
        f1_scores_df = pd.concat([f1_scores_df, row_df], ignore_index=True)

# Save the F1 scores and plot the result
f1_scores_df.to_csv("results/sizePerformances-dataset.csv", index=False)
mean_f1_scores = f1_scores_df.iloc[:, 1:].mean(axis=1)
mean_f1_scores_df = pd.DataFrame({
    'Dataset Size': f1_scores_df['Dataset Size'],
    'Mean F1 Score': mean_f1_scores
})
plt.plot(mean_f1_scores_df['Dataset Size'], mean_f1_scores_df['Mean F1 Score'], label='Mean F1 Score')
plt.xlabel('Dataset Size')
plt.ylabel('Mean F1 Score')
plt.title('Mean F1 Score vs Dataset Size')
plt.legend()
plt.show()