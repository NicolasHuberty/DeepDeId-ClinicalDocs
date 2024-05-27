# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
from all_datasets import load_dataset
import pandas as pd
import os

folder_path = "all_datasets/formatted/wikiNER"
try:
    os.makedirs(folder_path, exist_ok=True)
except Exception as e:
    pass

# Load the dataset
dataset = load_dataset("Jean-Baptiste/wikiner_fr")
tag_mapping = {
    0: 'O',
    1: 'LOC',
    2: 'PER',
    3: 'MISC',
    4: 'ORG'
}

with open(f"{folder_path}/train.tsv", "w", encoding="utf-8") as file:
    # Process each record in the dataset
    for record in dataset['train']:
        tokens = record['tokens']
        ner_tags = record['ner_tags']
        for token, tag in zip(tokens, ner_tags):
            string_tag = tag_mapping[tag]
            file.write(f"{token}\t{string_tag}\n")
        file.write('\n')

with open(f"{folder_path}/test.tsv", "w", encoding="utf-8") as file:
    # Process each record in the dataset
    for record in dataset['test']:
        tokens = record['tokens']
        ner_tags = record['ner_tags']
        for token, tag in zip(tokens, ner_tags):
            string_tag = tag_mapping[tag]
            file.write(f"{token}\t{string_tag}\n")
        file.write('\n')