# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

from utils import readFormattedFile, dataset_statistics

# Parse datasets
parser = argparse.ArgumentParser(description='Provide statistics of a dataset')
parser.add_argument('--datasets_path', nargs="+", default=["wikiNER/train.tsv","wikiNER/test.tsv"], help='Path to the datasets')
args = parser.parse_args()


# Function to merge two dictionaries by summing values of matching keys
def merge_dictionaries(d1, d2):
    merged_dict = d1.copy() 
    for key, value in d2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict

label_distrib = {}
for dataset_path in args.datasets_path:
    print(f"-----------------------{dataset_path}-----------------------")
    tokens, labels,unique = readFormattedFile(dataset_path,mapping="None")
    tokens_size =[]
    for doc in tokens:
        tokens_size.append(len(doc))
    print(f"Token average per doc: {np.mean(tokens_size)}")
    labels = dataset_statistics(tokens,labels,unique)
    label_distrib = merge_dictionaries(label_distrib,labels)

# Merge the dictionaries
merged_data = {}
for label, count in label_distrib.items():
    # Remove BIO-Tag and combine
    if label != 'O':
        new_label = label[2:] if label.startswith(('B-', 'I-')) else label
        if new_label in merged_data:
            merged_data[new_label] += count
        else:
            merged_data[new_label] = count

# Creating a DataFrame from the merged data
df = pd.DataFrame(list(merged_data.items()), columns=['Label', 'Count'])
print(df)
df = df.sort_values(by='Count', ascending=False)

# Plotting
dataset_name = str(args.datasets_path[0].split('/')[0])
fig, ax = plt.subplots(figsize=(5.2, 4.05))
df.plot.bar(x='Label', y='Count', ax=ax, legend=False)
ax.set_title(f'{dataset_name} Label Distribution')
ax.set_xlabel('Label')
ax.set_ylabel('Count')
plt.subplots_adjust(bottom=0.14)
plt.show()