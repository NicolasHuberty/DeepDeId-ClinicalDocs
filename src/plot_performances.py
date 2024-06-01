# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Argument parser setup
parser = argparse.ArgumentParser(description='Plot model performances from csv file')
parser.add_argument('--model_path', type=str, default="roberta-n2c2_-1-mapping_n2c2_removeBIO-epochs_5-batch_size_4", help='Name of the model for label retrieval and plotting')
args = parser.parse_args()

def plot_confusion_matrix(metrics):
    total_TP = metrics['True Positives'].sum()
    total_FP = metrics['False Positives'].sum()
    total_TN = metrics['True Negatives'].sum()
    total_FN = metrics['False Negatives'].sum()
    
    # Create a DataFrame for the confusion matrix
    confusion_data = pd.DataFrame({
        'Actual Positive': [total_TP, total_FN],
        'Actual Negative': [total_FP, total_TN]
    }, index=['Predicted Positive', 'Predicted Negative'])
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(confusion_data, annot=True, fmt="d", cmap='Greens', cbar=False)
    plt.title('Aggregated Confusion Matrix')

    # Adding metric labels
    for i, metric in enumerate(["TP", "FN", "FP", "TN"]):
        plt.text((i % 2) + 0.5, (i // 2) + 0.3, metric, va='center', ha='center', color='black', fontweight='bold', fontsize=14)

    plt.show()

def plot_precision_recall(metrics):
    labels = metrics['Label ID'].tolist()
    precision = metrics['Precision'].tolist()
    recall = metrics['Recall'].tolist()

    x = range(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, precision, width=0.4, label='Precision', color='b', align='center')
    plt.bar(x, recall, width=0.4, label='Recall', color='r', align='edge')
    plt.xlabel('Labels')
    plt.ylabel('Score')
    plt.title('Precision and Recall per Label')
    plt.xticks(x, labels, rotation='vertical')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_macro_averages(metrics):
    numeric_cols = metrics.select_dtypes(include=[float, int])
    macro_precision = numeric_cols['Precision'].mean()
    macro_recall = numeric_cols['Recall'].mean()
    macro_f1 = numeric_cols['F1 Score'].mean()
    labels = ['Precision', 'Recall', 'F1 Score']
    scores = [macro_precision, macro_recall, macro_f1]
    
    plt.figure(figsize=(8, 5))
    bars = sns.barplot(x=labels, y=scores, palette='viridis')
    plt.title('Macro Average Scores')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    # Add the text labels on each bar
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.4f'), 
                      (bar.get_x() + bar.get_width() / 2, 
                       bar.get_height()), ha='center', va='center',
                       size=12, xytext=(0, 8),
                       textcoords='offset points')

    plt.show()

# Load metrics from CSV
metrics = pd.read_csv(f"./results/{args.model_path}.csv")

# Plotting results
plot_confusion_matrix(metrics)
plot_precision_recall(metrics)
plot_macro_averages(metrics)
