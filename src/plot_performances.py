import argparse
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification
from utils import plot_f1_recall_precision, plt_fp_fn, plot_global_metrics, plot_label_distribution, complete_plot,plot_f1

parser = argparse.ArgumentParser(description='Plot model performances from CSV')
parser.add_argument('--model_name', type=str,default="pubMedBERT-n2c2_2014n2c2_2014-mapping_n2c2_removeBIO-epochs_5-batch_size_4" , help='Name of the model for label retrieval and plotting')
args = parser.parse_args()


def plot_results(metrics, labels, model_name):
    metrics_dict = {
        'support_per_label': metrics['Support'].to_list(),
        'TP': metrics['True Positives'].to_list(),
        'FP': metrics['False Positives'].to_list(),
        'FN': metrics['False Negatives'].to_list(),
        'TN': metrics['True Negatives'].to_list(),
        'precision_per_label': metrics['Precision'].to_list(),
        'recall_per_label': metrics['Recall'].to_list(),
        'f1_per_label': metrics['F1 Score'].to_list(),
    }
    plot_f1(metrics_dict, labels, model_name)



metrics = pd.read_csv(f"./results/{args.model_name}.csv")
model = BertForTokenClassification.from_pretrained(f"./model_save/{args.model_name}")
labels = [label for _, label in sorted(model.config.id2label.items())]
plot_results(metrics, labels,args.model_name)
