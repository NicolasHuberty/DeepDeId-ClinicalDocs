import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_f1(metrics, labels, model_path):
    colors = ['#2ca02c'] 
    model_details = model_path.split('-')
    dataset_name = model_details[1]
    mapping_method = model_details[2].split('_')[1]
    epochs = model_details[3].split('_')[1]
    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(len(labels))
    bar_width = 0.5 

    # Plot only F1 bars
    f1_bars = ax.bar(index, metrics['F1 Score'], bar_width, label='F1 Score', color=colors[0])

    def add_value_annotations(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        rotation=90,
                        ha='center', va='bottom', fontsize=10) 

    add_value_annotations(f1_bars)

    ax.set_xlabel('Labels', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'F1 Score by Label\nDataset: {dataset_name}, Mapping: {mapping_method}, Epochs: {epochs}', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_ylim(0, 1.1) 
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey') 
    ax.set_axisbelow(True)  

    plt.tight_layout()
    plt.savefig(f"./results/plots/{model_path}_f1.png", bbox_inches='tight')
    plt.show()