import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_f1(metrics, labels):
    print(metrics)
    colors = ['#2ca02c']    
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
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_ylim(0, 1.1) 
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey') 
    ax.set_axisbelow(True)  

    plt.tight_layout()
    plt.show()