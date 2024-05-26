import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)

# Set a style
plt.style.use('seaborn-v0_8-whitegrid')
file_path_1 = 'results/result11c-wikiNER.csv'
data1 = pd.read_csv(file_path_1)
data1 = data1.sort_values(by='Trained Records Size')
x1 = data1['Trained Records Size']
y1 = data1['macro avg']


file_path_2 = 'results/result11c-wikiNER.csv' 
data2 = pd.read_csv(file_path_2)
data2 = data2.sort_values(by='Trained Records Size')
x2 = data2['Trained Records Size']
y2 = data2['macro avg']

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label='Continuous Model From Scratch', marker='o', color='royalblue', markersize=6, linewidth=2)
plt.plot(x2, y2, label='Transfer learning', marker='x', color='crimson', markersize=4, linewidth=2)
plt.xlabel('Trained Iterations (total number of documents)', fontsize=12)
plt.ylabel('F1-Score (macro avg)', fontsize=12)
plt.title('Comparison of Continuous From Scratch vs. Transfer Learning', fontsize=14, fontweight='bold')
plt.legend(title='Model Update Strategy', fontsize=10,frameon=True)

plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')

model_details = (
                 "Epochs: 5\n"
                 "Batch Size: 4\n"
                 "Dataset: i2b2\n"
                )
plt.figtext(0.85, 0.25, model_details, ha="center", fontsize=9, bbox={"boxstyle":"round", "facecolor":"white", "alpha":0.5})
plt.show()
