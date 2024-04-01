import pandas as pd
import matplotlib.pyplot as plt
# Replace these file paths with the paths to your actual CSV files
file_path_1 = 'results/notTrainedRoberta.csv'
file_path_2 = 'results/RobertaContinuityx20.csv'

# Reading the CSV files
data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)

# Extracting the "Trained Records Size" and "I-CITY" columns
x1 = data1['Trained Records Size']
y1 = data1['macro avg']

x2 = data2['Trained Records Size']
y2 = data2['macro avg']

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label='Partial Evaluation done on 20% of the documents', marker='o')
plt.plot(x2, y2, label='Complete Evaluation done on a dedicated testing set', marker='x')

plt.xlabel('Trained Records Size')
plt.ylabel('F1-Score (macro average)')
plt.title('F1-Score Comparison for different Training Size')
plt.legend()
plt.grid(True)
plt.show()