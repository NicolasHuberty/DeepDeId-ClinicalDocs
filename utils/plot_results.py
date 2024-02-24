import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_f1(metrics, labels, model_path):
    colors = ['#2ca02c']  # Color for F1 score
    model_details = model_path.split('-')
    dataset_name = model_details[1]
    mapping_method = model_details[2].split('_')[1]
    epochs = model_details[3].split('_')[1]
    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(len(labels))
    bar_width = 0.5  # Adjusted for a single set of bars

    # Plot only F1 bars
    f1_bars = ax.bar(index, metrics['f1_per_label'], bar_width, label='F1 Score', color=colors[0])

    def add_value_annotations(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        rotation=90,
                        ha='center', va='bottom', fontsize=10)  # fontsize increased for readability

    add_value_annotations(f1_bars)

    ax.set_xlabel('Labels', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'F1 Score by Label\nDataset: {dataset_name}, Mapping: {mapping_method}, Epochs: {epochs}', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend

    ax.set_ylim(0, 1.1)  # Set y-axis limit to prevent cutting off annotations
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Add grid
    ax.set_axisbelow(True)  # Ensure grid is behind the bars

    plt.tight_layout()
    plt.savefig(f"./plots/{model_path}_f1.png", bbox_inches='tight')  # Adjust filename
    plt.show()
def plot_f1_recall_precision(metrics, labels, model_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    model_details = model_path.split('-')
    dataset_name = model_details[1]
    mapping_method = model_details[2].split('_')[1]
    epochs = model_details[3].split('_')[1]
    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(len(labels))
    bar_width = 0.25

    precision_bars = ax.bar(index - bar_width, metrics['precision_per_label'], bar_width, label='Precision', color=colors[0])
    recall_bars = ax.bar(index, metrics['recall_per_label'], bar_width, label='Recall', color=colors[1])
    f1_bars = ax.bar(index + bar_width, metrics['f1_per_label'], bar_width, label='F1 Score', color=colors[2])

    def add_value_annotations(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        rotation=90,
                        ha='center', va='bottom', fontsize=10)  # fontsize increased for readability

    add_value_annotations(precision_bars)
    add_value_annotations(recall_bars)
    add_value_annotations(f1_bars)

    ax.set_xlabel('Labels', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision, Recall, F1 Score by Label\nDataset: {dataset_name}, Mapping: {mapping_method}, Epochs: {epochs}', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)  # 'ha' aligns text correctly after rotation
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend out of the plot

    ax.set_ylim(0, 1.1)  # Set y-axis limit to prevent cutting off annotations
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Add grid
    ax.set_axisbelow(True)  # Ensure grid is behind the bars

    plt.tight_layout()
    plt.savefig(f"./plots/{model_path}.png", bbox_inches='tight')  # Save with tight bounding box
    plt.show()

def plt_fp_fn(metrics, labels, model_path):
    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(len(labels))
    bar_width = 0.35

    fp = ax.bar(index - bar_width / 2, metrics['FP'], bar_width, label='FP', color='green')
    fn = ax.bar(index + bar_width / 2, metrics['FN'], bar_width, label='FN', color='grey')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('False Positives and False Negatives by Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()
def plot_global_metrics(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_names = ['Precision', 'Recall', 'F1']
    metrics_values = [metrics['global_precision'], metrics['global_recall'], metrics['global_f1']]

    bars = ax.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'salmon'])

    ax.set_ylabel('Score')
    ax.set_title('Global Precision, Recall, F1-Score')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_label_distribution(metrics, labels, model_path):
    true_label_counts = np.sum(metrics['support_per_label'], axis=0)
    predicted_label_counts = np.sum([metrics['TP'], metrics['FP']], axis=0)

    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(len(labels))
    bar_width = 0.35

    true_dist = ax.bar(index - bar_width / 2, true_label_counts, bar_width, label='True Labels', color='lightblue')
    pred_dist = ax.bar(index + bar_width / 2, predicted_label_counts, bar_width, label='Predicted Labels', color='orange')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('Label Distribution: True vs Predicted')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_performance_metrics(metrics, labels, model_path):
    # Extract TP, FP, FN, TN for plotting
    TP = metrics['TP']
    FP = metrics['FP']
    FN = metrics['FN']
    TN = metrics['TN']
    
    n_groups = len(labels)
    fig, ax = plt.subplots(figsize=(15, 8))  # Adjusted for better visibility
    index = np.arange(n_groups)
    bar_width = 0.15  # Adjusted for spacing
    
    # Creating the bars
    #rects1 = ax.bar(index - bar_width*1.5, TP, bar_width, label='TP', color='g')
    rects2 = ax.bar(index - bar_width/2, FP, bar_width, label='FP', color='r')
    rects3 = ax.bar(index + bar_width/2, FN, bar_width, label='FN', color='b')
    #rects4 = ax.bar(index + bar_width*1.5, TN, bar_width, label='TN', color='y')

    # Adding annotations to each bar
    def add_annotations(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    #add_annotations(rects1)
    add_annotations(rects2)
    add_annotations(rects3)
    #add_annotations(rects4)

    # Setting the labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of TP, FP, FN, TN by Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45) 
    ax.legend()
    plt.tight_layout()
    plt.show()

def show_mismatch_labels(metrics,labels):
    print("Mismatched Tokens:")
    for info in metrics['mismatched_tokens_info']:
        print(f"Token: {info['token']}, True Label: {info['true_label']}, Predicted Label: {info['predicted_label']}")
def plot_gpt(metrics, label_list):
    # Plotting precision, recall, and F1-score by label
    precision = metrics['precision_per_label']
    recall = metrics['recall_per_label']
    f1_score = metrics['f1_per_label']

    df_perf = pd.DataFrame({
        'Label': label_list,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })

    df_perf_melted = df_perf.melt('Label', var_name='Metrics', value_name='Values')
    sns.barplot(x='Label', y='Values', hue='Metrics', data=df_perf_melted)
    plt.xticks(rotation=45, ha="right")
    plt.title("Performance Metrics by Label")
    plt.show()

    # Plotting TP, FP, FN, and support for each label
    TP = metrics['TP']
    FP = metrics['FP']
    FN = metrics['FN']
    # Generating support counts from TP, FP, FN
    support_counts = [metrics['support_per_label'].get(i, 0) for i in sorted(metrics['support_per_label'])]

    df_counts = pd.DataFrame({
        'Label': label_list,
        'True Positives': TP,
        'False Positives': FP,
        'False Negatives': FN,
        'Support': support_counts
    })

    df_counts_melted = df_counts.melt('Label', var_name='Metrics', value_name='Counts')
    sns.barplot(x='Label', y='Counts', hue='Metrics', data=df_counts_melted)
    plt.xticks(rotation=45, ha="right")
    plt.title("Counts of TP, FP, FN, Support by Label")
    plt.show()
def plot_sentence_length_distribution(tokenized_inputs):
    lengths = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
    sns.histplot(lengths, kde=True)
    plt.title("Distribution of Sentence Lengths")
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.show()

def plot_training_metrics(training_loss, evaluation_accuracy):
    epochs = range(1, len(training_loss) + 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, training_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Evaluation Accuracy', color=color)
    ax2.plot(epochs, evaluation_accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Training Loss and Evaluation Accuracy Over Epochs")
    plt.show()

def aggregate_BIO(metrics, labels):
    # Map labels to their metrics for easier access
    label_metrics_map = {label: {metric: metrics[metric][i] for metric in metrics} for i, label in enumerate(labels)}
    
    # Identify unique B-I label pairs and prepare for aggregation
    b_labels = [label for label in labels if label.startswith('B-')]
    i_labels = [label.replace('B-', 'I-', 1) for label in b_labels]
    bi_label_set = set(b_labels + i_labels)
    
    # Initialize containers for aggregated metrics and labels
    aggregated_metrics = {metric: [] for metric in metrics}
    aggregated_labels = []
    
    # Aggregate metrics
    for label in labels:
        if label in bi_label_set:
            b_label = 'B-' + label[2:] if label.startswith('I-') else label
            i_label = 'I-' + label[2:] if label.startswith('B-') else label
            
            # Aggregate if both B and I versions exist, otherwise just copy B or I
            if b_label in label_metrics_map and i_label in label_metrics_map:
                bi_label = label[2:] + '-BI'
                if bi_label not in aggregated_labels:
                    # Aggregate and average metrics
                    for metric in metrics:
                        b_metric = label_metrics_map[b_label][metric]
                        i_metric = label_metrics_map[i_label][metric]
                        aggregated_metrics[metric].append((b_metric + i_metric) / 2)
                    aggregated_labels.append(bi_label)
            elif label.startswith('B-') and b_label not in aggregated_labels:
                # Just copy B label metrics if I-version doesn't exist
                for metric in metrics:
                    aggregated_metrics[metric].append(label_metrics_map[label][metric])
                aggregated_labels.append(b_label)
        elif label not in aggregated_labels and label not in i_labels:
            # Copy non B/I labels directly
            for metric in metrics:
                aggregated_metrics[metric].append(label_metrics_map[label][metric])
            aggregated_labels.append(label)
    
    # Ensure aggregated metrics and labels are aligned
    return aggregated_metrics, aggregated_labels


def complete_plot(metrics,labels,model_path):
    print(metrics)
    plot_f1_recall_precision(metrics,labels,model_path)
    aggregate_metrics, aggregate_labels = aggregate_BIO(metrics,labels)
    print(aggregate_metrics)
    print(aggregate_labels)
    plot_f1_recall_precision(aggregate_metrics,aggregate_labels,model_path)
    plt_fp_fn(metrics,labels,model_path)
    plot_global_metrics(metrics)
    plot_label_distribution(metrics,labels,model_path)
    plot_performance_metrics(metrics,labels,model_path)