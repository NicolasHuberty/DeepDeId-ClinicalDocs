import matplotlib.pyplot as plt
from itertools import chain
import os
from mapping import load_mapping

def readFormattedFile(file_path, mapping="None"):
    """
    Reads a formatted file from a specified path and optionally applies a label mapping.
    Parameters:
    - file_path (str): Relative path to the file within the "./datasets/formatted/" directory.
    - mapping (str): The name of the mapping to apply. If None, no mapping is applied.
    Returns:
    - tuple of (list of list of str, list of list of str, list of str): Tokens, labels, and unique labels.
    """
    with open(os.path.join("./datasets/formatted/", file_path), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    texts, labels = [], []
    record_tokens, record_labels = [], []
    unique_labels = set()
    for line in lines:
        if line == "\n":
            if record_tokens:
                texts.append(record_tokens)
                labels.append(record_labels)
                record_tokens, record_labels = [], []
        else:
            token, label = line.strip().split()
            record_tokens.append(token)
            record_labels.append(label)
            unique_labels.add(label)
    if record_tokens:
        texts.append(record_tokens)
        labels.append(record_labels)

    if mapping != "None":
        print(f"Launch te mapping of the labels with {mapping}")
        mapping_dict = load_mapping(mapping)
        labels = [[mapping_dict.get(label, 'O') for label in sentence_labels] for sentence_labels in labels]
        unique_labels = set(chain.from_iterable(labels))

    return texts, labels, list(unique_labels)



def dataset_statistics(tokens, labels, unique_labels):
    """
    Parameters:
    - texts (list of list of str): Each inner list contains tokens from a single record.
    - labels (list of list of str): Each inner list contains labels for the tokens in the corresponding record.
    - unique_labels (list of str): A list of all unique labels present in the dataset.
    Returns:
    - Prints the total number of tokens, the total number of unique labels, and a distribution of the labels.
    - Visualizes the label distribution in a bar chart.
    """
    all_labels = list(chain.from_iterable(labels))
    total_tokens = sum(len(sentence) for sentence in tokens)
    total_labels = len(set(all_labels))
    label_distribution = {label: all_labels.count(label) for label in unique_labels}
    print(f"Number of documents: {len(tokens)}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Total number of unique labels: {total_labels}")
    print(f"Label distribution:{label_distribution}")

    # Plotting the label distribution
    plt.figure(figsize=(10, 6))
    plt.bar(label_distribution.keys(), label_distribution.values())
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels in the Dataset')
    plt.xticks(rotation=45)
    #plt.show()
    return label_distribution
