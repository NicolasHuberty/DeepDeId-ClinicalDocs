def tokenize_and_encode_labels(labels, tokens, label2id):
    """
    This function adjusts labels to align with tokenized inputs, using the special token `-100` 
    to indicate sub-tokens or tokens that should not be considered for training (e.g., special tokens like [CLS], [SEP]).
    Args:
        labels (list of list of str): Original labels for each token in each sentence.
        tokens: Tokenizer output, must have `word_ids` method to map tokens to their original words.
        label2id (dict): Mapping from label strings to label IDs.
    Returns:
        list of list of int: Encoded labels adjusted for sub-tokens and special tokens.
    """
    # Convert labels to their corresponding IDs
    simplified_labels = [[label2id.get(label, label2id['O']) for label in doc] for doc in labels]
    encoded_labels = []
    #Iterate over each document's labels
    for i, label in enumerate(simplified_labels):
        word_ids = tokens.word_ids(batch_index=i)  # Get word id for each token
        previous_word_idx = None
        label_ids = []
        # Assign labels to tokens, using -100 for sub-tokens and special tokens
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # New word
                label_ids.append(label[word_idx])
            else:  # Sub-token of a previous word
                label_ids.append(-100)
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)

    return encoded_labels
