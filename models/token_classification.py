def tokenize_and_encode_labels(labels, tokens, label2id):
    """
    This function maps textual labels to their respective IDs and aligns them with tokens,
    especially accounting for tokenization effects like sub-tokenization and special tokens

    Parameters:
    - labels (list of list of str): List of documents where each document is a list of labels
    - tokens: Tokenizer output that contains mappings between tokens and their corresponding word in the input
    - label2id (dict): Dictionary mapping label names to an integer ID

    Returns:
    - encoded_labels (list of list of int): Encoded labels aligned with tokenizer output, where
      sub-tokens and special tokens get a special ID of -100 to ignore in loss computation
    """
    # Convert labels to their corresponding IDs
    simplified_labels = [[label2id.get(label, label2id['O']) for label in doc] for doc in labels]
    encoded_labels = []
    # Iterate over each document's labels
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