def tokenize_and_encode_labels(labels, tokens,label2id):
    simplified_labels = [[label2id.get(label, label2id['O']) for label in doc] for doc in labels]
    encoded_labels = []
    for i, label in enumerate(simplified_labels):
        word_ids = tokens.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) 
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)
    return encoded_labels