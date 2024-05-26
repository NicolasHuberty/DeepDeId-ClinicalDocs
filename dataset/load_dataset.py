# Function to load a txt file without labels
def load_txt_dataset(file_path):
    DELIMITER = "<RECORD_SEPARATOR>"
    # Open and read file lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Prepare to capture formatted records
    formattedRecords = []
    currentRecord = []
    # Process each line to form records
    for line in lines:
        line = line.strip()
        tokens = line.split()
        for token in tokens:
            if token == DELIMITER:
                # Start a new record
                formattedRecords.append(currentRecord)
                currentRecord = []
            else: 
                # Append token to current record
                currentRecord.append(token)
    # Return the list of records
    return formattedRecords

# Function to load a formatted tsv dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    texts, labels = [], []
    record_tokens, record_labels = [], []
    # Process each line for tokens and labels
    for line in lines:
        if line == "\n":
            # Save complete record and reset for next
            texts.append(record_tokens)
            labels.append(record_labels)
            record_tokens, record_labels = [], []
        else:
            # Extract token and label
            token, label = line.strip().split()
            record_tokens.append(token)
            if(label):
                record_labels.append(label)
    # Append last processed record if exists
    if record_tokens:
        texts.append(record_tokens)
        labels.append(record_labels)
    return texts, labels
