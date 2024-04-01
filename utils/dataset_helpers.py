import sqlite3
import json

def load_config_field(project_name, field):
    with open(f"projects/{project_name}/config_project.json", 'r') as file:
        json_data = json.load(file)
    field_value = json_data.get(field, None)
    return field_value


def save_config_field(project_name, field, new_value):
    config_file_path = f"projects/{project_name}/config_project.json"
    # Load the current configuration
    with open(config_file_path, 'r') as file:
        json_data = json.load(file)
    json_data[field] = new_value
    with open(config_file_path, 'w') as file:
        json.dump(json_data, file, indent=4) 



def load_record_with_lowest_confidence(project_name):
    # Function used to retrieve the record with the lowest confidence
    conn = sqlite3.connect(f'./projects/{project_name}/dataset.db')
    c = conn.cursor()
    c.execute('''SELECT id, text, manual_labels, predicted_labels, manual_process, confidence FROM records WHERE confidence IS NOT NULL 
              AND manual_process = 0 ORDER BY confidence ASC LIMIT 1''')
    record = c.fetchone()
    if not record:
        c.execute('''SELECT id, text, manual_labels, predicted_labels, manual_process, confidence
                     FROM records WHERE manual_process = 0 ORDER BY id ASC LIMIT 1''')
        record = c.fetchone()
    conn.close()
    if record:
        id, text_str, manual_labels_str, predicted_labels_str, manual_process, confidence = record
        text = text_str.split(' ')
        manual_labels = manual_labels_str.split(',')
        predicted_labels = predicted_labels_str.split(',')
        return id, text, manual_labels, predicted_labels, confidence


def load_records_eval_set(project_name, eval_set=1, manual_process=1):
    # Function used to retrieve records with specific flags for eval_set and manual_process
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    query = '''SELECT id, text, manual_labels, predicted_labels, manual_process FROM records WHERE eval_record = ? AND manual_process = ?'''
    params = [eval_set, manual_process]
    c.execute(query, params)
    records = c.fetchall()
    conn.close()
    texts = []
    manual_labels_list = []
    predicted_labels_list = []
    manual_process_flags = []
    ids = []
    for record in records:
        id, text_str, manual_labels_str, predicted_labels_str, _ = record
        text = text_str.split(' ')
        manual_labels = list(manual_labels_str.split(','))
        predicted_labels = list(predicted_labels_str.split(','))
        ids.append(id)
        texts.append(text)
        manual_labels_list.append(manual_labels)
        predicted_labels_list.append(predicted_labels)
        manual_process_flags.append(True)
    return ids, texts, manual_labels_list, predicted_labels_list, manual_process_flags

def manual_process(project_name, manual_labels, record_id,allocate_to_eval = False):
    # Update the manual_labels when annotating
    conn = sqlite3.connect(f'./projects/{project_name}/dataset.db')
    c = conn.cursor()
    c.execute('UPDATE records SET manual_labels=?, manual_process=1, eval_record=? WHERE id=?', (manual_labels, allocate_to_eval, record_id))
    conn.commit()
    conn.close()

def set_manual_labels(project_name, manual_process_value, record_ids_list):
    # Set manual_process flag for specific records (used for fake_manual_annotation)
    conn = sqlite3.connect(f'./projects/{project_name}/dataset.db')
    c = conn.cursor()
    for record_id in record_ids_list:
        c.execute('''UPDATE records SET manual_process = ? WHERE id = ?''', (manual_process_value, record_id))
    conn.commit()
    conn.close()

def store_record_with_labels(project_name,record_id, text, manual_labels, predicted_labels):
    # Put the records on the database
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    text_str = ' '.join(text)
    manual_labels_str = ','.join(map(str, manual_labels)) 
    predicted_labels_str = ','.join(map(str, predicted_labels))
    c.execute('''INSERT INTO records (id, text, manual_labels, predicted_labels,manual_process,eval_record)
                 VALUES (?, ?, ?, ?, ?, ?)''',
                 (record_id, text_str, manual_labels_str, predicted_labels_str,False, 0))
    conn.commit()
    conn.close()


def store_predicted_labels(project_name,records_ids, predicted_labels):
    # Update the prediction for the records
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    for record_id, predictions in zip(records_ids, predicted_labels):
        labels, confidence = predictions
        record_id = int(record_id)
        predicted_labels_str = ','.join(map(str, labels))
        c.execute('''SELECT EXISTS(SELECT 1 FROM records WHERE id = ? LIMIT 1)''', (record_id,))
        exists = c.fetchone()[0]
        if not exists:
            print(f"No record found with ID: {record_id}")
        else:
            c.execute('''UPDATE records SET predicted_labels = ?, confidence = ? WHERE id = ?''', (predicted_labels_str,confidence, record_id))

    conn.commit()
    conn.close()
