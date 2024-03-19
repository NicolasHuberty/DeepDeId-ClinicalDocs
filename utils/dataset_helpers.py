import sqlite3
import json
import random
def load_config_field(project_name, field):
    with open(f"projects/{project_name}/config_project.json", 'r') as file:
        json_data = json.load(file)
    # Retrieve the value of the specified field from the json data
    field_value = json_data.get(field, None)  # Returns None if the field does not exist
    return field_value

def generate_label_colors(project_name):
    labels = load_config_field(project_name, 'labels')
    label_colors = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in labels if label != "O"}
    label_colors["O"] = "#FFFFFF" 
    return label_colors

def save_config_field(project_name, field, new_value):
    config_file_path = f"projects/{project_name}/config_project.json"
    # Load the current configuration
    with open(config_file_path, 'r') as file:
        json_data = json.load(file)
    json_data[field] = new_value
    with open(config_file_path, 'w') as file:
        json.dump(json_data, file, indent=4) 

def load_records_manual_process(project_name,manual_process=1):
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    c.execute('''SELECT id, text, manual_labels, predicted_labels, manual_process FROM records WHERE manual_process = ?''',(manual_process,))
    records = c.fetchall()
    conn.close()
    texts = []
    manual_labels_list = []
    predicted_labels_list = []
    manual_process_flags = []
    for record in records:
        _, text_str, manual_labels_str, predicted_labels_str, _ = record
        text = text_str.split(' ')
        manual_labels = list(manual_labels_str.split(','))
        predicted_labels = list(predicted_labels_str.split(','))
        texts.append(text)
        manual_labels_list.append(manual_labels)
        predicted_labels_list.append(predicted_labels)
        manual_process_flags.append(True)  
    return texts, manual_labels_list, predicted_labels_list, manual_process_flags


def store_record_with_labels(project_name,record_id, text, manual_labels, predicted_labels):
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    text_str = ' '.join(text)  # Reconstruire la phrase à partir de la liste des mots
    manual_labels_str = ','.join(map(str, manual_labels))  # Assumant que manual_labels est déjà une liste de labels correspondants
    predicted_labels_str = ','.join(map(str, predicted_labels))
    c.execute('''INSERT INTO records (id, text, manual_labels, predicted_labels,manual_process)
                 VALUES (?, ?, ?, ?, ?)''',
                 (record_id, text_str, manual_labels_str, predicted_labels_str,False))
    conn.commit()
    conn.close()


def store_predicted_labels(project_name,records_ids, predicted_labels):
    conn = sqlite3.connect(f'projects/{project_name}/dataset.db')
    c = conn.cursor()
    print(f"Size of records ids: {len(records_ids)} and size of predicted labels: {len(predicted_labels)}")
    for record_id, labels in zip(records_ids, predicted_labels):
        record_id = int(record_id)
        predicted_labels_str = ','.join(map(str, labels))
        c.execute('''SELECT EXISTS(SELECT 1 FROM records WHERE id = ? LIMIT 1)''', (record_id,))
        exists = c.fetchone()[0]
        if not exists:
            print(f"No record found with ID: {record_id}")
        else:
            c.execute('''UPDATE records SET predicted_labels = ? WHERE id = ?''', (predicted_labels_str, record_id))

    conn.commit()
    conn.close()