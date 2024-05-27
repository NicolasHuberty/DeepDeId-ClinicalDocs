# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import sqlite3
import json

def validate_json(json_data):
    # Check validity of the JSON file
    try:
        json.loads(json_data) 
        return True
    except json.JSONDecodeError:
        return False

def load_config_field(project_name, field):
    # Load a specific field value from a project JSON configuration file
    config_path = f"projects/{project_name}/config_project.json"
    with open(config_path, 'r') as file:
        json_data = file.read()
    if validate_json(json_data):
        data_dict = json.loads(json_data)
        field_value = data_dict.get(field, None)
        return field_value
    else:
        return None


def save_config_field(project_name, field, new_value):
    # Modify a specific field in a project JSON configuration file.
    config_file_path = f"projects/{project_name}/config_project.json"
    with open(config_file_path, 'r') as file:
        json_data = file.read()
    if validate_json(json_data):
        data_dict = json.loads(json_data)
        data_dict[field] = new_value
        # Validate the modified dictionary by converting it back to JSON string
        new_json_data = json.dumps(data_dict, indent=4)
        if validate_json(new_json_data):
            with open(config_file_path, 'w') as file:
                file.write(new_json_data)



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
    c.execute('UPDATE records SET manual_labels=?, manual_process=1, eval_record=? WHERE id=?', (','.join(manual_labels), allocate_to_eval, record_id))
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
