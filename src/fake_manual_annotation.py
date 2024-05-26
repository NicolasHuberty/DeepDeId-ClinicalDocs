import sys
from pathlib import Path
import argparse
import sqlite3
import random
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils import load_record_with_lowest_confidence
from src import handle_new_record
def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulate manual_annotations')
    parser.add_argument("--project_name", type=str, default="n2c2", help="Name of the Project")
    parser.add_argument('--num_records', type=int, default="10", help='Number of records to simulate annotation')
    parser.add_argument('--confidence_mode', type=bool, default=False, help='Annotate the lowest confidence first')
    args = parser.parse_args()
    return args


def fake_manual_annotation(project_name, num_records):
    # Annotate the num_records first documents on the database
    try:
        conn = sqlite3.connect(f'./projects/{project_name}/dataset.db')
        c = conn.cursor()
        # Retrieve last processed document
        c.execute('''SELECT MAX(id) FROM records WHERE manual_process = 2''')
        max_id_result = c.fetchone()
        start_id = max_id_result[0] if max_id_result[0] is not None else 0
        # Retrieve new documents that will be annotated
        c.execute('''SELECT id, manual_labels FROM records WHERE id > ? ORDER BY id LIMIT ?''', (start_id, num_records))
        record_ids = c.fetchall()
        conn.commit()
        conn.close()
        for record in record_ids:
            #modified_labels = [label if label == 'O' or random.random() >= 0.00 else 'O' for label in record[1].split(',')]
            handle_new_record(project_name,record[2],record[0])


    except sqlite3.OperationalError as e:
        print(f"SQLite Operational Error while simulate annotation: {e}")


def fake_manual_annotation_best_confidence(project_name, num_records):
    # Annotate the num_records documents on the database based on their confidence percentage
    # Retrieve lowest confidence document and update its processed status
    for _ in range(num_records):
        record = load_record_with_lowest_confidence(project_name)
        print(record)
        handle_new_record(project_name,','.join(record[2]),int(record[0]))
        #handle_new_record(project_name,record[2],int(record[0]))

        
def main():
    args = parse_arguments()
    if(args.confidence_mode):
        fake_manual_annotation_best_confidence(args.project_name,args.num_records)
    else:
        fake_manual_annotation(args.project_name,args.num_records)

if __name__ == '__main__':
    main()