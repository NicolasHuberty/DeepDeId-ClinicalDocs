import sqlite3
import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fake manual annotations')
    parser.add_argument("--project_name", type=str, default="customProject", help="Name of the Project")
    parser.add_argument('--num_records', type=int, default="10", help='Number of records to fakely annotated')
    args = parser.parse_args()
    return args


def fake_manual_annotation(project_name, x):
    conn = sqlite3.connect(f'./projects/{project_name}/dataset.db')
    c = conn.cursor()
    # Select the first x ids from the records table
    c.execute('''SELECT id FROM records ORDER BY id LIMIT ?''', (x,))
    record_ids = c.fetchall()  # This will fetch all selected ids

    # Update the manual_process field for each selected record
    for record_id in record_ids:
        c.execute('''UPDATE records SET manual_process = 1 WHERE id = ?''', (record_id[0],))

    conn.commit()
    conn.close()
    print(f"Updated manual_process for the first {x} records.")

def main():
    args = parse_arguments()
    fake_manual_annotation(args.project_name,args.num_records)

if __name__ == '__main__':
    main()