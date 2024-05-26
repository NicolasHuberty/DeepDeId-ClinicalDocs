import os
import xml.etree.ElementTree as ET
import csv
import re
DELIMITER = "<RECORD_SEPARATOR>"

def is_utf8_char(char):
    try:
        char.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False
    
    
def format_csv_dataset(file_path, save=False):
    DELIMITER = "<RECORD_SEPARATOR>"
    any_brackets_pattern = re.compile(r'<[^>]*>')    
    formattedRecords = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            text = f"Subject: {row['Subject']}\nFrom: {row['Sender']}\nTo: {row['To']}\nBody: {row['Body']}"
            cleaned_text = re.sub(any_brackets_pattern, '', text)
            # Keep only UTF-8 characters
            utf8_text = ''.join(c for c in cleaned_text if is_utf8_char(c))
            formattedRecords.append(f"{utf8_text}\n{DELIMITER}\n")
            
    if save:
        saveFile = os.path.splitext(file_path)[0] + "_formatted.txt"
        with open(saveFile, 'w', encoding='utf-8') as f:
            f.writelines(formattedRecords)

    return formattedRecords


if __name__ == "__main__":
    format_csv_dataset("./dataset/unformatted/merged_mails.csv",save=True)