import sys
from pathlib import Path
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from src import fake_manual_annotation, fake_manual_annotation_best_confidence
from src import process_new_records
from load_dataset import load_txt_dataset,load_dataset
import numpy as np
# Launch simulation of ILA when a project is correctly created
vals =  np.full(30, 10)
project_name = "mails"
for i in vals:
    print(f"Handle i {i}")
    fake_manual_annotation_best_confidence(project_name,i)
    process_new_records(project_name)
