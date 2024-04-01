import sys
from pathlib import Path
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
sys.path.append("datasets")
from src import fake_manual_annotation, fake_manual_annotation_best_confidence
from src import process_new_records
from load_dataset import load_txt_dataset,load_dataset

project_name = "n2c2"
eval_records, eval_labels = load_dataset("datasets/formatted/n2c2_2014/training3Map.tsv")
for i in range(0,100,1):
    print(f"Handle i {i}")
    fake_manual_annotation(project_name,1)
    process_new_records(project_name,new_predictions=0)
