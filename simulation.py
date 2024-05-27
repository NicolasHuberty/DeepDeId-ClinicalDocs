# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
import sys
from pathlib import Path
import argparse
# Add root path to system path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from src import fake_manual_annotation, fake_manual_annotation_best_confidence
from src import process_new_records
import numpy as np

parser = argparse.ArgumentParser(description='Create a Project from scratch to identify specific labels on documents')
parser.add_argument("--project_name", type=str, default="wikiNER", help="Name of the Project")
parser.add_argument("--steps",type=int,default=10,help="Number of documents process at each step")
parser.add_argument("--to",type=int,default=500,help="Number of documents annotation to simulate")

args = parser.parse_args()
# Launch simulation of ILA when a project is correctly created
vals =  np.full(int(int(args.to)/int(args.steps)), args.steps)
for i in vals:
    print(f"Handle i {i}")
    fake_manual_annotation_best_confidence(args.project_name,i)
    process_new_records(args.project_name)
