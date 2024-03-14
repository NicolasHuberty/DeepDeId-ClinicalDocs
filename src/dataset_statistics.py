from pathlib import Path
import sys
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils import readFormattedFile, dataset_statistics
tokens, labels,unique = readFormattedFile("wikiNER/train.tsv",mapping="None")
print(len(tokens))
dataset_statistics(tokens,labels,unique)