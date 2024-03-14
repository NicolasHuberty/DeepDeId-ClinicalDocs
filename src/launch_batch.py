import subprocess
import numpy as np
mappings = ["n2c2_removeBIO","None","n2c2_BIO","n2c2"]
epochs = [10,20,30]
batch_size = [1,2,4,8]
dataset_sizes = [1,2,5,10, 20, 50, 100, 200, 360, 720]
"""
size = [100,200,300,400,500,600,700]
for map in mappings:
    command = ["python", "main.py", "--mapping", map]
    print(f"Launch {command}")
    subprocess.run(command)
for epoch in epochs:
    command = ["python", "main.py", "--epochs", str(epoch)]
    print(f"Launch {command}")
    subprocess.run(command)
for batch_s in batch_size:
    command = ["python", "main.py", "--batch_size", str(batch_s)]
    print(f"Launch {command}")
    subprocess.run(command)
"""
for size in dataset_sizes:
    command = ["python", "main.py", "--dataset_size", str(size)]
    print(f"Launch {command}")
    subprocess.run(command)
