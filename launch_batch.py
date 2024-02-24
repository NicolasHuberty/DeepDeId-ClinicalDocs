import subprocess
mappings = ["n2c2_removeBIO","None","n2c2_BIO","n2c2"]
epochs = [10,20,30]
batch_size = [1,2,4,8]
"""
size = [100,200,300,400,500,600,700]
for map in mappings:
    command = ["python", "main.py", "--mapping", map]
    print(f"Launch {command}")
    subprocess.run(command)
"""
for epoch in epochs:
    command = ["python", "main.py", "--epochs", str(epoch)]
    print(f"Launch {command}")
    subprocess.run(command)
"""
for batch_s in batch_size:
    command = ["python", "main.py", "--batch_size", str(batch_s)]
    print(f"Launch {command}")
    subprocess.run(command)
"""