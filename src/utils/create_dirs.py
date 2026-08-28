from pathlib import Path

folders = [
    "data/raw",
    "data/processed",
    "data/tokenizer",
    "logs",
    "checkpoints",
]

for folder in folders:
    Path(folder).mkdir(parents = True, exist_ok = True)

print("Folders created.")