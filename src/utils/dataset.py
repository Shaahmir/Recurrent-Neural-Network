import torch
from torch.utils.data import Dataset

class ShardDataset(Dataset):

    def __init__(self, shard_path):
        
        data = torch.load(shard_path)
        self.inputs = data["input_ids"]
        self.targets = data["target_ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return (
            self.inputs[idx],
            self.targets[idx]
        )