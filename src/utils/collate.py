import torch
import numpy as np

from utils.dataset import ShardDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import cfg

def collate_fn(batch):

    inputs, targets = zip(*batch)

    inputs = pad_sequence(
        inputs,
        batch_first = True,
        padding_value = cfg.PAD
    )

    targets = pad_sequence(
        targets,
        batch_first = True,
        padding_value = cfg.PAD
    )

    return inputs, targets

# ------------ Exploratory Code (Disabled) ------------

# dataset = ShardDataset(
#     cfg.TRAIN_DIR / "shard_00000.pt"
# )

# loader = DataLoader(
#     dataset,
#     batch_size = 32, 
#     shuffle = True,
#     collate_fn = collate_fn
# )

# x,y = next(iter(loader))

# print(x.shape)
# print(y.shape)

# Visualizing noise in dataset

# lengths = []

# for x, y in dataset:
#     lengths.append(len(x))

# print("Maximum:", max(lengths))
# print("Average:", sum(lengths)/len(lengths))

# print(np.percentile(lengths, 50))
# print(np.percentile(lengths, 90))
# print(np.percentile(lengths, 95))
# print(np.percentile(lengths, 99))