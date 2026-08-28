from config import cfg
from utils.multishards import MultiShardDataset
from torch.utils.data import DataLoader
from utils.collate import collate_fn

dataset = MultiShardDataset(
    cfg.TRAIN_DIR
)

loader = DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 0
)

print(len(dataset))
print(dataset[0][0][:10])
print(dataset[50000][0][:10])
print(dataset[100000][0][:10])

x,y = next(iter(loader))

print(x.shape)
print(y.shape)