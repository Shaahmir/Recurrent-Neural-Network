from model.encoder import Encoder

from utils.dataset import ShardDataset
from utils.collate import collate_fn

from torch.utils.data import DataLoader
from config import cfg

dataset = ShardDataset(
    cfg.TRAIN_DIR / "shard_00000.pt"
)

loader = DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate_fn
)

x, y = next(iter(loader))

model = Encoder(
    vocab_size = cfg.MODEL["vocab_size"],
    embedding_dim = cfg.MODEL["embedding_dim"],
    hidden_size = cfg.MODEL["encoder_hidden"],
    num_layers = cfg.MODEL["num_layers"],
    dropout = cfg.MODEL["dropout"],
    pad_idx = cfg.MODEL["pad_idx"]
)

lengths = (x != 0).sum(dim = 1)

outputs, hidden, cell = model(
    x,
    lengths
)

print(outputs.shape)
print(hidden.shape)
print(cell.shape)