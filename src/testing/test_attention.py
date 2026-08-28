from model.encoder import Encoder
from model.attention import BahdanauAttention

from utils.dataset import ShardDataset
from utils.collate import collate_fn

from torch.utils.data import DataLoader
from config import cfg

dataset = ShardDataset(
    cfg.TRAIN_DIR / "shard_00000.pt"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

x, y = next(iter(loader))

encoder = Encoder(
    vocab_size = 16000,
    embedding_dim = 256,
    hidden_size = 512,
    num_layers = 2,
    dropout = 0.3,
    pad_idx = 0
)

attention = BahdanauAttention(
    encoder_hidden_size = cfg.MODEL["encoder_hidden"],
    decoder_hidden_size = cfg.MODEL["decoder_hidden"],
)

lengths = (x != 0).sum(dim = 1)

encoder_outputs, hidden, cell = encoder(
    x,
    lengths
)

encoder_proj = attention.W(
    encoder_outputs
)

weights = attention(hidden[-1], encoder_proj)

print(weights.shape)
print(weights.sum(dim = 1))