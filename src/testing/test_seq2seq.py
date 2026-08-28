from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import Decoder
from model.seq2seq import Seq2Seq

from utils.dataset import ShardDataset
from utils.collate import collate_fn

from torch.utils.data import DataLoader
from config import cfg

device = cfg.DEVICE

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

x = x.to(device)
y = y.to(device)

encoder = Encoder(
    vocab_size = cfg.MODEL["vocab_size"],
    embedding_dim = cfg.MODEL["embedding_dim"],
    hidden_size = cfg.MODEL["encoder_hidden"],
    num_layers = cfg.MODEL["num_layers"],
    dropout = cfg.MODEL["dropout"],
    pad_idx = cfg.MODEL["pad_idx"]
)

attention = BahdanauAttention(
    encoder_hidden_size = cfg.MODEL["encoder_hidden"],
    decoder_hidden_size = cfg.MODEL["decoder_hidden"]
)

decoder = Decoder(
    vocab_size = cfg.MODEL["vocab_size"],
    embedding_dim = cfg.MODEL["embedding_dim"],
    encoder_hidden_size = cfg.MODEL["encoder_hidden"],
    decoder_hidden_size = cfg.MODEL["decoder_hidden"],
    num_layers = cfg.MODEL["num_layers"],
    dropout = cfg.MODEL["dropout"],
    pad_idx = cfg.MODEL["pad_idx"],
    attention = attention
)

model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)

outputs = model(x, y)
print(outputs.shape)