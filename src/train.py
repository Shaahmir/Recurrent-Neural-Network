import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from trainer import Trainer

from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import Decoder
from model.seq2seq import Seq2Seq

from utils.collate import collate_fn
from utils.checkpoint import load_checkpoint

start_epoch = 0
device = cfg.DEVICE

train_shards = sorted(
    Path(cfg.TRAIN_DIR).glob("*.pt")
)

valid_shards = sorted(
    Path(cfg.VALID_DIR).glob("*.pt")
)

if len(train_shards) == 0:
    raise RuntimeError("No training shard found!")

if len(valid_shards) == 0:
    raise RuntimeError("No validation shard found!")

attention = BahdanauAttention(
    cfg.MODEL["encoder_hidden"],
    cfg.MODEL["decoder_hidden"],
)

encoder = Encoder(
    vocab_size = cfg.MODEL["vocab_size"],
    embedding_dim = cfg.MODEL["embedding_dim"],
    hidden_size = cfg.MODEL["encoder_hidden"],
    num_layers = cfg.MODEL["num_layers"],
    dropout = cfg.MODEL["dropout"],
    pad_idx = cfg.MODEL["pad_idx"]
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

optimizer = AdamW(
    model.parameters(),
    lr = cfg.LEARNING_RATE,
    weight_decay = cfg.WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss(
    ignore_index = cfg.MODEL["pad_idx"]
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max = cfg.EPOCHS
)

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    scheduler = scheduler,
    device = device
)

# Load Checkpoint

latest = cfg.CHECKPOINT_DIR / "latest.pt"
batch_latest = cfg.CHECKPOINT_DIR / "batch_latest.pt"

latest_epoch = (
    torch.load(latest, map_location = "cpu")["epoch"]
    if latest.exists() else None
)

batch_epoch = (
    torch.load(batch_latest, map_location = "cpu")["epoch"]
    if batch_latest.exists() else None
)

if batch_epoch is not None and (latest_epoch is None or batch_epoch > latest_epoch):

    checkpoint = load_checkpoint(
        batch_latest,
        model,
        optimizer,
        scheduler,
        trainer.scaler
    )

    start_epoch = checkpoint["epoch"]
    start_shard = checkpoint.get("shard_index")
    start_shard = start_shard + 1 if start_shard is not None else 0

    print(f"Resuming Epoch {start_epoch + 1}, Shard {start_shard} (Mid Epoch) ...")

elif latest_epoch is not None:

    checkpoint = load_checkpoint(
        latest,
        model,
        optimizer,
        scheduler,
        trainer.scaler
    )

    start_epoch = checkpoint["epoch"] + 1
    start_shard = 0

    print(f"Resuming from Epoch {start_epoch} ...")

else:
    start_epoch = 0
    start_shard = 0
    print("Starting from Epoch 1 ...")

if start_shard >= len(train_shards):
    start_epoch += 1
    start_shard = 0

trainer.fit(
    train_shards = train_shards,
    valid_shards = valid_shards,
    batch_size = cfg.BATCH_SIZE,
    collate_fn = collate_fn,
    epochs = cfg.EPOCHS,
    num_workers = cfg.NUM_WORKERS,
    pin_memory = cfg.PIN_MEMORY,
    persistent_workers = cfg.PERSISTENT_WORKERS,
    start_epoch = start_epoch,
    start_shard = start_shard
)