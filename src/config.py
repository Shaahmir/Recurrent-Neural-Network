import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen = True)
class Config:

    # Paths Configurations

    ROOT: Path = Path(__file__).resolve().parent.parent

    RAW_DATA: Path = ROOT / "data" / "raw"
    PROCESSED_DATA: Path = ROOT / "data" / "processed"
    TOKENIZER_DIR: Path = ROOT / "data" / "tokenizer"

    TRAIN_DIR: Path = PROCESSED_DATA / "train"
    VALID_DIR: Path = PROCESSED_DATA /"valid"

    CHECKPOINT_DIR: Path = ROOT / "checkpoints"
    LOG_DIR: Path = ROOT / "logs"

    # Device configurations

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Tokenizer Configurations

    VOCAB_SIZE: int = 16000

    # Training Configurations

    BATCH_SIZE = 28
    ACCUMULATION_STEPS = 2
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    NUM_WORKERS = 2
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    CHECKPOINT_BATCH = 1786

    # Model Configurations

    MODEL = {
        "vocab_size": 16000,
        "embedding_dim": 256,
        "encoder_hidden": 512,
        "decoder_hidden": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "pad_idx": 0,
    }

    # Dataset

    MAX_HISTORY: int = 8

    MAX_INPUT_LENGTH: int = 1024
    MAX_TARGET_LENGTH: int = 256

    VALID_SPLIT: float = 0.02
    SHARD_SIZE: int = 50000

    RANDOM_SEED: int = 42

    # Special Tokens

    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

cfg = Config()