import torch
import sentencepiece as spm
from config import cfg

sp = spm.SentencePieceProcessor()

sp.load(
    str(cfg.TOKENIZER_DIR / "chatbot.model")
)

model_data = torch.load(cfg.TRAIN_DIR / 'shard_00000.pt')

print(sp.decode(model_data["input_ids"][0].tolist()))
print(sp.decode(model_data["target_ids"][0].tolist()))

for x in model_data["input_ids"][:10]:
    print(sp.decode(x.tolist()))
    print()
    print()

for x in model_data["target_ids"][:10]:
    print(sp.decode(x.tolist()))
    print()
    print()