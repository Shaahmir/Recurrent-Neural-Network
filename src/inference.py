import torch
import sentencepiece as spm

from model.encoder import Encoder
from model.attention import BahdanauAttention
from model.decoder import Decoder
from model.seq2seq import Seq2Seq

from utils.beam_search import beam_seach_decode
from utils.checkpoint import load_checkpoint
from config import cfg
from collections import deque

GRAY = "\033[90m"
RED = "\033[91m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
RESET = "\033[0m"

def build_model(device):
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

    return model

def main():
    device = cfg.DEVICE
    sp = spm.SentencePieceProcessor()

    sp.load(
        str(cfg.TOKENIZER_DIR / "chatbot.model")
    )

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    model = build_model(device)

    load_checkpoint(
        cfg.CHECKPOINT_DIR / "latest.pt",
        model
    )

    model.eval()

    max_history = getattr(cfg, "MAX_HISTORY", 8)
    history =  deque(maxlen = max_history)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"{GRAY}Total Parameters     : {total:,}{RESET}")
    print(f"{GRAY}Trainable Parameters : {trainable:,}{RESET}")
    print()

    print(f"{GRAY}Where should we begin?{RESET}")
    print()

    while True:

        text = input(f"{RED}You: {WHITE}").strip()

        if not text:
            continue

        if text.lower() == "exit":
            break

        if text.lower() == "/reset":
            history.clear()
            print(f"{GRAY}Chat history cleared.{RESET}")
            continue

        history.append(f"User: {text}")
        content = "\n".join(history)

        ids = sp.encode(content, add_bos = True, add_eos = True)
        
        source = torch.tensor(
            [ids],
            dtype = torch.long,
            device = device
        )

        output_ids = beam_seach_decode(
            model = model,
            source = source,
            bos_id = bos_id,
            eos_id = eos_id,
            beam_width = 5,
            max_length = 64,
            length_penalty_alpha = 0.7
        )

        response = sp.decode(
            output_ids if output_ids else ""
        )

        print(f"\n{YELLOW}AI: {RESET}{response}\n")

if __name__ == "__main__":
    main()
