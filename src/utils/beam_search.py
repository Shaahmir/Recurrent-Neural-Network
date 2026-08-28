import torch
from config import cfg

def _length_penalty(length: int, alpha: float = 0.7) -> float:
    return ((5.0 + length) / 6.0) ** alpha

@torch.no_grad()
def beam_seach_decode(
    model,
    source: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_width: int = 5,
    max_length: int = 64,
    length_penalty_alpha: float = 0.7
):
    model.eval()

    if source.dim() != 2:
        raise ValueError("Source must have a shape (batch_size, seq_len)")

    if source.size(0) != 1:
        raise ValueError("Beam search currently supports batch_size = 1")

    device = cfg.DEVICE
    lengths = (source != 0).sum(dim = 1)
    encoder_outputs, hidden, cell = model.encoder(source, lengths)
    encoder_proj = model.decoder.attention.W(
        encoder_outputs
    )
    mask = source != 0

    beams = [{
        "tokens": [bos_id],
        "score": 0.0,
        "hidden": hidden,
        "cell": cell
    }]

    completed = []

    for _ in range(max_length):
        candidates = []

        for beam in beams:
            tokens = beam["tokens"]
            score = beam["score"]
            h = beam["hidden"]
            c = beam["cell"]

            if tokens[-1] == eos_id:
                completed.append(beam)
                continue

            input_token = torch.tensor(
                [tokens[-1]],
                device = device,
                dtype = torch.long
            )

            prediction, next_hidden, next_cell = model.decoder(
                input_token, h, c, encoder_outputs, encoder_proj, mask
            )

            log_probs = torch.log_softmax(
                prediction.squeeze(0),
                dim = -1
            )

            top_log_probs, top_indices = torch.topk(
                log_probs, beam_width
            )

            for log_p, idx in zip(top_log_probs.tolist(), top_indices.tolist()):

                candidates.append({
                    "tokens": tokens + [idx],
                    "score": score + log_p,
                    "hidden": next_hidden.clone(),
                    "cell": next_cell.clone()
                })

        if not candidates:
            break

        def norm_score(b):
            return b["score"] / _length_penalty( len(b["tokens"]), alpha = length_penalty_alpha)

        candidates.sort(
            key = norm_score,
            reverse = True
        )

        beams = candidates[:beam_width]

    completed.extend(beams)

    best = max(
        completed,
        key = lambda b: b["score"] / _length_penalty(len(b["tokens"]), alpha = length_penalty_alpha)
    )

    tokens = best["tokens"][1:]

    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]

    return tokens