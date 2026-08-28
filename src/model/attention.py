import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size):

        super().__init__()

        self.W = nn.Linear(
            encoder_hidden_size * 2,
            encoder_hidden_size
        )

        self.U = nn.Linear(
            decoder_hidden_size,
            decoder_hidden_size
        )

        self.V = nn.Linear(
            decoder_hidden_size,
            1
        )

    def forward(self, hidden, encoder_proj, mask = None):

        hidden = hidden.unsqueeze(1)

        energy = torch.tanh(
            encoder_proj + self.U(hidden)
        )

        scores = self.V(
            energy
        ).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask, 
                float("-inf")
        )

        attention = torch.softmax(
            scores, dim = 1
        )

        return attention