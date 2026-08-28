import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.dropout = nn.Dropout(
            p = dropout
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first = True
        )

    def _merge_directions(self, state):

        num_layers = self.lstm.num_layers
        batch_size = state.size(1)
        hidden_size = state.size(2)

        state = state.view(
            num_layers, 2, batch_size, hidden_size
        )

        state = state.sum(dim = 1)

        return state

    def forward(self, x, lengths):
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first = True,
            total_length = x.size(1)
        )

        hidden = self._merge_directions(hidden)
        cell = self._merge_directions(cell)

        return outputs, hidden, cell