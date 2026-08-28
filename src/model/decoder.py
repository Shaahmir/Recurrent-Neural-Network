import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        encoder_hidden_size,
        decoder_hidden_size,
        num_layers,
        dropout,
        pad_idx,
        attention
    ):

        super().__init__()

        self.attention = attention

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx = pad_idx
        )

        self.dropout = nn.Dropout(
            p = dropout
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim + encoder_hidden_size * 2,
            hidden_size = decoder_hidden_size,
            num_layers = num_layers,
            dropout = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )

        self.fc = nn.Linear(
            decoder_hidden_size + encoder_hidden_size * 2,
            vocab_size
        )

    def forward(self, input_token, hidden, cell, encoder_outputs, encoder_proj, mask):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        attention = self.attention(
            hidden[-1],
            encoder_proj,
            mask
        )

        attention = attention.unsqueeze(1)

        context = torch.bmm(
            attention,
            encoder_outputs
        )

        lstm_input = torch.cat(
            (embedded, context),
            dim = 2
        )

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(
            torch.cat(
                (output, context),
                dim = 1
            )
        )

        return prediction, hidden, cell