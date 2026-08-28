import random
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        
        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_len,
            vocab_size,
            device = self.device
        )

        lengths = (source != 0).sum(dim = 1)

        encoder_outputs, hidden, cell = self.encoder(
            source,
            lengths
        )

        encoder_proj = self.decoder.attention.W(
            encoder_outputs
        )

        mask = source != 0

        input_token = target[:, 0]

        for t in range(1, target_len):

            prediction, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                encoder_proj,
                mask
            )

            outputs[:, t] = prediction
            best_token = prediction.argmax(dim = 1)

            teacher_force = (
                random.random() < teacher_forcing_ratio
            )

            input_token = (
                target[:, t] if teacher_force else best_token
            )

        return outputs

    # @torch.no_grad()
    # def generate(self, source, bos_id, eos_id, max_length = 256):

    #     self.eval()
    #     lengths = (source != 0).sum(dim = 1)
        
    #     encoder_outputs, hidden, cell = self.encoder(
    #         source,
    #         lengths
    #     )

    #     mask = source != 0

    #     input_token = torch.tensor(
    #         [bos_id],
    #         device = source.device
    #     )

    #     generated = []

    #     for _ in range(max_length):

    #         prediction, hidden, cell =  self.decoder(
    #             input_token,
    #             hidden,
    #             cell,
    #             encoder_outputs,
    #             mask
    #         )

    #         next_token = prediction.argmax(dim = 1)

    #         if next_token.item() == eos_id:
    #             break

    #         generated.append(
    #             next_token.item()
    #         )

    #         input_token = next_token

    #     return generated