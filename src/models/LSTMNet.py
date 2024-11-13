import torch
import torch.nn as nn


class LSTMNet_model(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super(LSTMNet_model, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, embedded, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)

        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        dense_outputs = self.fc(hidden)

        outputs = self.sigmoid(dense_outputs)

        return outputs
