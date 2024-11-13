import unittest
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/models"))
)
from LSTMNet import LSTMNet_model


class TestLSTMNet(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 100
        self.hidden_dim = 50
        self.output_dim = 1
        self.n_layers = 2
        self.bidirectional = True
        self.dropout = 0.5
        self.model = LSTMNet_model(
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.bidirectional,
            self.dropout,
        )

    def test_forward(self):
        batch_size = 4
        seq_lengths = [10, 8, 6, 4]
        embedded = [torch.randn(length, self.embedding_dim) for length in seq_lengths]
        embedded_padded = pad_sequence(embedded, batch_first=True)
        text_lengths = torch.tensor(seq_lengths)

        outputs = self.model(embedded_padded, text_lengths)
        self.assertEqual(outputs.shape, (batch_size, self.output_dim))

    def test_bidirectional(self):
        self.assertTrue(self.model.lstm.bidirectional)

    def test_num_layers(self):
        self.assertEqual(self.model.lstm.num_layers, self.n_layers)

    def test_dropout(self):
        self.assertEqual(self.model.lstm.dropout, self.dropout)


if __name__ == "__main__":
    unittest.main()
