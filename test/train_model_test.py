import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from train import train_model, binary_accuracy


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, text_lengths):
        return self.linear(x)


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.model = DummyModel().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCEWithLogitsLoss()

        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100, 1)).float()
        train_dataset = TensorDataset(X_train, y_train, torch.ones(100))
        self.train_dataloader = DataLoader(train_dataset, batch_size=10)

        X_test = torch.randn(20, 10)
        y_test = torch.randint(0, 2, (20, 1)).float()
        test_dataset = TensorDataset(X_test, y_test, torch.ones(20))
        self.test_dataloader = DataLoader(test_dataset, batch_size=10)

    def test_train_model(self):
        train_loss_hist, test_loss_hist = train_model(
            self.model,
            self.optimizer,
            self.criterion,
            self.train_dataloader,
            self.test_dataloader,
            num_epochs=1,
            device=self.device,
        )
        self.assertEqual(len(train_loss_hist), 1)
        self.assertEqual(len(test_loss_hist), 1)

    def test_binary_accuracy(self):
        preds = torch.tensor([0.2, 0.8, 0.6, 0.4])
        y = torch.tensor([0, 1, 1, 0]).float()
        acc = binary_accuracy(preds, y)
        self.assertAlmostEqual(acc.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
