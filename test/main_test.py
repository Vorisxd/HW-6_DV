import unittest
import torch
import sys
import os
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.models.LSTMNet import LSTMNet_model
from src.data.make_dataset import prepare_data_loaders
from src.train import train_model
from cfg.params import *


class TestTrainingProcess(unittest.TestCase):
    def setUp(self):
        # Prepare data loaders
        self.train_dataloader, self.test_dataloader = prepare_data_loaders(
            data_path=DATA_PATH,
            batch_size=BATCH_SIZE,
            emb_dim=EMB_DIM,
        )

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        self.model = LSTMNet_model(
            embedding_dim=EMB_DIM,
            hidden_dim=NUM_HIDDEN_NODES,
            output_dim=NUM_OUTPUT_NODES,
            n_layers=NUM_LAYERS,
            bidirectional=BIDIRECTION,
            dropout=DROPOUT,
        )
        self.model = self.model.to(self.device)

        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.BCELoss().to(self.device)

        # Set random seed for reproducibility
        torch.manual_seed(0)
        self.num_epochs = 1  # Use 1 epoch for testing to save time

        # Configure logger
        logger.add("/logs/training_logs.log", level="INFO")

    def test_training_process(self):
        # Train the model
        train_loss_hist, test_loss_hist = train_model(
            self.model,
            self.optimizer,
            self.criterion,
            self.train_dataloader,
            self.test_dataloader,
            self.num_epochs,
            self.device,
        )

        # Check if training and testing loss histories are not empty
        self.assertTrue(len(train_loss_hist) > 0, "Training loss history is empty")
        self.assertTrue(len(test_loss_hist) > 0, "Testing loss history is empty")

    def test_logs_recording(self):
        # Check if logs are recorded in the specified log file
        log_file_path = "/logs/training_logs.log"
        self.assertTrue(os.path.exists(log_file_path), "Log file does not exist")
        with open(log_file_path, "r") as log_file:
            logs = log_file.read()
            self.assertIn("Using device:", logs, "Device log not found")
            self.assertIn(
                "Model initialized and moved to device",
                logs,
                "Model initialization log not found",
            )
            self.assertIn(
                "Optimizer and loss function set up",
                logs,
                "Optimizer and loss function setup log not found",
            )
            self.assertIn(
                "Training completed", logs, "Training completion log not found"
            )


if __name__ == "__main__":
    unittest.main()
