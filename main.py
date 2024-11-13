import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger
import matplotlib.pyplot as plt
from cfg.params import *  # noqa: F403
from src.train import train_model
from src.models.LSTMNet import LSTMNet_model
from src.data.make_dataset import prepare_data_loaders
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logger.add("logs/training_logs.log", level="INFO")


def main():
    # Prepare data loaders
    train_dataloader, test_dataloader = prepare_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        emb_dim=EMB_DIM,  # noqa: F405
    )  # noqa: F405

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize model
    model = LSTMNet_model(
        embedding_dim=EMB_DIM,
        hidden_dim=NUM_HIDDEN_NODES,
        output_dim=NUM_OUTPUT_NODES,
        n_layers=NUM_LAYERS,
        bidirectional=BIDIRECTION,
        dropout=DROPOUT,
    )
    model = model.to(device)
    logger.info("Model initialized and moved to device")

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss().to(device)
    logger.info("Optimizer and loss function set up")

    # Set random seed for reproducibility
    torch.manual_seed(0)
    num_epochs = 10

    # Train the model
    train_loss_hist, test_loss_hist = train_model(
        model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        num_epochs,
        device,
    )
    logger.info("Training completed")

    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label="Training Loss")
    plt.plot(test_loss_hist, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join("src", "visualization", "loss_plot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
