import torch
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer
from models.simple_nn import SimpleNN
import torch.nn as nn
import torch.optim as optim


async def handle_client(websocket):
    print("Client connected!")

    # Example dataset
    inputs = torch.randn(100, 3)
    targets = torch.randint(0, 2, (100, 2)).float()

    # Create DataLoader
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training after client connects
    await Trainer.train(model, train_loader, criterion, optimizer, websocket)
