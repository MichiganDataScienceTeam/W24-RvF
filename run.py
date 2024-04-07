from submission.dataset import get_loaders
from submission.train import train_model, plot_performance, load_model
from submission.model import Model
import torch
import os

# TODO: To test locally, change dataset from `data/rvk140k` to whichever dataset you use locally
train_loader, val_loader = get_loaders(data_directory="data/rvf140k")

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = torch.nn.CrossEntropyLoss()

checkpoint_dir = f"checkpoints_{os.getenv('SLURM_JOB_ID')}"
history = train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs=5,
    checkpoint_dir=checkpoint_dir,
)
plot_performance(history)
