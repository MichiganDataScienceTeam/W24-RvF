from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
seaborn.set_theme()


def save_model(
    epoch: int,
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
):
    """
    Save the model to a file.

    Args:
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to be saved.
        checkpoint_dir (Path): The directory to save the model to.
    """
    checkpoint_dir = Path(checkpoint_dir) / model.__class__.__name__
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        checkpoint_dir / f"model_{epoch}.pt",
    )


def load_model(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    epoch: int,
):
    """
    Load the model from a file.

    Args:
        model (torch.nn.Module): The model to be loaded.
        checkpoint_dir (Path): The directory to load the model from.
        epoch (int): The epoch number of the model to load.
    """
    checkpoint_dir = Path(checkpoint_dir)
    model.load_state_dict(
        torch.load(checkpoint_dir / model.__class__.__name__ / f"model_{epoch}.pt")
    )


def evaluate(
    model: torch.nn.Module, criterion: Callable, loader: DataLoader
) -> tuple[float]:
    """
    Evaluate the performance of a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (Callable): The loss function used for evaluation.
        loader (DataLoader): The data loader containing the evaluation data.

    Returns:
        tuple[float]: A tuple containing the accuracy and average loss.

    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        loss = 0.0
        for X, y in loader:
            outputs = model(X.to(device))
            if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                logits = outputs.logits
            else:
                logits = outputs
            loss += criterion(logits, y.to(device)).detach().sum().item()
            _, predicted = torch.max(logits.data, 1)  # get predicted digit
            y = y.to(device)  # move y to the same device as predicted
            total += len(y)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total, loss / total


def train_model(
    model: torch.nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    save_checkpoint: bool = True,
) -> dict[str, list[float]]:
    """
    Train a given model using the specified criterion, optimizer, and data loaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (Callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the test dataset.
        epochs (int, optional): The number of training epochs. Defaults to 10.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and test losses and accuracies.
    """
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        for X, y in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(X.to(device))
            if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                logits = outputs.logits
            else:
                logits = outputs
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()

        if save_checkpoint:
            save_model(epoch, model, Path("checkpoints"))

        train_accuracy, train_loss = evaluate(model, criterion, train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_accuracy, test_loss = evaluate(model, criterion, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}: Loss - (Train {train_loss:.2f}/Test {test_loss:.2f}, "
            f"Accuracy - (Train {train_accuracy:.2f}/Test {test_accuracy:.2f})"
        )

    return {
        "loss": {
            "train": train_losses,
            "test": test_losses,
        },
        "accuracy": {
            "train": train_accuracies,
            "test": test_accuracies,
        },
    }
