"""
MDST Real vs Fake Face Detection Project - Winter 2024

dataset.py - Starter code for loading and preprocessing the dataset.
"""

from pathlib import Path
from typing import Union, Callable
import pandas as pd
import torch
import numpy.typing as npt
import torchvision
import imageio.v3 as iio

class RvFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        data_directory: Union[str, Path] = "data/rvf10k",
        preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
    ):
        self.data_directory = Path(data_directory)
        self.metadata = pd.read_csv(self.data_directory / f"{split}.csv")

        if preprocessor is None:
            self.preprocessor = torchvision.transforms.ToTensor()
        else:
            self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        image_metadata = self.metadata.iloc[index]
        path = self.data_directory / image_metadata["path"]
        image = self.preprocessor(iio.imread(path))
        return image, image_metadata["label"]

mean = torch.zeros((3,))
variance = torch.zeros((3,))
tensor_converter = v2.ToTensor()
train_dataset = RvFDataset("train", "data/rvf140k")

for image, _ in train_dataset:
  mean += tensor_converter(image).mean(dim=(1, 2))
mean /= len(train_dataset)

for image, _ in train_dataset:
  image = tensor_converter(image)
  variance += ((image - mean.view(3, 1, 1))**2).mean(dim=(1, 2))
std = torch.sqrt(variance/len(train_dataset))

def preprocess(image) -> torch.Tensor:
  """
  Preprocesses an image by applying a series of transformation.

  Args:
      image (npt.ArrayLike): The input image to be preprocessed.

  Returns:
      torch.Tensor: The preprocessed image as a tensor.
  """
  tensor_converter = v2.Compose([ # Step 0: Convert from PIL Image to Torch Tensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
  ])
  normalizer = v2.Normalize(mean=mean, std=std)

  preprocessor = v2.Compose([
    tensor_converter,
    normalizer,
  ])
  return preprocessor(image)

def get_loaders(
    batch_size: int = 32,
    preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = preprocess,
    pin_memory: bool = False,
    data_directory: Union[str, Path] = "data/rvf10k",
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns the DataLoader objects for the training and validation sets.

    Args:
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of workers for the DataLoader. Defaults to None.
        pin_memory (bool): Whether to pin memory in the DataLoader. Defaults to False.
        data_directory (Union[str, Path]): The directory where the data is stored.
        preprocessor (Callable[[npt.ArrayLike], torch.Tensor]): The preprocessor function to use.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The DataLoader objects for the training and validation sets.
    """
    train_loader = torch.utils.data.DataLoader(
        RvFDataset("train", data_directory, preprocessor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        RvFDataset("valid", data_directory, preprocessor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
