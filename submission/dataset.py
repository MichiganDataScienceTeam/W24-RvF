import torch
from torchvision import transforms
import numpy.typing as npt

def preprocess(image: npt.ArrayLike) -> torch.Tensor:
    """
    Preprocesses an image by applying a series of transformation.

    Args:
        image (npt.ArrayLike): The input image to be preprocessed.

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    # Convert image to tensor
    tensor = torch.tensor(image, dtype=torch.float32)
     mean = [0.485, 0.456, 0.406]  # Mean values for the RGB channels
    std = [0.229, 0.224, 0.225]    # Standard deviation values for the RGB channels
    tensor = torch.tensor(image, dtype=torch.float32)
    custom_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip
        transforms.RandomRotation(degrees=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Permute to (3, 224, 224)
        transforms.Resize((224, 224)),  # Resize image to (224, 224)
        transforms.Normalize(mean=mean, std=std), 
    ])
    # Add other transformations as needed
    transformed_tensor = custom_transforms(tensor)
    return transformed_tensor

import torch
from torchvision import transforms
import numpy.typing as npt

def preprocess2(image: npt.ArrayLike) -> torch.Tensor:
    """
    Preprocesses an image by applying a series of transformation.

    Args:
        image (npt.ArrayLike): The input image to be preprocessed.

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    # Convert image to tensor
    tensor = torch.tensor(image, dtype=torch.float32)
     mean = [0.485, 0.456, 0.406]  # Mean values for the RGB channels
    std = [0.229, 0.224, 0.225]    # Standard deviation values for the RGB channels
    tensor = torch.tensor(image, dtype=torch.float32)
    custom_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip
        transforms.RandomRotation(degrees=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # zoom range of 0.8 to 1.2
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # shift range of 20% of image size
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Permute to (3, 224, 224)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.5)
        transforms.Resize((224, 224)),  # Resize image to (224, 224)
        transforms.Normalize(mean=mean, std=std), 
    ])
    # Add other transformations as needed
    transformed_tensor = custom_transforms(tensor)
    return transformed_tensor


















