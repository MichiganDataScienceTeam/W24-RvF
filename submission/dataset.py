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
    custom_transforms = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=1),  # horizontal flip
        transforms.RandomRotation(degrees=(-20, 20)),  # rotation range of -20 to 20 degrees
        transforms.RandomAffine(degrees=0, shear=(-20, 20)),  # shear range of -20 to 20 degrees
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # zoom range of 0.8 to 1.2
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # shift range of 20% of image size
    ])
    
    # Add other transformations as needed
    transformed_tensor = custom_transforms(tensor)
    return transformed_tensor









