"""Image loading and preprocessing utilities for neural style transfer.

Provides functions to load images and normalize them for VGG19 input using
ImageNet statistics (mean and std).
"""

from PIL import Image
import torch
from torchvision import transforms

# ImageNet normalization constants (from torchvision pretraining)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB.
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL Image object in RGB format
    """
    return Image.open(path).convert("RGB")

def build_transform() -> transforms.Compose:
    """Build image transformation pipeline for model input.
    
    Converts PIL images to tensors and applies ImageNet normalization.
    
    Returns:
        Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor in range [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize with ImageNet stats
    ])

def process_image(path: str, device=None) -> torch.Tensor:
    """Load and preprocess an image for neural style transfer.
    
    Handles loading, normalization, and batching.
    
    Args:
        path: Path to the image file
        device: Torch device to move tensor to (optional)
        
    Returns:
        Normalized tensor of shape [1, 3, H, W] on specified device
    """
    transform = build_transform()
    image = load_image(path)
    # Add batch dimension: [3, H, W] -> [1, 3, H, W]
    tensor = transform(image).unsqueeze(0)

    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)

    return tensor.to(torch.float32)


def load_image_for_vgg(path: str, device, size=None) -> torch.Tensor:
    """Load and preprocess an image for VGG19 input with optional resizing.
    
    Uses the same ImageNet normalization as process_image. This is an alternative
    to process_image that allows specifying a target size.
    
    Args:
        path: Path to the image file
        device: Torch device to move tensor to
        size: Optional resize size as int or (H, W) tuple
    
    Returns:
        Normalized tensor of shape [1, 3, H, W] on specified device
    """
    img = Image.open(path).convert("RGB")

    # Build transformation pipeline
    tfms = []
    if size is not None:
        # Resize to specified dimensions
        tfms.append(transforms.Resize(size))
    tfms += [
        transforms.ToTensor(),  # Convert to tensor in range [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet normalization
    ]

    # Apply transformations: PIL image -> [1, 3, H, W] tensor
    x = transforms.Compose(tfms)(img).unsqueeze(0).to(device)
    return x