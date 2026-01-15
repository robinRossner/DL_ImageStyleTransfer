from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")

def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def process_image(path, device=None):
    transform = build_transform()
    image = load_image(path)
    tensor = transform(image).unsqueeze(0)  # add batch dim

    if device is not None:
        tensor = tensor.to(device)

    return tensor.to(torch.float32)


def load_image_for_vgg(path: str, device, size=None):
    """
    Load an image and prepare it for VGG input.
    Uses the same normalization (ImageNet mean/std) as process_image.
    
    Args:
        path: Path to the image file
        device: torch device to move tensor to
        size: Optional resize size (int or (H,W) tuple)
    
    Returns:
        Tensor of shape [1, 3, H, W] normalized for VGG
    """
    img = Image.open(path).convert("RGB")

    tfms = []
    if size is not None:
        # size can be int or (H,W) depending on how you do it elsewhere
        tfms.append(transforms.Resize(size))
    tfms += [
        transforms.ToTensor(),  # -> [3,H,W] in 0..1
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    x = transforms.Compose(tfms)(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x