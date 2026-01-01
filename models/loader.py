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

def process_image(path, transform, device=None):
    image = load_image(path)
    tensor = transform(image).unsqueeze(0)  # add batch dim

    if device is not None:
        tensor = tensor.to(device)

    return tensor.to(torch.float32)