from loader import load_image
import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
vgg = vgg.to(device).eval()

for param in vgg.parameters():
    param.requires_grad = False
