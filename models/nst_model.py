from loader import process_image
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

import torch.optim as optimization

import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")  # RTX 3060 Ti
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def content_loss(target_feature, content_feature):
    return torch.mean((target_feature - content_feature) ** 2)

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    return torch.mm(features, features.t())

def style_loss(target, style):
    G = gram_matrix(target)
    S = gram_matrix(style)
    return torch.mean((G - S) ** 2)

def save(target, name):
  #the image needs to be denormalized first
  denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
  #remove the additional dimension
  img = target.clone().squeeze()
  img = denormalization(img).clamp(0, 1)
  save_image(img, f'{name}.png')

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['19', '21']  # conv4_2 + conv4_3

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        style_features = []
        content_feature = None

        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                style_features.append(x)
            if name == self.content_layer:
                content_feature = x

        return style_features, content_feature

def run_and_save(content_dir, style_dir, vgg, steps, alpha, beta, lr=0.001, name="nst_result"):
    """Runs neural style transfer and saves the output image.
    Args:
        content_dir (str): Path to the content image.
        style_dir (str): Path to the style image.
        vgg (nn.Module): Pretrained VGG model for feature extraction.
        steps (int): Number of optimization steps.
        alpha (float): Weight for content loss.
        beta (float): Weight for style loss.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        name (str, optional): Name for the saved output image. Defaults to "nst_result".
    """
    content_img = process_image(content_dir, device)
    style_img = process_image(style_dir, device)

    target_img = content_img.clone().requires_grad_(True)
    optimizer = optimization.Adam([target_img], lr=lr)

    with torch.no_grad():
        style_features, _ = vgg(style_img)
        _, content_feature = vgg(content_img)

    for step in tqdm(range(steps)):
        target_style_features, target_content_feature = vgg(target_img)

        c_loss = content_loss(target_content_feature, content_feature)

        s_loss = 0
        for t, s in zip(target_style_features, style_features):
            s_loss += style_loss(t, s)

        total_loss = alpha * c_loss + beta * s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    save(target_img, name)

#load the model
vgg = VGG().to(device).eval()

content = "test_content_dog_128.png"
style = "test_style_spiral_128.jpg"

run_and_save(content, style, vgg, 10000, 10, 100000, lr=0.0005, name="high_style_weight")
run_and_save(content, style, vgg, 10000, 10, 10000, lr=0.0005, name="medium_style_weight")
run_and_save(content, style, vgg, 10000, 10, 1000, lr=0.0005, name="low_style_weight")
run_and_save(content, style, vgg, 10000, 10, 100, lr=0.0005, name="very_low_style_weight")