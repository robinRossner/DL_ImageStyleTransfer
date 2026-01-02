from loader import process_image
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

import torch.optim as optimization

import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def get_content_loss(target_feature, content_feature):
    return torch.mean((target_feature - content_feature) ** 2)

def gram_matrix(input, c, h, w):
  #c-channels; h-height; w-width 
  input = input.view(c, h*w) 
  #matrix multiplication on its own transposed form
  G = torch.mm(input, input.t())
  return G
  
def get_style_loss(target, style):
  _, c, h, w = target.size()
  G = gram_matrix(target, c, h, w) #gram matrix for the target image
  S = gram_matrix(style, c, h, w) #gram matrix for the style image
  return torch.mean((G-S)**2)/(c*h*w)

def save(target, name):
  #the image needs to be denormalized first
  denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
  #remove the additional dimension
  img = target.clone().squeeze()
  img = denormalization(img).clamp(0, 1)
  save_image(img, f'{name}.png')

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.select_features = ['0', '5', '10', '19', '28'] #conv layers
    self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
  
  def forward(self, output):
    features = []
    for name, layer in self.vgg._modules.items():
      output = layer(output)
      if name in self.select_features:
        features.append(output)
    return features

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
    content_img = process_image(content_dir)
    style_img = process_image(style_dir)

    target_img = content_img.clone().requires_grad_(True)
    optimizer = optimization.Adam([target_img], lr=lr)

    for step in tqdm(range(steps)):
    #get feature vectors representations for every image
      target_feature = vgg(target_img)
      content_feature = vgg(content_img)
      style_feature = vgg(style_img)

      style_loss = 0
      content_loss = 0

      for target, content, style in zip(target_feature, content_feature, style_feature):
          content_loss += get_content_loss(target, content)
          style_loss += get_style_loss(target, style)
    
      total_loss = alpha*content_loss+beta*style_loss
    
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

    save(target_img, name)

#load the model
vgg = VGG().to(device).eval()

model = vgg19(pretrained=True).features

run_and_save("test_content_dog_128.png", "test_style_spiral_128.jpg", vgg, 10000, 1, 100000, lr=0.003, name="high_style_weight")
run_and_save("test_content_dog_128.png", "test_style_spiral_128.jpg", vgg, 10000, 1, 10000, lr=0.003, name="medium_style_weight")
run_and_save("test_content_dog_128.png", "test_style_spiral_128.jpg", vgg, 10000, 1, 1000, lr=0.003, name="low_style_weight")
run_and_save("test_content_dog_128.png", "test_style_spiral_128.jpg", vgg, 10000, 1, 100, lr=0.003, name="very_low_style_weight")