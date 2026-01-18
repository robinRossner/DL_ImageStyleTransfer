import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from loader import process_image
from nst_model import VGGFeatures
from torchvision.utils import save_image

"""
content image ─► encoder (VGG) ─► content features
style image   ─► encoder (VGG) ─► style features
content feat + style feat ─► AdaIN ─► stylized features ─► decoder ─► output image
Adaptive Instance Normalization (AdaIN) performs arbitrary style transfer in a single
forward pass by aligning the channel-wise mean and variance of content features to those
of the style features. A tunable parameter α controls the strength of style transfer,
enabling smooth interpolation between content preservation and stylization.
"""

def calc_mean_std(feat, eps=1e-5):
    """
    feat: [B, C, H, W]
    returns mean and std: [B, C, 1, 1]
    """
    B, C = feat.size()[:2]
    feat_var = feat.view(B, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(B, C, 1, 1)
    feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)

    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


class Decoder(nn.Module):
    """
    Simple decoder architecture used in AdaIN papers
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)


def adain_stylize(content_path, style_path, alpha=1.0, device="cuda"):
    # Load images
    content = process_image(content_path, device=device)
    style   = process_image(style_path, device=device)

    # Encoder (VGG up to relu4_1)
    vgg = VGGFeatures().to(device).eval()
    decoder = Decoder().to(device).eval()

    with torch.no_grad():
        c_feat = vgg(content)[20]  # relu4_1
        s_feat = vgg(style)[20]

        t = adain(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat

        out = decoder(t)

    return out

c_p = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_22.jpg"
s_p = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/style/processed/test_style_gogh.png"
output_tensor = adain_stylize(c_p, s_p, alpha=0.5, device="mps")
save_image(output_tensor, "stylized_output.jpg")
print("Image saved successfully!")