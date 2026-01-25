"""Adaptive Instance Normalization (AdaIN) for Neural Style Transfer.

Implements the AdaIN-based style transfer method from Huang & Belongie (2017).
Uses a pretrained VGG encoder, AdaIN feature alignment, and a learned decoder
for efficient, feed-forward style transfer.

References:
    Huang et al. (2017): "Arbitrary Style Transfer in Real-time with Adaptive
    Instance Normalization"
"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, utils
import os
import yaml

# Load configuration
with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]


def load_adain_image(path: str, device, size: int = 512) -> torch.Tensor:
    """Load and preprocess image for AdaIN stylization.
    
    Loads image in raw [0, 1] range without ImageNet normalization,
    resizes and center-crops to square dimensions.
    
    Args:
        path: Path to image file
        device: Torch device for tensor
        size: Target square size in pixels (default 512)
        
    Returns:
        Preprocessed tensor of shape [1, 3, size, size] on device
    """
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize(size),  # Resize to fit in square
        transforms.CenterCrop(size),  # Center-crop to exact size
        transforms.ToTensor(),  # Convert to [0, 1] range
    ])
    return t(img).unsqueeze(0).to(device)

class VGGEncoder(nn.Module):
    """VGG-based encoder for feature extraction up to relu4_1.
    
    Implements the VGG architecture from Huang & Belongie (2017) using
    ReflectionPad2d for boundary handling. Loads pretrained weights and
    freezes all parameters for inference.
    """
    
    def __init__(self, weights_path: str, device):
        """Initialize VGG encoder with pretrained weights.
        
        Args:
            weights_path: Path to pretrained VGG weights
            device: Device to load weights on
        """
        super().__init__()
        self.vgg = nn.Sequential(
            # Initial color layer
            nn.Conv2d(3, 3, (1, 1)),
            # Block 1: relu1_1, relu1_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            # Block 2: relu2_1, relu2_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            # Block 3: relu3_1, relu3_2, relu3_3, relu3_4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            # Block 4: relu4_1 (bottleneck for AdaIN)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()  # relu4_1 - bottleneck layer
        )
        # Load pretrained weights with lenient matching
        self.vgg.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        # Freeze all encoder parameters (inference only)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features at relu4_1 level.
        
        Args:
            x: Input tensor of shape [B, 3, H, W] in [0, 1] range
            
        Returns:
            Feature tensor of shape [B, 512, H/8, W/8]
        """
        return self.vgg(x)

class Decoder(nn.Module):
    """Decoder network for reconstructing stylized images from features.
    
    Inverts the VGG encoder using upsampling and convolutions with ReflectionPad2d
    for boundary handling. Outputs images in [0, 1] range.
    """
    
    def __init__(self):
        """Initialize decoder architecture."""
        super().__init__()
        self.decoder = nn.Sequential(
            # Upsample and decode to conv3_4 level (256 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            # Upsample to conv2_2 level (128 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            # Upsample to conv1_2 level (64 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            # Output RGB image
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from encoded features.
        
        Args:
            x: Feature tensor from encoder
            
        Returns:
            Reconstructed image tensor of same spatial size as input
        """
        return self.decoder(x)

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple:
    """Calculate mean and standard deviation of feature tensor.
    
    Computes per-channel statistics across spatial dimensions, used for
    normalizing content features before AdaIN alignment.
    
    Args:
        feat: Feature tensor of shape [B, C, H, W]
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (mean, std) tensors, each shape [B, C, 1, 1]
    """
    N, C = feat.size()[:2]
    # Compute variance per channel
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # Compute standard deviation
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # Compute mean per channel
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Adaptive Instance Normalization - align content to style statistics.
    
    Transfers style by aligning the feature statistics of content with those of style.
    Normalizes content features to zero mean/unit variance, then shifts/scales using
    style statistics.
    
    Args:
        content_feat: Content feature tensor [B, C, H, W]
        style_feat: Style feature tensor [B, C, H, W]
        
    Returns:
        Aligned feature tensor with shape [B, C, H, W]
    """
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    # Normalize content: (x - mean) / std
    normalized = (content_feat - c_mean) / c_std
    # Denormalize with style statistics: normalized * style_std + style_mean
    return normalized * s_std + s_mean

def run_stylization(
    content_path: str,
    style_path: str,
    v_path: str,
    d_path: str,
    alpha: float,
    device
) -> torch.Tensor:
    """Run AdaIN stylization on content image using style reference.
    
    Encodes content and style images, performs AdaIN feature alignment,
    optionally blends with original content (alpha parameter), and decodes
    back to image space.
    
    Args:
        content_path: Path to content image
        style_path: Path to style reference image
        v_path: Path to encoder weights
        d_path: Path to decoder weights
        alpha: Blending factor (0=content, 1=full style)
        device: Torch device for computation
        
    Returns:
        Stylized image tensor of shape [1, 3, H, W] clamped to [0, 1]
    """
    # Load images
    content = load_adain_image(content_path, device)
    style = load_adain_image(style_path, device)

    # Initialize encoder and decoder
    encoder = VGGEncoder(v_path, device).to(device).eval()
    decoder = Decoder().to(device)
    decoder.decoder.load_state_dict(torch.load(d_path, map_location=device))
    decoder.eval()

    with torch.no_grad():
        # Extract features
        c_feat = encoder(content)
        s_feat = encoder(style)
        
        # Apply AdaIN for style transfer
        t = adain(c_feat, s_feat)
        # Interpolation for strength ablation: blend between aligned and original
        t = alpha * t + (1 - alpha) * c_feat
        
        # Decode back to image space
        output = decoder(t)
    
    # Clamp to valid pixel range
    return output.clamp(0, 1)

def folder_adain() -> None:
    """Process all content-style image pairs with AdaIN.
    
    Runs stylization on all combinations of images in content and style folders,
    evaluates results using LPIPS/FID metrics, and logs to CSV.
    """
    content_folder = data_dir + "/content/processed/"
    style_folder = data_dir + "/style/processed/"

    # Get all image files
    content_images = [f for f in os.listdir(content_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    alpha = 1.0
    
    # Process each content-style pair
    for i, content_img in enumerate(content_images):
        print("Processing content image:", content_img)
        print("Progress:" + str(i + 1) + "/" + str(len(content_images)))
        for j, style_img in enumerate(style_images):
            print(f"  Progress:" + str(j + 1) + "/" + str(len(style_images)))
            content_path = os.path.join(content_folder, content_img)
            style_path = os.path.join(style_folder, style_img)
            
            # Extract image IDs from filenames
            name = f"style{style_img.split('_')[1].split('.')[0]}_img{content_img.split('_')[1].split('.')[0]}"
            out_path = output_dir + f"/adain_all/{name}_alpha_{alpha}.png"

            # Load model weights
            v_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/vgg_normalised.pth"
            d_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/decoder.pth"
            
            # Run stylization
            result = run_stylization(content_path, style_path, v_weights, d_weights, alpha, device_config)
            utils.save_image(result, out_path)

            # Evaluate and log metrics
            from eval import eval_triplet_and_log
            csv_path = os.path.join(output_dir, "metrics", "metrics.csv")
            eval_triplet_and_log(
                out_path=out_path,
                content_path=content_path,
                style_path=style_path,
                method_name=f"adain_alpha_{alpha}",
                csv_path=csv_path,
                device=device_config,
            )

if __name__ == "__main__":
    # Strength ablation: test alpha blending with multiple styles on single content
    device = "mps"
    style_folder = data_dir + "/style/processed/"
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    content_img = data_dir + "/content/processed/img_9.jpg"
    
    # Apply each style to content with alpha=0.8 (80% stylization)
    for style_img in style_images:
        style_path = os.path.join(style_folder, style_img)
        name = f"style{style_img.split('_')[1].split('.')[0]}_img9"
        out_path = output_dir + f"/adain_ablation/{name}_alpha_0.8.png"
        
        # Load model weights
        v_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/vgg_normalised.pth"
        d_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/decoder.pth"
        
        result = run_stylization(content_img, style_path, v_weights, d_weights, 0.8, device)
        utils.save_image(result, out_path)
