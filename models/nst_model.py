"""
Neural Style Transfer (NST) using VGG19 features and LBFGS optimization.

This module implements the classic Gatys et al. (2015) neural style transfer
algorithm with LBFGS optimization for high-quality stylization.
"""

import os
from typing import Dict, Optional
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

from loader import process_image

# ImageNet normalization constants - must match loader.py
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# Load configuration
with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]

# Set compute device (GPU if available, else CPU)
if device_config == "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)
print("Using device:", device)


def denorm_and_save(x: torch.Tensor, path: str) -> None:
    """
    Denormalize VGG-normalized image and save to disk.
    
    Converts from ImageNet-normalized space (as used during optimization)
    back to [0, 1] pixel space, clamps values, and saves as PNG.
    
    Args:
        x: Tensor of shape (1, 3, H, W) in VGG-normalized space
        path: Output file path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Remove batch dimension and move to CPU
    x = x.detach().cpu().squeeze(0)
    
    # Reshape normalization constants for broadcasting
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)

    # Denormalize: x' = (x - mean) / std -> x = x' * std + mean
    x = x * std + mean
    x = x.clamp(0.0, 1.0)
    save_image(x, path)


class VGGFeatures(nn.Module):
    """
    Pretrained VGG19 feature extractor for NST.
    
    Extracts intermediate feature maps from specific layers used for
    computing content and style losses. Layer indices correspond to
    ReLU activations in VGG19.features.
    
    Layer mapping:
        relu1_1 = 1,   relu2_1 = 6,   relu3_1 = 11,  relu4_1 = 20,
        relu4_2 = 22 (content),      relu5_1 = 29
    """
    
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 and freeze weights
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        # Style loss computed from multiple layers (multi-scale)
        self.style_layers = [1, 6, 11, 20, 29]
        # Content loss from deeper layer (relu4_2)
        self.content_layer = 22

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Dictionary mapping layer indices to feature tensors
        """
        feats: Dict[int, torch.Tensor] = {}
        needed = set(self.style_layers + [self.content_layer])

        # Forward pass through VGG, collecting intermediate features
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in needed:
                feats[i] = x
            if i >= max(needed):
                # Early exit: stop once all needed layers are extracted
                break
        return feats


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute unnormalized Gram matrix from feature maps.
    
    The Gram matrix captures style by computing correlations between
    feature channels. This unnormalized version is normalized per-layer
    during loss computation for paper-style normalization.
    
    Args:
        feat: Feature tensor of shape (B, C, H, W)
        
    Returns:
        Gram matrix of shape (B, C, C)
    """
    B, C, H, W = feat.shape
    # Reshape to (B, C, H*W) - each row is a feature channel flattened
    F = feat.view(B, C, H * W)
    # Compute Gram matrix: G_ij = sum_k F_ik * F_jk
    G = torch.bmm(F, F.transpose(1, 2))
    return G


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error between two tensors."""
    return torch.mean((a - b) ** 2)


def clamp_normalized_(x: torch.Tensor) -> None:
    """
    Clamp image to valid pixel range in-place, then renormalize.
    
    After each optimization step, clamp pixel values to [0, 1], then
    transform back to VGG-normalized space. This keeps optimization
    stable by preventing degenerate pixel values.
    
    Args:
        x: VGG-normalized tensor to clamp in-place
    """
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    with torch.no_grad():
        # Denormalize to pixel space
        px = x * std + mean
        # Clamp to valid range
        px.clamp_(0.0, 1.0)
        # Renormalize back to VGG space
        x.copy_((px - mean) / std)


def neural_style_transfer_lbfgs(
    content_path: str,
    style_path: str,
    out_path: str = "nst_out.png",
    steps: int = 1200,
    alpha: float = 1.0,
    beta: float = 1e4,
    save_every: int = 400,
    style_layer_weights: Optional[Dict[int, float]] = None,
    content: Optional[torch.Tensor] = None,
    style: Optional[torch.Tensor] = None
) -> None:
    """
    Neural Style Transfer using LBFGS optimization (Gatys et al., 2015).
    
    Optimizes an image to minimize content loss (match content image features)
    and style loss (match style image texture/patterns).
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        out_path: Output file path
        steps: Number of LBFGS optimization steps
        alpha: Content loss weight
        beta: Style loss weight
        save_every: Save intermediate results every N steps (unused in current version)
        style_layer_weights: Per-layer style loss weights (default: equal weights)
        content: Precomputed content image tensor (optional)
        style: Precomputed style image tensor (optional)
    """
    
    # Load images using the image loader
    if content is None:
        content = process_image(content_path, device=device)
    if style is None:
        style = process_image(style_path, device=device)

    # Initialize VGG feature extractor
    vgg = VGGFeatures().to(device)

    # Set default equal weights for all style layers
    if style_layer_weights is None:
        style_layer_weights = {l: 1.0 for l in vgg.style_layers}

    # Precompute content and style targets (frozen during optimization)
    with torch.no_grad():
        c_feats = vgg(content)
        s_feats = vgg(style)

        # Content target: features from content image
        content_target = c_feats[vgg.content_layer]
        # Style targets: Gram matrices from style image
        style_targets = {l: gram_matrix(s_feats[l]) for l in vgg.style_layers}

    # Initialize optimized image from content image
    target = content.clone().requires_grad_(True)

    # Use LBFGS optimizer for high-quality results
    optimizer = optim.LBFGS([target], max_iter=1, history_size=50, line_search_fn="strong_wolfe")

    # Track losses for logging
    last_c = 0.0
    last_s = 0.0

    # Optimization loop
    for step in range(1, steps + 1):

        def closure():
            """Closure function for LBFGS: compute loss and gradients."""
            nonlocal last_c, last_s
            optimizer.zero_grad(set_to_none=True)

            # Extract features from current image
            feats = vgg(target)

            # Content loss: MSE between feature activations
            c_loss = mse(feats[vgg.content_layer], content_target)

            # Style loss: weighted sum over layers with Gatys et al. normalization
            s_loss = torch.tensor(0.0, device=device)
            for l in vgg.style_layers:
                w_l = style_layer_weights.get(l, 1.0)

                # Get feature map dimensions
                B, C, H, W = feats[l].shape
                N = C  # Number of channels
                M = H * W  # Spatial size

                # Compute Gram matrix for current features
                G = gram_matrix(feats[l])
                A = style_targets[l]  # Precomputed target Gram matrix

                # Layer-wise style loss (MSE of Gram matrices)
                layer_loss = torch.mean((G - A) ** 2)
                # Paper normalization: divide by (4 * N^2 * M^2)
                layer_loss = layer_loss / (4.0 * (N ** 2) * (M ** 2))

                s_loss = s_loss + w_l * layer_loss

            # Total loss: weighted combination of content and style
            total = alpha * c_loss + beta * s_loss
            total.backward()

            # Store for logging
            last_c = float(c_loss.detach())
            last_s = float(s_loss.detach())

            return total

        # Optimization step
        loss = optimizer.step(closure)
        # Clamp to valid pixel range to maintain stability
        clamp_normalized_(target)

        # Logging
        if step % 50 == 0 or step == 1:
            if step == 1:
                print("Starting NST optimization...")
                start_time = time.time()
            elif step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{steps} done. Elapsed time: {elapsed:.2f} seconds.")
            else:
                print(f"Step {step}/{steps} done.")
        
        # Save final result
        if step == steps:
            step_path = out_path.replace(".png", f"_step{step}.png")
            denorm_and_save(target, step_path)
            print(
                f"[{step}/{steps}] "
                f"total={float(loss):.4f} content={last_c:.6f} style={last_s:.6f} "
                f"saved={step_path}"
            )
def calc_grid_betaxstep(content: str, style: str, steps: int, name: str) -> None:
    """
    Grid search over beta (style weight) values.
    
    Evaluates NST with different beta values to find optimal style weight.
    
    Args:
        content: Path to content image
        style: Path to style image
        steps: Number of optimization steps
        name: Output name prefix
    """
    beta_vals = [1e4, 1e5, 1e6, 1e7, 1e8]
    for beta in beta_vals:
        neural_style_transfer_lbfgs(
            content_path=content,
            style_path=style,
            out_path=output_dir + f"/grid_search/{name}_beta_{beta}.png",
            steps=steps,
            alpha=1.0,
            beta=beta,
            save_every=400
        )


def folder_nst() -> None:
    """
    Process all content-style image pairs in configured folders.
    
    Runs NST on all combinations of images in content and style folders,
    evaluates results using LPIPS/FID metrics, and logs to CSV.
    """
    content_folder = data_dir + "/content/processed/"
    style_folder = data_dir + "/style/processed/"

    # Get all image files
    content_images = [f for f in os.listdir(content_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    step = 1200
    beta = 1e7
    
    # Process each content image with each style image
    for i, content_img in enumerate(content_images):
        print("Processing content image:", content_img)
        print("Progress:" + str(i + 1) + "/" + str(len(content_images)))
        content_path = os.path.join(content_folder, content_img)
        content = process_image(content_path, device=device)

        for j, style_img in enumerate(style_images):
            print(f"  Progress:" + str(j + 1) + "/" + str(len(style_images)))
            style_path = os.path.join(style_folder, style_img)
            
            # Extract image IDs from filenames
            name = f"style{style_img.split('_')[1].split('.')[0]}_img{content_img.split('_')[1].split('.')[0]}"
            out_path = output_dir + f"/nst_all/{name}_beta_{beta}.png"

            # Run NST
            neural_style_transfer_lbfgs(
                content_path=content_path,
                style_path=style_path,
                out_path=out_path,
                steps=step,
                alpha=1.0,
                beta=beta,
                save_every=1200,
                content=content
            )
            
            # Determine output path for evaluation
            out_path_step = out_path.replace(".png", f"_step{step}.png")
            eval_out_path = out_path_step if os.path.exists(out_path_step) else out_path

            # Evaluate and log metrics
            from eval import eval_triplet_and_log
            csv_path = os.path.join(output_dir, "metrics", "metrics.csv")
            eval_triplet_and_log(
                out_path=eval_out_path,
                content_path=content_path,
                style_path=style_path,
                method_name=f"nst_beta_{beta}",
                csv_path=csv_path,
                device=device,
            )

if __name__ == "__main__":
    # Example: Single content-style pair
    content_path = data_dir + "/content/processed/img_26.jpg"
    style_path = data_dir + "/style/processed/style_3.jpg"
    step = 1200
    beta = 1e7
    name = "style3_img26"
    out_path = output_dir + "/plump.png"
    
    neural_style_transfer_lbfgs(
        content_path=content_path,
        style_path=style_path,
        out_path=out_path,
        steps=step,
        alpha=1.0,
        beta=beta,
        save_every=1200
    )
