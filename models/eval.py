"""Evaluation metrics for neural style transfer results.

Computes multiple quantitative metrics comparing output image to content and style images
using VGG19 features. Supports logging results to CSV for batch evaluation.

Metrics computed:
- Raw MSE: Mean squared error between features/Gram matrices
- Normalized Distance: MSE normalized by dynamic range (0 to 1)
- Log Score: Log-scaled metric similar to PSNR (0 to 100+)
"""

import os
import csv
import time
import argparse
import math
import torch
import yaml
from PIL import Image
from torchvision import transforms

from nst_model import VGGFeatures, gram_matrix, mse

# ImageNet normalization constants
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

# Load configuration
with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]

# Set compute device
if device_config == "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)


def load_image_for_vgg(path: str, device, size=None) -> torch.Tensor:
    """Load and preprocess an image for VGG19 evaluation.
    
    Loads image, converts to RGB, optionally resizes, and applies ImageNet normalization.
    
    Args:
        path: Path to the image file
        device: Torch device to move tensor to
        size: Optional resize dimensions (int or (H, W) tuple)
        
    Returns:
        Normalized tensor of shape [1, 3, H, W] on specified device
    """
    img = Image.open(path).convert("RGB")
    tfms = []
    if size is not None:
        tfms.append(transforms.Resize(size))
    tfms += [
        transforms.ToTensor(),  # Convert to tensor in range [0, 1]
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),  # ImageNet normalization
    ]
    x = transforms.Compose(tfms)(img).unsqueeze(0).to(device)
    return x


def calculate_layer_metrics(t1: torch.Tensor, t2: torch.Tensor) -> tuple:
    """Calculate raw, normalized, and log-scaled metrics between two feature tensors.
    
    Provides three complementary metrics for comparing features or Gram matrices:
    - Raw MSE: Direct feature space distance
    - Normalized Distance: MSE scaled to [0, 1] range based on dynamic range
    - Log Score: PSNR-like metric in [0, 100] where higher is better
    
    Args:
        t1: First tensor (output features)
        t2: Second tensor (target features)
        
    Returns:
        Tuple of (raw_mse, norm_dist, log_score) as floats
    """
    # Raw MSE - direct feature-space error
    raw_mse = mse(t1, t2).item()
    
    # Normalized distance - scale by dynamic range to make metric comparable across layers
    # Calculate maximum possible squared error based on actual tensor magnitudes
    max_val = max(t1.max().item(), t2.max().item())
    max_possible_sq_diff = (max_val ** 2) + 1e-8  # Add epsilon to avoid division by zero
    norm_dist = raw_mse / max_possible_sq_diff
    
    # Log score - similar to PSNR but inverted
    # Higher score = better match. Capped at 100 for near-perfect matches
    # Prevent log(0) by clipping normalized distance to minimum value
    safe_dist = max(norm_dist, 1e-10)
    log_score = -10 * math.log10(safe_dist)
    if log_score > 100:
        log_score = 100.0
        
    return raw_mse, norm_dist, log_score


def compute_metrics(out_t: torch.Tensor, content_t: torch.Tensor, style_t: torch.Tensor, vgg: VGGFeatures) -> dict:
    """Compute all content and style metrics for output image.
    
    Extracts VGG features and computes:
    - Content metrics: Feature similarity at relu4_2 layer
    - Style metrics: Gram matrix similarity averaged across multiple layers
    
    Args:
        out_t: Output image tensor [1, 3, H, W]
        content_t: Content image tensor [1, 3, H, W]
        style_t: Style image tensor [1, 3, H, W]
        vgg: VGGFeatures model (frozen, eval mode)
        
    Returns:
        Dictionary with keys: content_raw, content_norm, content_log,
                              style_raw, style_norm, style_log
    """
    with torch.no_grad():
        # Extract features from all three images
        f_out = vgg(out_t)
        f_c = vgg(content_t)
        f_s = vgg(style_t)

        # Content metrics - compare feature activations at content layer (relu4_2)
        c_raw, c_norm, c_log = calculate_layer_metrics(
            f_out[vgg.content_layer],
            f_c[vgg.content_layer]
        )

        # Style metrics - average Gram matrix metrics across all style layers
        s_raw_acc, s_norm_acc, s_log_acc = 0.0, 0.0, 0.0
        
        for l in vgg.style_layers:
            # Compute Gram matrices (capture texture/style)
            G_out = gram_matrix(f_out[l])
            G_style = gram_matrix(f_s[l])
            
            # Calculate metrics for this layer
            r, n, lg = calculate_layer_metrics(G_out, G_style)
            s_raw_acc += r
            s_norm_acc += n
            s_log_acc += lg
            
        # Average across layers
        num_styles = len(vgg.style_layers)
        s_raw = s_raw_acc / num_styles
        s_norm = s_norm_acc / num_styles
        s_log = s_log_acc / num_styles

    return {
        "content_raw": c_raw, "content_norm": c_norm, "content_log": c_log,
        "style_raw": s_raw, "style_norm": s_norm, "style_log": s_log
    }


def write_metrics_csv(csv_path: str, rows: list) -> None:
    """Write metric rows to CSV file, creating header if needed.
    
    Appends rows to CSV file. Creates header on first write.
    
    Args:
        csv_path: Path to CSV file (created if missing)
        rows: List of row tuples/lists to write
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    header = (
        "method", "output", "content", "style",
        "content_raw", "content_norm", "content_log",
        "style_raw", "style_norm", "style_log"
    )

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def eval_triplet_and_log(
    out_path: str,
    content_path: str,
    style_path: str,
    method_name: str,
    csv_path: str,
    device
) -> dict:
    """Load images, compute metrics, and log to CSV.
    
    Full evaluation pipeline: loads output, content, and style images,
    computes VGG-based metrics, and appends results to CSV file.
    
    Args:
        out_path: Path to output/stylized image
        content_path: Path to content image
        style_path: Path to style image
        method_name: Name of method (for CSV logging)
        csv_path: Path to CSV file for logging results
        device: Torch device for computation
        
    Returns:
        Dictionary of computed metrics
    """
    # Load images - use content dimensions for resizing others
    x_c = load_image_for_vgg(content_path, device)
    H, W = x_c.shape[-2:]

    x = load_image_for_vgg(out_path, device, size=(H, W))
    x_s = load_image_for_vgg(style_path, device, size=(H, W))

    # Extract filenames for CSV logging
    content_name = os.path.basename(content_path)
    style_name = os.path.basename(style_path)
    output_name = os.path.basename(out_path)

    # Compute metrics
    vgg = VGGFeatures().to(device).eval()
    m = compute_metrics(out_t=x, content_t=x_c, style_t=x_s, vgg=vgg)

    # Format row for CSV with appropriate precision
    # raw: 6 decimals (very small values), norm: 5 decimals, log: 1 decimal (dB scale)
    row = [
        method_name,
        output_name,
        content_name,
        style_name,
        f"{m['content_raw']:.6f}",
        f"{m['content_norm']:.5f}",
        f"{m['content_log']:.1f}",
        f"{m['style_raw']:.6f}",
        f"{m['style_norm']:.5f}",
        f"{m['style_log']:.1f}"
    ]

    write_metrics_csv(csv_path, [row])
    return m

def get_average_metrics(csv_path: str) -> tuple:
    """Read CSV and compute average metrics across all entries.
    
    Aggregates metrics from CSV file to get overall performance statistics.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Tuple of 6 floats: (avg_content_raw, avg_content_norm, avg_content_log,
                            avg_style_raw, avg_style_norm, avg_style_log)
                            
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Accumulate metrics
    total_content_raw = 0.0
    total_content_norm = 0.0
    total_content_log = 0.0
    total_style_raw = 0.0
    total_style_norm = 0.0
    total_style_log = 0.0
    count = 0

    # Read CSV and sum all metrics
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total_content_raw += float(row["content_raw"])
            total_content_norm += float(row["content_norm"])
            total_content_log += float(row["content_log"])
            total_style_raw += float(row["style_raw"])
            total_style_norm += float(row["style_norm"])
            total_style_log += float(row["style_log"])
            count += 1

    # Handle empty CSV
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute averages
    avg_content_raw = total_content_raw / count
    avg_content_norm = total_content_norm / count
    avg_content_log = total_content_log / count
    avg_style_raw = total_style_raw / count
    avg_style_norm = total_style_norm / count
    avg_style_log = total_style_log / count

    return avg_content_raw, avg_content_norm, avg_content_log, avg_style_raw, avg_style_norm, avg_style_log


if __name__ == "__main__":
    # Print aggregate metrics from evaluation CSV
    csv_path = os.path.join(output_dir, "metrics", "metrics.csv")
    print(get_average_metrics(csv_path))

    # Example usage (commented out):
    # eval_triplet_and_log(
    #     out_path="nst_1e6_step1200.png",
    #     content_path="/path/to/content.jpg",
    #     style_path="/path/to/style.jpg",
    #     method_name="nst_lbfgs",
    #     csv_path=csv_path,
    #     device=device,
    # )


# Log score interpretation guide:
# 0-10:   Terrible match (completely different)
# 10-20:  Poor match (visible similarity, but major errors)
# 20-40:  Good match (strong style/content similarity)
# 40+:    Excellent match (very hard to distinguish)
# 100:    Identity (exact same image)