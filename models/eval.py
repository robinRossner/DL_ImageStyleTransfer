import os, csv, time, argparse, math
import torch
import yaml
from PIL import Image
from torchvision import transforms

from nst_model import VGGFeatures, gram_matrix, mse

# --- Config & Setup ---
VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD  = (0.229, 0.224, 0.225)

with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]

if device_config == "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)


def load_image_for_vgg(path: str, device, size=None):
    """Loads and normalizes image for VGG."""
    img = Image.open(path).convert("RGB")
    tfms = []
    if size is not None:
        tfms.append(transforms.Resize(size))
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ]
    x = transforms.Compose(tfms)(img).unsqueeze(0).to(device)
    return x


def calculate_layer_metrics(t1, t2):
    """
    Computes Raw, Normalized, and Log-Scaled metrics for a pair of tensors.
    Returns: (raw_mse, norm_dist, log_score)
    """
    # 1. Raw MSE
    raw_mse = mse(t1, t2).item()
    
    # 2. Normalized Distance (0 to 1)
    # Calculate dynamic range max possible squared error
    max_val = max(t1.max().item(), t2.max().item())
    max_possible_sq_diff = (max_val ** 2) + 1e-8
    
    norm_dist = raw_mse / max_possible_sq_diff
    
    # 3. Log Score (0 to 100, higher is better)
    # Similar to PSNR. Cap at 100 for perfect matches.
    # We clip norm_dist to a tiny value to prevent log(0)
    safe_dist = max(norm_dist, 1e-10)
    log_score = -10 * math.log10(safe_dist)
    if log_score > 100: 
        log_score = 100.0
        
    return raw_mse, norm_dist, log_score


def compute_metrics(out_t, content_t, style_t, vgg: VGGFeatures):
    """
    Computes full suite of metrics for Content and Style.
    Returns dictionaries for clarity.
    """
    with torch.no_grad():
        f_out = vgg(out_t)
        f_c   = vgg(content_t)
        f_s   = vgg(style_t)

        # --- Content Metrics ---
        # Compare feature maps at content_layer
        c_raw, c_norm, c_log = calculate_layer_metrics(
            f_out[vgg.content_layer], 
            f_c[vgg.content_layer]
        )

        # --- Style Metrics ---
        # Average the metrics across all style layers
        s_raw_acc, s_norm_acc, s_log_acc = 0.0, 0.0, 0.0
        
        for l in vgg.style_layers:
            G_out = gram_matrix(f_out[l])
            G_style = gram_matrix(f_s[l])
            
            r, n, lg = calculate_layer_metrics(G_out, G_style)
            s_raw_acc  += r
            s_norm_acc += n
            s_log_acc  += lg
            
        num_styles = len(vgg.style_layers)
        s_raw  = s_raw_acc  / num_styles
        s_norm = s_norm_acc / num_styles
        s_log  = s_log_acc  / num_styles

    return {
        "content_raw": c_raw, "content_norm": c_norm, "content_log": c_log,
        "style_raw": s_raw,   "style_norm": s_norm,   "style_log": s_log
    }


def write_metrics_csv(csv_path: str, rows):
    """Writes rows to CSV, creating header if file doesn't exist."""
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
    device,
):
    """
    Loads images, computes all 6 metrics, and logs to CSV.
    """
    # 1. Load images
    x_c = load_image_for_vgg(content_path, device)
    H, W = x_c.shape[-2:]

    x   = load_image_for_vgg(out_path,   device, size=(H, W))
    x_s = load_image_for_vgg(style_path, device, size=(H, W))

    # 2. Prepare metadata
    content_name = os.path.basename(content_path)
    style_name   = os.path.basename(style_path)
    output_name  = os.path.basename(out_path)

    # 3. Compute Metrics
    vgg = VGGFeatures().to(device).eval()
    m = compute_metrics(out_t=x, content_t=x_c, style_t=x_s, vgg=vgg)

    # 4. Format row
    # precision: raw=6 (tiny numbers), norm=5, log=1 (it's decibels)
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

def get_average_metrics(csv_path: str):
    """
    Reads the CSV and computes average metrics for content and style.
    Returns: (avg_content_raw, avg_content_norm, avg_content_log, avg_style_raw, avg_style_norm, avg_style_log)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    total_content_raw = 0.0
    total_content_norm = 0.0
    total_content_log = 0.0
    total_style_raw = 0.0
    total_style_norm = 0.0
    total_style_log = 0.0
    count = 0

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

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    avg_content_raw = total_content_raw / count
    avg_content_norm = total_content_norm / count
    avg_content_log = total_content_log / count
    avg_style_raw = total_style_raw / count
    avg_style_norm = total_style_norm / count
    avg_style_log = total_style_log / count

    return avg_content_raw, avg_content_norm, avg_content_log, avg_style_raw, avg_style_norm, avg_style_log

if __name__ == "__main__":
    print(get_average_metrics("/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/out/metrics/metricsAdain.csv"))

    csv_path = os.path.join(output_dir, "metrics", "metrics.csv")
    """
    eval_triplet_and_log(
            out_path="nst_1e6_step1200.png",
            content_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            style_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/style/processed/style_6.jpg",
            method_name="match",
            csv_path=csv_path,
            device=device,
        )

    
    eval_triplet_and_log(
            out_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            content_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            style_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/style/processed/style_6.jpg",
            method_name="out=content",
            csv_path=csv_path,
            device=device,
        )
    
    eval_triplet_and_log(
            out_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            content_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            style_path="/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/data/content/processed/img_14.jpg",
            method_name="allsame",
            csv_path=csv_path,
            device=device,
        )

    eval_triplet_and_log(
        out_path="EXAMPLE_OUTPUT_PATH.jpg",
        content_path="EXAMPLE_CONTENT_PATH.jpg",
        style_path="EXAMPLE_STYLE_PATH.jpg",
        method_name="precomputed",
        csv_path=csv_path,
        device=device,
    )"""


"""
How to interpret the new "Log" columns:
0 - 10: Terrible match (Completely different)

10 - 20: Poor match (Visible similarity, but major errors)

20 - 40: Good match (Strong style/content similarity)

40+: Excellent match (Very hard to distinguish)

100: Identity (Exact same image)
"""