import os, csv, time, argparse
import torch
import yaml
from PIL import Image
from torchvision import transforms

from loader import process_image
from nst_model import VGGFeatures, gram_matrix, mse
# from adain.adain_model import TO BE IMPLEMENTED

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
    img = Image.open(path).convert("RGB")

    tfms = []
    if size is not None:
        # size can be int or (H,W) depending on how you do it elsewhere
        tfms.append(transforms.Resize(size))
    tfms += [
        transforms.ToTensor(),  # -> [3,H,W] in 0..1
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ]

    x = transforms.Compose(tfms)(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    )


def compute_metrics(out_t, content_t, style_t, vgg: VGGFeatures):
    """
    All tensors are expected to be in the SAME normalized space as VGG expects
    (i.e., whatever process_image produces).
    """
    with torch.no_grad():
        f_out = vgg(out_t)
        f_c   = vgg(content_t)
        f_s   = vgg(style_t)

        # Content proxy: VGG feature distance at relu4_2 (your content_layer = 22)
        content_dist = mse(f_out[vgg.content_layer], f_c[vgg.content_layer]).item()

        # Style proxy: Gram distance across your style layers
        style_dist = 0.0
        for l in vgg.style_layers:
            G = gram_matrix(f_out[l])
            A = gram_matrix(f_s[l])
            style_dist += mse(G, A).item()

    return content_dist, style_dist


def generate_output(method: str, content_path: str, style_path: str, **kwargs) -> torch.Tensor:
    """
    Returns a stylized tensor (normalized VGG space).
    Implement per method by importing the right function.
    """

    if method.lower() == "nst":
        # NST: drive from file paths, returns tensor
        from nst_model import run_nst_return_tensor
        return run_nst_return_tensor(
            content_path=content_path,
            style_path=style_path,
            steps=kwargs.get("steps", 600),
            alpha=kwargs.get("alpha", 1.0),
            beta=kwargs.get("beta", 1e4),
        )

    elif method.lower() == "adain":
        # AdaIN: you implement this to return a normalized tensor in same space
        # Expected signature: adain_stylize(content_path, style_path, alpha=..., ...)
        from adain_model import adain_stylize  # <-- change to your actual module/function
        return adain_stylize(
            content_path=content_path,
            style_path=style_path,
            alpha=kwargs.get("alpha", 1.0),
        )

    else:
        raise ValueError(f"Unknown method: {method}")
    
def write_metrics_csv(
    csv_path: str,
    rows,
    header=("method", "content", "style", "content_vgg_dist", "style_gram_dist")
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    write_header = not os.path.exists(csv_path)

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
    Evaluate a single (output, content, style) triplet and append metrics to CSV.
    All images are loaded from disk and converted to VGG space.
    """

    # ---- load content first (sets spatial size) ----
    x_c = load_image_for_vgg(content_path, device)
    H, W = x_c.shape[-2:]

    # ---- load output + style, resized to match ----
    x   = load_image_for_vgg(out_path,   device, size=(H, W))
    x_s = load_image_for_vgg(style_path, device, size=(H, W))

    # ---- names for CSV (use filenames) ----
    content_name = os.path.basename(content_path)
    style_name   = os.path.basename(style_path)
    output_name  = os.path.basename(out_path)

    # ---- VGG + metrics ----
    vgg = VGGFeatures().to(device).eval()

    content_dist, style_dist = compute_metrics(
        out_t=x,
        content_t=x_c,
        style_t=x_s,
        vgg=vgg
    )

    # ---- write to CSV ----
    rows = [[
        method_name,
        output_name,
        content_name,
        style_name,
        content_dist,
        style_dist
    ]]

    write_metrics_csv(
        csv_path,
        rows,
        header=("method", "output", "content", "style",
                "content_vgg_dist", "style_gram_dist")
    )

    return content_dist, style_dist


if __name__ == "__main__":
    csv_path = os.path.join(output_dir, "metrics", "metrics.csv")

    eval_triplet_and_log(
        out_path="EXAMPLE_OUTPUT_PATH.jpg",
        content_path="EXAMPLE_CONTENT_PATH.jpg",
        style_path="EXAMPLE_STYLE_PATH.jpg",
        method_name="precomputed",
        csv_path=csv_path,
        device=device,
    )