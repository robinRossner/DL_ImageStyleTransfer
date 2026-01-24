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

# Must match loader.py normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]

if device_config == "None":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)
print("Using device:", device)


def denorm_and_save(x: torch.Tensor, path: str) -> None:
    """x is VGG-normalized (1,3,H,W). Save as [0,1]."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    x = x.detach().cpu().squeeze(0)
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0.0, 1.0)
    save_image(x, path)


class VGGFeatures(nn.Module):
    """
    Torchvision VGG19 layer indices:
      relu1_1 = 1
      relu2_1 = 6
      relu3_1 = 11
      relu4_1 = 20
      relu4_2 = 22  (content)
      relu5_1 = 29
    """
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self.style_layers = [1, 6, 11, 20, 29]
        self.content_layer = 22

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        feats: Dict[int, torch.Tensor] = {}
        needed = set(self.style_layers + [self.content_layer])

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in needed:
                feats[i] = x
            if i >= max(needed):
                break
        return feats


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Unnormalized Gram matrix (B,C,H,W) -> (B,C,C)."""
    B, C, H, W = feat.shape
    F = feat.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2))
    return G  # unnormalized for paper-style normalization later


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def clamp_normalized_(x: torch.Tensor) -> None:
    """
    Clamp in pixel space [0,1], then renormalize back to VGG space.
    Keeps optimization stable.
    """
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    with torch.no_grad():
        px = x * std + mean
        px.clamp_(0.0, 1.0)
        x.copy_((px - mean) / std)


def neural_style_transfer_lbfgs(
    content_path: str,
    style_path: str,
    out_path: str = "nst_out.png",
    steps: int = 1200,          # LBFGS steps (each calls closure once here)
    alpha: float = 1.0,
    beta: float = 1e4,
    save_every: int = 400,
    style_layer_weights: Optional[Dict[int, float]] = None,
    content: Optional[torch.Tensor] = None,
    style: Optional[torch.Tensor] = None
) -> None:

    # Load images with YOUR loader (no resizing here)
    if content is None:
        content = process_image(content_path, device=device)
    if style is None:
        style = process_image(style_path, device=device)

    vgg = VGGFeatures().to(device)

    # Set default equal weights for style layers
    if style_layer_weights is None:
        style_layer_weights = {l: 1.0 for l in vgg.style_layers}

    # Precompute targets
    with torch.no_grad():
        c_feats = vgg(content)
        s_feats = vgg(style)

        content_target = c_feats[vgg.content_layer]
        style_targets = {l: gram_matrix(s_feats[l]) for l in vgg.style_layers}

    # content image init
    target = content.clone().requires_grad_(True)

    optimizer = optim.LBFGS([target], max_iter=1, history_size=50, line_search_fn="strong_wolfe")

    # For logging
    last_c = 0.0
    last_s = 0.0

    for step in range(1, steps + 1):

        def closure():
            nonlocal last_c, last_s
            optimizer.zero_grad(set_to_none=True)

            feats = vgg(target)

            # Content loss (relu4_2)
            c_loss = mse(feats[vgg.content_layer], content_target)

            # Style loss (weighted sum over layers with paper normalization)
            s_loss = torch.tensor(0.0, device=device)
            for l in vgg.style_layers:
                w_l = style_layer_weights.get(l, 1.0)

                B, C, H, W = feats[l].shape
                N = C
                M = H * W

                G = gram_matrix(feats[l])
                A = style_targets[l]  # precomputed grams for style

                layer_loss = torch.mean((G - A) ** 2)
                layer_loss = layer_loss / (4.0 * (N ** 2) * (M ** 2))

                s_loss = s_loss + w_l * layer_loss

            total = alpha * c_loss + beta * s_loss
            total.backward()

            # store for logging
            last_c = float(c_loss.detach())
            last_s = float(s_loss.detach())

            return total

        loss = optimizer.step(closure)
        clamp_normalized_(target)


        if step % 50 == 0 or step == 1:
            if step == 1:
                print("Starting NST optimization...")
                start_time = time.time()
            elif step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{steps} done. Elapsed time: {elapsed:.2f} seconds.")
            print(f"Step {step}/{steps} done.")
        #if step == 1 or step % save_every == 0 or step == steps:
        if step == steps:
            step_path = out_path.replace(".png", f"_step{step}.png")
            denorm_and_save(target, step_path)
            print(
                f"[{step}/{steps}] "
                f"total={float(loss):.4f} content={last_c:.6f} style={last_s:.6f} "
                f"saved={step_path}"
            )
def calc_grid_betaxstep(content, style, steps, name):
    beta_vals = [1e4, 1e5, 1e6, 1e7, 1e8]
    for beta in beta_vals:
        neural_style_transfer_lbfgs(
            content_path=content,
            style_path=style,
            out_path= output_dir + f"/grid_search/{name}_beta_{beta}.png",
            steps=steps,
            alpha=1.0,
            beta=beta,
            save_every= 400
        )

def folder_nst():
    content_folder = data_dir + "/content/processed/"
    style_folder = data_dir + "/style/processed/"

    content_images = [f for f in os.listdir(content_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    step = 1200
    beta = 1e7
    for i, content_img in enumerate(content_images):
        print("Processing content image:", content_img)
        print("Progress:" + str(i+1) + "/" + str(len(content_images)))
        content_path = os.path.join(content_folder, content_img)
        content = process_image(content_path, device=device)

        for j, style_img in enumerate(style_images):
            print(f"  Progress:" + str(j+1) + "/" + str(len(style_images)))
            style_path = os.path.join(style_folder, style_img)
            name = f"style{style_img.split('_')[1].split('.')[0]}_img{content_img.split('_')[1].split('.')[0]}"
            out_path = output_dir + f"/nst_all/{name}_beta_{beta}.png"

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
            out_path_step = out_path.replace(".png", f"_step{step}.png")
            eval_out_path = out_path_step if os.path.exists(out_path_step) else out_path

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
    folder_nst()
