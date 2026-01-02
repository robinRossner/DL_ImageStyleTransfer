import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

from loader import process_image  # <-- your loader

# Must match loader.py normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Normalized Gram matrix (B,C,H,W) -> (B,C,C)."""
    B, C, H, W = feat.shape
    F = feat.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2))
    return G / (C * H * W)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw


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
    steps: int = 300,          # LBFGS steps (each calls closure once here)
    alpha: float = 1.0,
    beta: float = 1e4,
    tv_weight: float = 1e-6,   # set 0.0 for “pure” Gatys
    save_every: int = 50,
) -> None:

    # Load images with YOUR loader (no resizing here)
    content = process_image(content_path, device=device)
    style = process_image(style_path, device=device)

    vgg = VGGFeatures().to(device)

    # Precompute targets
    with torch.no_grad():
        c_feats = vgg(content)
        s_feats = vgg(style)

        content_target = c_feats[vgg.content_layer]
        style_targets = {l: gram_matrix(s_feats[l]) for l in vgg.style_layers}

    # Gatys-style init: white noise image
    target = torch.randn_like(content).to(device).requires_grad_(True)

    optimizer = optim.LBFGS([target], max_iter=1, history_size=50, line_search_fn="strong_wolfe")

    # For logging
    last_c = 0.0
    last_s = 0.0
    last_tv = 0.0

    for step in range(1, steps + 1):

        def closure():
            nonlocal last_c, last_s, last_tv
            optimizer.zero_grad(set_to_none=True)

            feats = vgg(target)

            # Content loss (relu4_2)
            c_loss = mse(feats[vgg.content_layer], content_target)

            # Style loss (sum over layers of Gram MSE)
            s_loss = torch.tensor(0.0, device=device)
            for l in vgg.style_layers:
                s_loss = s_loss + mse(gram_matrix(feats[l]), style_targets[l])

            # TV regularization (optional)
            t_loss = tv_loss(target)

            total = alpha * c_loss + beta * s_loss + tv_weight * t_loss
            total.backward()

            # store for logging
            last_c = float(c_loss.detach())
            last_s = float(s_loss.detach())
            last_tv = float(t_loss.detach())

            return total

        loss = optimizer.step(closure)
        clamp_normalized_(target)

        if step == 1 or step % save_every == 0 or step == steps:
            step_path = out_path.replace(".png", f"_step{step}.png")
            denorm_and_save(target, step_path)
            print(
                f"[{step}/{steps}] "
                f"total={float(loss):.4f} content={last_c:.6f} style={last_s:.6f} tv={last_tv:.6f} "
                f"saved={step_path}"
            )

    denorm_and_save(target, out_path)
    print("Final saved:", out_path)


if __name__ == "__main__":
    content = "test_content_dog_512.png"
    style = "test_style_spiral_512.jpg"

    neural_style_transfer_lbfgs(
        content_path=content,
        style_path=style,
        out_path="nst_gatys_lbfgs.png",
        steps=300,        # try 300-1000 depending on CPU/GPU
        alpha=1.0,
        beta=1e4,
        tv_weight=1e-6,   # set 0.0 if you want no TV
        save_every=50,
    )

    # Beta comparison (should differ now)
    neural_style_transfer_lbfgs(
        content_path=content,
        style_path=style,
        out_path="nst_beta_low.png",
        steps=300,
        alpha=1.0,
        beta=1e2,
        tv_weight=1e-6,
        save_every=150,
    )

    neural_style_transfer_lbfgs(
        content_path=content,
        style_path=style,
        out_path="nst_beta_high.png",
        steps=300,
        alpha=1.0,
        beta=1e5,
        tv_weight=1e-6,
        save_every=150,
    )