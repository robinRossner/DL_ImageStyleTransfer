from loader import process_image
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from typing import List, Dict, Union
import torch.optim as optim
import os

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")  # RTX 3060 Ti
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -------------------------
# VGG feature extractor
# -------------------------
class VGGFeatures(nn.Module):
    """
    Extracts intermediate VGG19 features by layer index (torchvision ordering).
    Common NST choice:
      Style: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
      Content: relu4_2
    """
    def __init__(self, layer_ids: List[int]):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self.layer_ids = sorted(set(layer_ids))

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                feats[i] = x
            if i >= self.layer_ids[-1]:
                break
        return feats


# -------------------------
# Loss helpers
# -------------------------
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (B,C,H,W). Returns normalized Gram: (B,C,C)
    Normalization is critical; without it beta often "doesn't matter".
    """
    B, C, H, W = feat.shape
    F = feat.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2))
    return G / (C * H * W)


def content_loss(target_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
    return torch.mean((target_feat - content_feat) ** 2)


def style_loss_gram(target_feat: torch.Tensor, style_gram: torch.Tensor) -> torch.Tensor:
    return torch.mean((gram_matrix(target_feat) - style_gram) ** 2)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Standard TV loss to reduce high-frequency noise.
    """
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw


# -------------------------
# NST (Gatys-style optimization)
# -------------------------
def neural_style_transfer(
    content_path: str,
    style_paths: Union[str, List[str]],
    out_path: str = "nst_out.png",
    max_size: int = 512,

    # Gatys-like layers (torchvision vgg19 indices):
    # relu1_1=1, relu2_1=6, relu3_1=11, relu4_1=20, relu5_1=29
    # relu4_2=22
    style_layers: List[int] = [1, 6, 11, 20, 29],
    content_layer: int = 22,

    # loss weights
    alpha: float = 1.0,        # content
    beta: float = 1e4,         # style
    tv_weight: float = 1e-6,   # TV regularization

    # per-layer style weights (Gatys often uses equal; you can tweak)
    style_layer_weights: Dict[int, float] = None,

    # multi-style blending weights (sums to 1 recommended)
    style_blend_weights: List[float] = None,

    # optimization
    num_steps: int = 1500,
    optimizer_name: str = "lbfgs",   # "lbfgs" (Gatys) or "adam"
    lr: float = 0.02,               # used for Adam; LBFGS ignores lr mostly
    init: str = "content",          # "content", "noise", or "mixed"
    mixed_ratio: float = 0.5,       # if init=="mixed": target = r*content + (1-r)*noise
    clamp_each_step: bool = True,   # keep pixels reasonable after updates

    save_every: int = 250,
) -> None:
    """
    Optimization-based NST very close to Gatys et al.
    Supports multiple styles (blend grams) similar to multi-style references.
    """

    if isinstance(style_paths, str):
        style_paths = [style_paths]

    # default equal layer weights
    if style_layer_weights is None:
        style_layer_weights = {l: 1.0 for l in style_layers}

    # default equal style blending
    if style_blend_weights is None:
        style_blend_weights = [1.0 / len(style_paths)] * len(style_paths)
    else:
        assert len(style_blend_weights) == len(style_paths), "style_blend_weights length must match style_paths"

    # load images
    content = process_image(content_path, device=device, max_size=max_size)
    styles = [process_image(p, device=device, max_size=max_size, force_square=False) for p in style_paths]

    # build feature extractor
    needed_layers = style_layers + [content_layer]
    vgg = VGGFeatures(layer_ids=needed_layers).to(device)

    # precompute content target feature
    with torch.no_grad():
        content_feats = vgg(content)
        content_target = content_feats[content_layer]

    # precompute blended style Gram targets
    style_targets: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        # initialize grams to 0
        for l in style_layers:
            style_targets[l] = torch.zeros(1, content.shape[1], content.shape[1], device=device)

        for w_style, style_img in zip(style_blend_weights, styles):
            feats = vgg(style_img)
            for l in style_layers:
                style_targets[l] += w_style * gram_matrix(feats[l])

    # initialize target
    if init == "content":
        target = content.clone()
    elif init == "noise":
        target = torch.randn_like(content)
    elif init == "mixed":
        noise = torch.randn_like(content)
        target = mixed_ratio * content + (1.0 - mixed_ratio) * noise
    else:
        raise ValueError("init must be one of: content, noise, mixed")

    target = target.requires_grad_(True)

    # optimizer (Gatys uses L-BFGS)
    if optimizer_name.lower() == "lbfgs":
        optimizer = optim.LBFGS([target], max_iter=1, history_size=50, line_search_fn="strong_wolfe")
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam([target], lr=lr)
    else:
        raise ValueError("optimizer_name must be 'lbfgs' or 'adam'")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # helper: clamp target in normalized space by clamping in pixel space via denorm/renorm
    mean = torch.tensor(VGG_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(VGG_STD, device=device).view(1, 3, 1, 1)

    def clamp_target_01():
        with torch.no_grad():
            x = target * std + mean
            x.clamp_(0.0, 1.0)
            target.copy_((x - mean) / std)

    # optimization loop
    for step in range(1, num_steps + 1):

        def closure():
            optimizer.zero_grad(set_to_none=True)

            feats = vgg(target)

            # content loss (Johnson perceptual content loss = MSE in feature space)
            c_loss = content_loss(feats[content_layer], content_target)

            # style loss (Gatys: MSE between Gram matrices across multiple layers)
            s_loss = torch.tensor(0.0, device=device)
            for l in style_layers:
                w_l = style_layer_weights.get(l, 1.0)
                s_loss = s_loss + w_l * style_loss_gram(feats[l], style_targets[l])

            # TV loss for smoothness (practical improvement)
            tv = total_variation_loss(target)

            total = alpha * c_loss + beta * s_loss + tv_weight * tv
            total.backward()
            return total

        if optimizer_name.lower() == "lbfgs":
            loss_val = optimizer.step(closure)
        else:
            loss_val = closure()
            optimizer.step()

        if clamp_each_step:
            clamp_target_01()

        if step % save_every == 0 or step == 1 or step == num_steps:
            tmp_path = out_path.replace(".png", f"_step{step}.png")
            save_tensor_image(target, tmp_path)
            print(f"[{step}/{num_steps}] saved: {tmp_path} | loss: {float(loss_val):.4f}")

    # final save
    save_tensor_image(target, out_path)
    print("Final saved:", out_path)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    content = "test_content_dog_128.png"
    style = "test_style_spiral_128.jpg"

    # Gatys-like baseline (LBFGS), beta changes will now actually matter
    neural_style_transfer(
        content_path=content,
        style_paths=style,
        out_path="nst_gatys_lbfgs.png",
        max_size=256,          # try 256 or 512 for better separation
        alpha=1.0,
        beta=1e4,
        tv_weight=1e-6,
        num_steps=1500,
        optimizer_name="lbfgs",
        init="content",
        save_every=300,
    )

    # Show beta effect quickly (same settings, different beta)
    neural_style_transfer(
        content_path=content,
        style_paths=style,
        out_path="nst_beta_low.png",
        max_size=256,
        alpha=1.0,
        beta=1e2,
        tv_weight=1e-6,
        num_steps=800,
        optimizer_name="lbfgs",
        init="content",
        save_every=400,
    )

    neural_style_transfer(
        content_path=content,
        style_paths=style,
        out_path="nst_beta_high.png",
        max_size=256,
        alpha=1.0,
        beta=1e5,
        tv_weight=1e-6,
        num_steps=800,
        optimizer_name="lbfgs",
        init="content",
        save_every=400,
    )