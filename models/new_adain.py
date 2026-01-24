# adain_run.py
# Usage:
#   python adain_run.py --content path/to/content.jpg --style path/to/style.jpg --out out.png --alpha 0.8
# Optional:
#   --size 512  (max side; keeps aspect ratio)
#   --cpu       (force CPU)

import argparse, os, urllib.request
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DEC_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"

def ensure(path, url):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)

def load_img(path, max_side):
    img = Image.open(path).convert("RGB")
    if max_side and max_side > 0:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)

def calc_mean_std(feat, eps=1e-5):
    # feat: N,C,H,W
    n, c = feat.size()[:2]
    feat_var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std

def adain(content, style):
    c_mean, c_std = calc_mean_std(content)
    s_mean, s_std = calc_mean_std(style)
    normalized = (content - c_mean) / c_std
    return normalized * s_std + s_mean

def build_encoder():
    # Matches naoto0804 AdaIN vgg_normalised.pth (up to relu4_1)
    return nn.Sequential(
        nn.Conv2d(3, 3, 1, 1, 0),                     # <- important: 3x3 1x1
        nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),

        nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),

        nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, ceil_mode=True),

        nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3), nn.ReLU(inplace=True),
    )

def build_decoder():
    # Matches decoder.pth from the same release
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
    )
def match_rgb(style, content, eps=1e-5):
    # channel-wise mean/std match in RGB
    s_mean = style.mean(dim=(2,3), keepdim=True)
    s_std  = style.std(dim=(2,3), keepdim=True) + eps
    c_mean = content.mean(dim=(2,3), keepdim=True)
    c_std  = content.std(dim=(2,3), keepdim=True) + eps
    return (style - s_mean) / s_std * c_std + c_mean

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--alpha", type=float, default=1.0, help="style strength in [0,1]")
    ap.add_argument("--size", type=int, default=512, help="resize so max side <= size (0 disables)")
    ap.add_argument("--models", default="models", help="where to store/load weights")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--preserve_color", action="store_true")
    ap.add_argument("--adain_layer", type=int, default=4, choices=[3,4], help="3 = AdaIN at relu3_1 (often less crackly), 4 = relu4_1 (default)")
    ap.add_argument("--decoder_ckpt", default="", help="optional path to alternate decoder checkpoint (.pth or .pth.tar)")


    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    vgg_path = os.path.join(args.models, "vgg_normalised.pth")
    dec_path = os.path.join(args.models, "decoder.pth")
    ensure(vgg_path, VGG_URL)
    ensure(dec_path, DEC_URL)

    enc = build_encoder().to(device).eval()
    # full encoder is Sequential; relu3_1 ends at index 17 in our build
    enc_layers = list(enc.children())
    enc3 = nn.Sequential(*enc_layers[:18]).to(device).eval()   # up to relu3_1
    enc34 = nn.Sequential(*enc_layers[18:]).to(device).eval()  # relu3_1 -> relu4_1

    dec = build_decoder().to(device).eval()
    state = torch.load(vgg_path, map_location="cpu")
    enc.load_state_dict(state, strict=False)   # <-- ignore extra VGG layers in checkpoint

    if args.decoder_ckpt:
        dec.load_state_dict(load_ckpt(args.decoder_ckpt), strict=False)
    else:
        dec.load_state_dict(torch.load(dec_path, map_location="cpu"))


    c = load_img(args.content, args.size).to(device)
    s = load_img(args.style, args.size).to(device)

    if args.preserve_color:
        s = match_rgb(s, c).clamp(0, 1)

    if args.adain_layer == 3:
    # AdaIN at relu3_1 (256ch), then push through VGG to relu4_1 (512ch) for decoder
        c3 = enc3(c)
        s3 = enc3(s)
        t3 = adain(c3, s3)
        a = max(0.0, min(1.0, args.alpha))
        t3 = a * t3 + (1.0 - a) * c3
        t = enc34(t3)   # now 512ch at relu4_1
    else:
        # Default: AdaIN at relu4_1 (512ch)
        c4 = enc(c)
        s4 = enc(s)
        t = adain(c4, s4)
        a = max(0.0, min(1.0, args.alpha))
        t = a * t + (1.0 - a) * c4

    out = dec(t).clamp(0, 1)

    out = out.detach().cpu()
    if out.dim() == 4:
        out = out[0]  # remove batch dim -> C,H,W

    img = (out.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(img).save(args.out)

    print("Saved:", args.out)

if __name__ == "__main__":
    main()


"""
for a in 0.2 0.4 0.5 0.6 0.8 1.0; do
  python models/new_adain.py \
    --content data/content/processed/img_5.jpg \
    --style data/style/processed/style_7.jpg \
    --out out/adain_all/style7_img5_new_${a}.jpg \
    --alpha ${a}
done

"""