import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, utils
import os
import yaml


with open("../config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
device_config = config["training"]["device"]

# 1. FIXED LOADER: Raw [0, 1] scaling only 
def load_adain_image(path, device, size=512):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(), # Standard [0, 1] scaling
    ])
    return t(img).unsqueeze(0).to(device)

# 2. ARCHITECTURE: Huang & Belongie (2017) specification 
class VGGEncoder(nn.Module):
    def __init__(self, weights_path, device):
        super().__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
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
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()   # relu4_1 bottleneck
        )
        # Load pre-trained weights without prefix errors
        self.vgg.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.vgg(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3)
        )

    def forward(self, x):
        return self.decoder(x)

# 3. ADAIN LOGIC: Feature statistics transfer [cite: 14, 31]
def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

# 4. INFERENCE FUNCTION
def run_stylization(content_path, style_path, v_path, d_path, alpha, device):
    content = load_adain_image(content_path, device)
    style = load_adain_image(style_path, device)

    encoder = VGGEncoder(v_path, device).to(device).eval()
    decoder = Decoder().to(device)
    # Target internal sequential module for weight loading
    decoder.decoder.load_state_dict(torch.load(d_path, map_location=device))
    decoder.eval()

    with torch.no_grad():
        c_feat = encoder(content)
        s_feat = encoder(style)
        
        # AdaIN interpolation for strength ablation 
        t = adain(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat
        
        output = decoder(t)
    return output.clamp(0, 1)

def folder_adain():
    content_folder = data_dir + "/content/processed/"
    style_folder = data_dir + "/style/processed/"

    content_images = [f for f in os.listdir(content_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    step = 1200
    alpha = 1.0
    for i, content_img in enumerate(content_images):
        print("Processing content image:", content_img)
        print("Progress:" + str(i+1) + "/" + str(len(content_images)))
        for j, style_img in enumerate(style_images):
            print(f"  Progress:" + str(j+1) + "/" + str(len(style_images)))
            content_path = os.path.join(content_folder, content_img)
            style_path = os.path.join(style_folder, style_img)
            name = f"style{style_img.split('_')[1].split('.')[0]}_img{content_img.split('_')[1].split('.')[0]}"
            out_path = output_dir + f"/adain_all/{name}_alpha_{alpha}.png"

            v_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/vgg_normalised.pth"
            d_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/decoder.pth"
            result = run_stylization(content_path, style_path, v_weights, d_weights, alpha, device)
            utils.save_image(result, out_path)

            from eval import eval_triplet_and_log
            csv_path = os.path.join(output_dir, "metrics", "metrics.csv")
            eval_triplet_and_log(
                out_path=out_path,
                content_path=content_path,
                style_path=style_path,
                method_name=f"adain_alpha_{alpha}",
                csv_path=csv_path,
                device=device,
            )

# 5. STRENGTH ABLATION EXECUTION 
if __name__ == "__main__":
    device = "mps" 
    style_folder = data_dir + "/style/processed/"
    style_images = [f for f in os.listdir(style_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    content_img = data_dir + "/content/processed/img_9.jpg"
    for style_img in style_images:
        # apply style to content with 0.8 alpha
        style_path = os.path.join(style_folder, style_img)
        name = f"style{style_img.split('_')[1].split('.')[0]}_img9"
        out_path = output_dir + f"adain_ablation/{name}_alpha_0.8.png"
        v_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/vgg_normalised.pth"
        d_weights = "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/models/decoder.pth"
        result = run_stylization(content_img, style_path, v_weights, d_weights, 0.8, device)
        utils.save_image(result, out_path)
