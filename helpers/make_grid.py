import os, glob, re, math
from PIL import Image
import matplotlib.pyplot as plt

def make_sweep_grid(
    img_dir: str,
    pair_prefix: str,      
    betas=(1e4, 1e5, 1e6, 1e7, 1e8),
    steps=(400, 800, 1200, 1600, 2000),            # <-- use your actual step numbers (e.g., 1..5) OR (400..2000)
    out_path="sweep_grid_1.png",
):
    # Map: (beta, step) -> filepath
    paths = {}

    # NOTE: your files have beta_ and step (like beta_10000.0_step1.png)
    pattern = os.path.join(img_dir, f"{pair_prefix}_beta_*_step*.png")
    for fp in glob.glob(pattern):
        m = re.search(r"_beta_([0-9eE+\-\.]+)_step(\d+)\.png$", os.path.basename(fp))
        if not m:
            continue

        beta = float(m.group(1))
        step = int(m.group(2))
        paths[(beta, step)] = fp

    nrows, ncols = len(betas), len(steps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4*ncols, 2.4*nrows), dpi=200)

    for r, beta in enumerate(betas):
        for c, step in enumerate(steps):
            ax = axes[r, c] if nrows > 1 else axes[c]
            ax.axis("off")

            # find matching file
            fp = None
            for (b, s), path in paths.items():
                if s == int(step) and abs(b - float(beta)) < 1e-6:
                    fp = path
                    break

            if fp and os.path.exists(fp):
                img = Image.open(fp).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")

            # titles/labels now swapped
            if r == 0:
                ax.set_title(f"{step} steps")
            if c == 0:
                ax.text(-0.02, 0.5, r"$\beta=10^{%d}$" % int(round(math.log10(beta))),
                        transform=ax.transAxes, rotation=90, va="center", ha="right")


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

# Example call (adjust steps to whatever your filenames contain):
make_sweep_grid(
    "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/out/grid_search",
    "style1_img9",
    betas=(1e4, 1e5, 1e6, 1e7, 1e8),
    steps=(400, 800, 1200, 1600, 2000),   # <-- if your files are step1, step2, ...
    out_path="grid_style1_img9.png"
)

make_sweep_grid(
    "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/out/grid_search",
    "style7_img5",
    betas=(1e4, 1e5, 1e6, 1e7, 1e8),
    steps=(400, 800, 1200, 1600, 2000),   # <-- if your files are step1, step2, ...
    out_path="grid_style7_img5.png"
)