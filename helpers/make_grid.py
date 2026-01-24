import os, re, glob, math
import matplotlib.pyplot as plt
from PIL import Image


def make_sweep_grid(
    img_dir: str,
    pair_prefix: str,
    param: str = "nst",   # "nst" or "adain"
    betas=(1e4, 1e5, 1e6, 1e7, 1e8),
    steps=(400, 800, 1200, 1600, 2000),
    alphas=(0.1, 0.2, 0.3, 0.4, 0.5),
    out_path="sweep_grid_1.png",
):
    """
    param="nst":
      expects files like: {pair_prefix}_beta_10000.0_step1.png  (or .jpg/.jpeg)
      grid is (len(betas) rows) x (len(steps) cols)

    param="adain":
      expects files like: {pair_prefix}_0.4.jpg  (or .png/.jpeg) where 0.4 is alpha
      grid is 1 row x (len(alphas) cols), no "step"
    """

    param = param.lower().strip()
    if param not in ("nst", "adain"):
        raise ValueError("param must be 'nst' or 'adain'")

    paths = {}

    if param == "nst":
        # Map: (beta, step) -> filepath
        pattern = os.path.join(img_dir, f"{pair_prefix}_beta_*_step*.*")
        for fp in glob.glob(pattern):
            base = os.path.basename(fp)
            m = re.search(r"_beta_([0-9eE+\-\.]+)_step(\d+)\.(png|jpg|jpeg)$", base, re.IGNORECASE)
            if not m:
                continue
            beta = float(m.group(1))
            step = int(m.group(2))
            paths[(beta, step)] = fp

        nrows, ncols = len(betas), len(steps)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), dpi=200)
        if nrows == 1:
            axes = axes[None, :]  # make 2D for consistent indexing

        for r, beta in enumerate(betas):
            for c, step in enumerate(steps):
                ax = axes[r, c]
                ax.axis("off")

                # find matching file (tolerant float compare)
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

                # labels
                if r == 0:
                    ax.set_title(f"{step} steps")
                if c == 0:
                    ax.text(
                        -0.02, 0.5,
                        r"$\beta=10^{%d}$" % int(round(math.log10(beta))),
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right"
                    )

    else:  # param == "adain"
        print("\n[DEBUG] ===== AdaIN sweep grid =====")
        print("[DEBUG] img_dir:", img_dir)
        print("[DEBUG] pair_prefix:", pair_prefix)

        # Map: alpha -> filepath
        paths = {}

        pattern = os.path.join(img_dir, f"{pair_prefix}*.*")
        all_matches = glob.glob(pattern)
        print("[DEBUG] glob pattern:", pattern)
        print("[DEBUG] glob matches:", len(all_matches))
        print("[DEBUG] first 10 matches:", [os.path.basename(x) for x in all_matches[:10]])

        for fp in all_matches:
            base = os.path.basename(fp)
            stem, ext = os.path.splitext(base)

            # only accept image extensions
            if ext.lower() not in (".png", ".jpg", ".jpeg"):
                print(f"[DEBUG] skip (ext)   : {base}")
                continue

            # must start with the prefix (case sensitive)
            if not stem.startswith(pair_prefix):
                print(f"[DEBUG] skip (prefix): {base}  (stem='{stem}')")
                continue

            parts = stem.split("_")
            if len(parts) < 2:
                print(f"[DEBUG] skip (parts) : {base}  (parts={parts})")
                continue

            alpha_str = parts[-1]
            try:
                alpha = float(alpha_str)
            except ValueError:
                print(f"[DEBUG] skip (float) : {base}  (alpha_str='{alpha_str}')")
                continue

            key = round(alpha, 6)
            paths[key] = fp
            print(f"[DEBUG] OK            : {base}  -> alpha={alpha} (key={key})")

        print("[DEBUG] Found alphas:", sorted(paths.keys()))

        if alphas is None:
            alphas = tuple(sorted(paths.keys()))
            print("[DEBUG] Using detected alphas:", alphas)
        else:
            alphas = tuple(round(float(a), 6) for a in alphas)
            print("[DEBUG] Requested alphas:", alphas)

        nrows, ncols = 1, len(alphas)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), dpi=200)
        if ncols == 1:
            axes = [axes]

        for c, alpha in enumerate(alphas):
            ax = axes[c]
            ax.axis("off")

            fp = paths.get(round(float(alpha), 6))
            print(f"[DEBUG] lookup alpha={alpha} ->", os.path.basename(fp) if fp else None)

            if fp and os.path.exists(fp):
                img = Image.open(fp).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")

            ax.set_title(r"$\alpha=%s$" % (str(alpha)))


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


"""make_sweep_grid(
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
)"""

make_sweep_grid(
    "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/out/adain_all",
    "style3_img9",
    alphas=(0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    out_path="adain_grid_style3_img9.png",
    param="adain"
)

make_sweep_grid(
    "/Users/robin/Desktop/Uni/2025W/Deep Learning/DL_ImageStyleTransfer/out/adain_all",
    "style7_img26",
    alphas=(0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    out_path="adain_grid_style7_img26.png",
    param="adain"
)