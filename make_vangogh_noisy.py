"""
Phase 2.1: Add random Gaussian noise to SD-gen Van Gogh images.
Used as a control experiment for Ex2 — random noise SHOULD NOT bypass IMMA.

Output: data/vangogh_noisy/ with same metadata.csv structure as data/vangogh/.
"""
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ============ Config ============
SRC_DIR = Path("data/vangogh")
DST_DIR = Path("data/vangogh_noisy")
EPSILON = 8 / 255  # L_inf perturbation budget, same as PGD baseline
SEED = 42

# ============ Setup ============
DST_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ============ Process images ============
src_pngs = sorted(SRC_DIR.glob("*.png"))
print(f"Found {len(src_pngs)} images in {SRC_DIR}")
print(f"Adding L_inf random noise with epsilon = {EPSILON:.4f} ({EPSILON*255:.1f}/255)")
print()

for src_path in src_pngs:
    # Load image as float in [0, 1]
    img = np.array(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0

    # Sample noise: uniform in [-epsilon, +epsilon]
    # Using uniform (not Gaussian) to exactly match L_inf ball, same convention as PGD init
    noise = rng.uniform(low=-EPSILON, high=EPSILON, size=img.shape).astype(np.float32)

    # Add and clip to [0,1]
    noisy = np.clip(img + noise, 0.0, 1.0)

    # Save back to uint8
    noisy_uint8 = (noisy * 255.0).round().astype(np.uint8)
    dst_path = DST_DIR / src_path.name
    Image.fromarray(noisy_uint8).save(dst_path)

print(f"Saved {len(src_pngs)} noisy images to {DST_DIR}")

# ============ Copy metadata.csv (same prompts) ============
src_meta = SRC_DIR / "metadata.csv"
dst_meta = DST_DIR / "metadata.csv"
if src_meta.exists():
    shutil.copy(src_meta, dst_meta)
    print(f"Copied metadata: {src_meta} -> {dst_meta}")
else:
    print(f"WARNING: {src_meta} not found, please create it manually.")

# ============ Sanity check: print stats ============
print("\n=== Sanity check ===")
sample_src = np.array(Image.open(src_pngs[0]).convert("RGB"), dtype=np.float32) / 255.0
sample_dst = np.array(Image.open(DST_DIR / src_pngs[0].name).convert("RGB"), dtype=np.float32) / 255.0
diff = sample_dst - sample_src
print(f"  First image: {src_pngs[0].name}")
print(f"  Max |perturbation| in pixel space: {np.abs(diff).max():.4f} (expected ≤ {EPSILON:.4f})")
print(f"  Mean |perturbation|:              {np.abs(diff).mean():.4f}")
print(f"  Note: Due to uint8 rounding, max may be slightly less than epsilon.")
