"""
compute_sgr.py - Compute IMMA's Similarity Gap Ratio (SGR) over the 50-epoch LoRA adaptation.

SGR (paper Eq. 7), per epoch:
    SGR_e = ( M_e(ref, w/o IMMA) - M_e(ref, w/ IMMA) ) / M_e(ref, w/o IMMA)
where M is an image-similarity metric (CLIP / DINO / 1-LPIPS).

The paper states the style gap "remains steady throughout the epochs" and Table 1
reports a single value per artist. Empirically (and consistent with a steady curve),
that value matches the MEAN of the per-epoch SGR over the 50 adaptation epochs, NOT
the final epoch (which can collapse to ~0 once both arms over-fit) nor the early peak.
We therefore report the 50-epoch MEAN as the headline SGR, and also print the early
peak and the final epoch for transparency.

CLIP / DINO per-epoch similarities are read from the CSVs written by the author's
eval.py. LPIPS is recomputed here with the SAME preprocessing as eval.py
(resize 64x64, range [0,1], average over all reference x generated pairs) but
vectorized on GPU; the value is mathematically identical to eval.py (whose LPIPS
path runs one pair at a time on CPU and is impractically slow).

Usage:
    python compute_sgr.py --run_dir runs_repro/tyleredlin --reference_dir data/tyleredlin
"""
import os, glob, csv, argparse
import pandas as pd, torch
from PIL import Image
from torchvision import transforms

IMSIZE = 64  # eval.py imsize
_loader = transforms.Compose([transforms.Resize((IMSIZE, IMSIZE), antialias=True),
                              transforms.ToTensor()])


def _load(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _loader(img)  # [0,1] CHW, matches eval.py image_loader(lpips=False)


def _epochs(gen_dir):
    files = glob.glob(os.path.join(gen_dir, "*.png"))
    return sorted({os.path.basename(f).split("_")[0] for f in files
                   if "_" in os.path.basename(f) and os.path.basename(f).split("_")[0].isdigit()}, key=int)


def compute_lpips_csv(ref_dir, base_dir, imma_dir, out_csv, device, model):
    """Per-epoch sim = 1 - mean LPIPS over all (reference x generated) pairs. Same as eval.py, vectorized."""
    refs = torch.stack([_load(f) for f in sorted(glob.glob(os.path.join(ref_dir, "*.png")))]).to(device)
    R = refs.shape[0]

    def per_epoch(gen_dir):
        out = {}
        with torch.no_grad():
            for ep in _epochs(gen_dir):
                gens = sorted(glob.glob(os.path.join(gen_dir, f"{ep}_*.png")))
                if not gens:
                    continue
                tot, cnt = 0.0, 0
                for g in gens:
                    gt = _load(g).unsqueeze(0).to(device).repeat(R, 1, 1, 1)
                    tot += model(gt, refs).sum().item()
                    cnt += R
                out[ep] = 1.0 - tot / cnt
        return out

    wo, wi = per_epoch(base_dir), per_epoch(imma_dir)
    eps = sorted(set(wo) & set(wi), key=int)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "w/o IMMA", "w/ IMMA"])
        for e in eps:
            w.writerow([e, wo[e], wi[e]])


def sgr_stats(csv_path):
    """Return dict with mean / peak(+epoch) / final(+epoch) of the per-epoch SGR (%)."""
    df = pd.read_csv(csv_path)
    ep_col = df.columns[0]
    df["_e"] = df[ep_col].astype(int)
    df = df.sort_values("_e").reset_index(drop=True)
    sgr = (df["w/o IMMA"] - df["w/ IMMA"]) / df["w/o IMMA"] * 100.0
    i_peak = int(sgr.values.argmax())
    return {
        "mean": float(sgr.mean()),
        "peak": float(sgr.max()), "peak_epoch": int(df["_e"][i_peak]),
        "final": float(sgr.iloc[-1]), "final_epoch": int(df["_e"].iloc[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="dir with relearn/{without_imma,with_imma} and eval/{clip,dino}.csv")
    ap.add_argument("--reference_dir", required=True, help="dir of reference (clean) images")
    args = ap.parse_args()

    evald = os.path.join(args.run_dir, "eval")
    base = os.path.join(args.run_dir, "relearn", "without_imma")
    imma = os.path.join(args.run_dir, "relearn", "with_imma")

    import lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device).eval()
    compute_lpips_csv(args.reference_dir, base, imma, os.path.join(evald, "lpips.csv"), device, model)

    print(f"\nSGR over 50 epochs  ({args.run_dir}):")
    print(f"  {'metric':6s} {'MEAN(report)':>13s} {'peak':>16s} {'final':>13s}")
    with open(os.path.join(evald, "sgr_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_SGR", "peak_SGR", "peak_epoch", "final_SGR", "final_epoch"])
        for m in ["lpips", "clip", "dino"]:
            s = sgr_stats(os.path.join(evald, m + ".csv"))
            print(f"  {m.upper():6s} {s['mean']:>+12.2f}% {s['peak']:>+10.2f}%@ep{s['peak_epoch']:<2d} {s['final']:>+11.2f}%")
            w.writerow([m.upper(), f"{s['mean']:.2f}", f"{s['peak']:.2f}", s['peak_epoch'],
                        f"{s['final']:.2f}", s['final_epoch']])
    print(f"saved -> {os.path.join(evald, 'sgr_summary.csv')}")


if __name__ == "__main__":
    main()
