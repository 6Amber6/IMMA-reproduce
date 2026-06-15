import os, glob, argparse
from PIL import Image, ImageDraw, ImageFont

ARTISTS = {"vangogh": "Van Gogh", "tyleredlin": "Tyler Edlin", "kilianeng": "Kilian Eng"}

def gens_at_epoch(d, epoch):
    fs = sorted(glob.glob(os.path.join(d, f"{epoch}_*.png")),
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not fs:
        eps = sorted({int(os.path.basename(f).split("_")[0])
                      for f in glob.glob(os.path.join(d, "*_*.png"))
                      if os.path.basename(f).split("_")[0].isdigit()})
        if eps:
            epoch = eps[-1]
            fs = sorted(glob.glob(os.path.join(d, f"{epoch}_*.png")),
                        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return fs, epoch

def list_refs(d):
    fs = []
    for e in ("*.png", "*.jpg", "*.jpeg"):
        fs += glob.glob(os.path.join(d, e))
    return sorted(fs)

def nearest_refs(ref_paths, gen_paths, device):
    try:
        import torch, clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        def emb(paths):
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
            with torch.no_grad():
                f = model.encode_image(ims).float()
            return f / f.norm(dim=-1, keepdim=True)
        rf, gf = emb(ref_paths), emb(gen_paths)
        return (gf @ rf.t()).argmax(dim=1).tolist(), True
    except Exception as e:
        print(f"  (CLIP unavailable: {e}; spread-pick)")
        n = len(gen_paths); step = max(1, len(ref_paths) // n)
        return [min(i * step, len(ref_paths) - 1) for i in range(n)], False

def pick_distinct(nn_idx):
    cols, used = [], set()
    for gi, ri in enumerate(nn_idx):
        if ri not in used:
            cols.append((gi, ri)); used.add(ri)
        if len(cols) == 3: break
    while len(cols) < 3 and len(cols) < len(nn_idx):
        for gi, ri in enumerate(nn_idx):
            if gi not in [c[0] for c in cols]:
                cols.append((gi, ri)); break
    return cols[:3]

def load_sq(p, s): return Image.open(p).convert("RGB").resize((s, s))
def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=list(ARTISTS) + ["other"])
    ap.add_argument("--epoch", type=int, default=49)
    ap.add_argument("--cell", type=int, default=256)
    a = ap.parse_args()
    artist = ARTISTS.get(a.key, a.key)
    ref_dir = f"data/{a.key}_crossgen"
    nat_wo  = f"runs_natural/{a.key}_crossgen/relearn/without_imma"
    nat_wi  = f"runs_natural/{a.key}_crossgen/relearn/with_imma"
    out     = f"compare_{a.key}_crossgen.png"
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    refs = list_refs(ref_dir)
    wo_all, ep = gens_at_epoch(nat_wo, a.epoch)
    wi_all, _  = gens_at_epoch(nat_wi, a.epoch)
    if not refs or not wo_all or not wi_all:
        print(f"ERROR [{a.key}]: refs={len(refs)} wo={len(wo_all)} wi={len(wi_all)}"); return
    nn_idx, used = nearest_refs(refs, wo_all, device)
    cols = pick_distinct(nn_idx)
    print(f"[{a.key}] epoch={ep} CLIP={used}")
    for k, (gi, ri) in enumerate(cols):
        print(f"  col{k+1}: ref='{os.path.basename(refs[ri])}' <- wo seed {gi}")
    rows = [
        (f"{artist} (SDXL ref)", [refs[ri]   for _, ri in cols]),
        ("Cross-gen w/o IMMA",   [wo_all[gi] for gi, _ in cols]),
        ("Cross-gen w/ IMMA",    [wi_all[gi] for gi, _ in cols]),
    ]
    cell, lab_w, pad, title_h = a.cell, 250, 8, 40
    W = lab_w + 3 * (cell + pad) + pad
    H = title_h + len(rows) * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(canvas)
    fl, ft = font(18), font(19)
    draw.text((pad, 11), f"{artist}: SDXL cross-gen attack (epoch {ep})", fill="black", font=ft)
    y = title_h + pad
    for label, fs in rows:
        draw.text((pad, y + cell // 2 - 10), label, fill="black", font=fl)
        x = lab_w
        for p in fs:
            canvas.paste(load_sq(p, cell), (x, y)); x += cell + pad
        y += cell + pad
    canvas.save(out)
    print(f"saved -> {out}  ({W}x{H})")

if __name__ == "__main__":
    main()
