import os, glob, argparse
from PIL import Image, ImageDraw, ImageFont

ARTISTS = [("vangogh","Van Gogh"),("tyleredlin","Tyler Edlin"),("kilianeng","Kilian Eng")]
PAPER_WITH = {
    "vangogh":    "runs/20260611_202332/relearn_clean/with_imma",
    "tyleredlin": "runs_repro/tyleredlin/relearn/with_imma",
    "kilianeng":  "runs_repro/kilianeng/relearn/with_imma",
}
ROW_LABELS = ["SDXL ref", "Paper w/ IMMA", "Cross-gen w/ IMMA"]

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
    for e in ("*.png","*.jpg","*.jpeg"):
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
        n = len(gen_paths); step = max(1, len(ref_paths)//n)
        return [min(i*step, len(ref_paths)-1) for i in range(n)], False

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
def ctext(d, cx, y, text, fnt):
    try: w = d.textlength(text, font=fnt)
    except Exception: w = len(text)*fnt.size*0.5
    d.text((cx - w/2, y), text, fill="black", font=fnt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=49)
    ap.add_argument("--cell", type=int, default=200)
    a = ap.parse_args()
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    cell, pad, lab_w, head_h, title_h, gap = a.cell, 6, 165, 28, 36, 24
    blocks = []
    for key, name in ARTISTS:
        refs = list_refs(f"data/{key}_crossgen")
        cg, ep = gens_at_epoch(f"runs_natural/{key}_crossgen/relearn/with_imma", a.epoch)
        pp, _  = gens_at_epoch(PAPER_WITH[key], a.epoch)
        if not refs or not cg or not pp:
            print(f"  !! WARN [{key}] refs={len(refs)} cg={len(cg)} paper={len(pp)} (check paths)")
        nn, used = nearest_refs(refs, cg, device) if (refs and cg) else ([0,1,2], False)
        cols = pick_distinct(nn) if nn else [(0,0),(1,1),(2,2)]
        r_ref = [refs[ri] for _, ri in cols] if refs else []
        r_pp  = [pp[gi]   for gi, _ in cols] if pp else []
        r_cg  = [cg[gi]   for gi, _ in cols] if cg else []
        blocks.append((name, [r_ref, r_pp, r_cg]))
        print(f"[{key}] ep={ep} CLIP={used} cols={[(gi, os.path.basename(refs[ri]) if refs else '?') for gi,ri in cols]}")
    block_w = 3*(cell+pad)
    W = lab_w + len(ARTISTS)*(block_w+gap)
    H = title_h + head_h + 3*(cell+pad) + pad
    cv = Image.new("RGB",(W,H),"white"); d = ImageDraw.Draw(cv)
    ft, fh, fl = font(20), font(18), font(15)
    d.text((pad, 8), "IMMA: paper-setting blocks the style, SDXL cross-gen breaks through", font=ft, fill="black")
    y0 = title_h + head_h
    for r, lab in enumerate(ROW_LABELS):
        d.text((pad, y0 + r*(cell+pad) + cell//2 - 8), lab, font=fl, fill="black")
    x0 = lab_w
    for name, rows in blocks:
        ctext(d, x0 + block_w/2, title_h + 2, name, fh)
        for r, row in enumerate(rows):
            y = y0 + r*(cell+pad); x = x0
            for p in row:
                if p: cv.paste(load_sq(p, cell), (x, y))
                x += cell+pad
        x0 += block_w + gap
    cv.save("compare_crossgen_all.png")
    print("saved -> compare_crossgen_all.png", cv.size)

if __name__ == "__main__":
    main()
