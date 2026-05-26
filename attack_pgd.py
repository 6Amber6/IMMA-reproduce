"""
Ex2 Phase 2.2: Generate L_inf adversarial perturbations against the
ESD + IMMA immunized UNet via PGD with random restarts.

Implementation follows Madry et al. (ICLR 2018) standard:
  - K steps of projected gradient descent within an L_inf ball of radius epsilon
  - N independent random initializations (restarts)
  - The restart with the lowest FINAL-step loss is kept

[v5 - PGD-40 with restarts + resume]
  - fp32 unet/vae (stable gradients)
  - EOT=4 (averages out diffusion stochasticity)
  - Rebuild graph inside EOT loop (no graph reuse)
  - 3 random restarts, keep restart with lowest final loss
  - --skip_existing flag for resume mode
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_image_as_tensor(path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_tensor_as_image(tensor, path):
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def encode_prompt(prompt, tokenizer, text_encoder, device):
    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(tokens.input_ids.to(device))[0]
    return embeds


def build_immunized_unet(pretrained_name, delta_ckpt, imma_ckpt, device, dtype):
    print(f"Loading UNet from {pretrained_name} ...")
    unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
    print(f"Applying ESD delta from {delta_ckpt} ...")
    unet.load_state_dict(torch.load(delta_ckpt, map_location="cpu"))
    print(f"Applying IMMA xatten from {imma_ckpt} ...")
    model_dict = unet.state_dict()
    imma_dict = torch.load(imma_ckpt, map_location="cpu")
    print(f"  IMMA ckpt overrides {len(imma_dict)} tensors")
    model_dict.update(imma_dict)
    unet.load_state_dict(model_dict)
    unet = unet.to(device=device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()
    return unet


def pgd_single_run(
    x_clean, prompt_embeds,
    unet, vae, scheduler,
    epsilon, alpha, num_steps,
    t_min, t_max, eot_samples,
    device, dtype,
    restart_idx,
    log_every=10,
):
    """
    One standard PGD-K run (single random init).
    Returns: (delta_final, loss_history).
    """
    delta = (torch.rand_like(x_clean) * 2 - 1) * epsilon
    delta = delta.detach().to(dtype=torch.float32)
    delta.requires_grad_(True)

    vae_scale_factor = 0.18215
    loss_history = []

    for step in range(num_steps):
        grad_accum = torch.zeros_like(delta)
        loss_accum = 0.0

        # EOT: fresh autograd graph each iter
        for _ in range(eot_samples):
            x_adv = (x_clean + delta).clamp(0.0, 1.0)
            x_for_vae = (x_adv * 2.0 - 1.0).unsqueeze(0).to(dtype=dtype)

            latent = vae.encode(x_for_vae).latent_dist.sample() * vae_scale_factor

            t = torch.randint(t_min, t_max, (1,), device=device).long()
            noise = torch.randn_like(latent)
            noisy_latent = scheduler.add_noise(latent, noise, t)

            noise_pred = unet(noisy_latent, t, encoder_hidden_states=prompt_embeds).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            grad_accum = grad_accum + grad.detach()
            loss_accum += loss.item()

        grad_avg = grad_accum / eot_samples
        loss_avg = loss_accum / eot_samples
        loss_history.append(loss_avg)

        # Standard PGD step
        with torch.no_grad():
            delta = delta - alpha * grad_avg.sign()
            delta = delta.clamp(-epsilon, epsilon)
            delta = (x_clean + delta).clamp(0.0, 1.0) - x_clean
        delta = delta.detach().requires_grad_(True)

        if (step + 1) % log_every == 0 or step == 0:
            print(f"      restart {restart_idx}  step {step+1:3d}/{num_steps}: loss = {loss_avg:.4f}")

    return delta.detach(), loss_history


def pgd_attack_with_restarts(
    x_clean, prompt_embeds,
    unet, vae, scheduler,
    epsilon, alpha, num_steps, num_restarts,
    t_min, t_max, eot_samples,
    device, dtype,
):
    """
    PGD-K with N random restarts (Madry standard).
    Keep the restart with the lowest FINAL-step loss.
    """
    best_final_loss = float("inf")
    best_delta = None
    best_history = None
    best_restart = -1

    for r in range(num_restarts):
        print(f"    --- Restart {r+1}/{num_restarts} ---")
        delta_r, hist_r = pgd_single_run(
            x_clean, prompt_embeds, unet, vae, scheduler,
            epsilon, alpha, num_steps,
            t_min, t_max, eot_samples,
            device, dtype,
            restart_idx=r+1,
        )
        final_loss = hist_r[-1]
        print(f"    Restart {r+1} final loss: {final_loss:.4f}  (init: {hist_r[0]:.4f})")
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_delta = delta_r.clone()
            best_history = hist_r
            best_restart = r + 1

    x_adv = (x_clean + best_delta).clamp(0.0, 1.0)
    return x_adv, best_history, best_restart


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--delta_ckpt", default="diffusers-VanGogh-ESDx1-UNET.pt")
    parser.add_argument("--imma_ckpt", default="results/relearning/imma_unet_xatten_layer.pt")
    parser.add_argument("--src_dir", default="data/vangogh")
    parser.add_argument("--dst_dir", default="data/vangogh_adv")
    parser.add_argument("--prompt", default="An artwork by Van Gogh")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--alpha", type=float, default=2/255)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--num_restarts", type=int, default=3,
                        help="Number of random restarts (Madry standard)")
    parser.add_argument("--t_min", type=int, default=50)
    parser.add_argument("--t_max", type=int, default=500)
    parser.add_argument("--eot_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--use_fp32", action="store_true", default=True)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already exist in dst_dir (resume mode)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.use_fp32 else torch.float16

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PGD-{args.num_steps} with {args.num_restarts} restarts (Madry et al. 2018)")
    print(f"  eps   = {args.epsilon:.4f}  ({args.epsilon*255:.1f}/255)")
    print(f"  alpha = {args.alpha:.4f}  ({args.alpha*255:.1f}/255)")
    print(f"  EOT samples = {args.eot_samples}")
    print(f"  t in [{args.t_min}, {args.t_max})")
    print(f"  dtype = {dtype}")
    print(f"  skip_existing = {args.skip_existing}")
    print("=" * 60)
    print()

    # ===== Load models =====
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet = build_immunized_unet(
        args.pretrained_model_name_or_path,
        args.delta_ckpt, args.imma_ckpt,
        device, dtype,
    )

    prompt_embeds = encode_prompt(args.prompt, tokenizer, text_encoder, device)
    print(f"Prompt: '{args.prompt}'")
    print(f"Prompt embeds shape: {prompt_embeds.shape}")
    print()

    # ===== Targets =====
    all_pngs = sorted(src_dir.glob("*.png"),
                      key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    targets = all_pngs[:args.num_images]

    # [resume] Skip images that already exist in dst_dir
    if args.skip_existing:
        skipped = []
        to_attack = []
        for p in targets:
            if (dst_dir / p.name).exists():
                skipped.append(p.name)
            else:
                to_attack.append(p)
        if skipped:
            print(f"[resume] Skipping {len(skipped)} already-attacked images: {sorted(skipped, key=lambda x: int(x.split('.')[0]))}")
        targets = to_attack

    print(f"Attacking {len(targets)} images from {src_dir}")
    print()
    if len(targets) == 0:
        print("Nothing to do. Exiting.")
        return

    # ===== Attack loop =====
    t_start = time.time()
    all_results = []
    for idx, src_path in enumerate(targets):
        print(f"[{idx+1}/{len(targets)}] {src_path.name}")
        x_clean = load_image_as_tensor(src_path, size=args.resolution).to(device)

        x_adv, best_history, best_restart = pgd_attack_with_restarts(
            x_clean, prompt_embeds, unet, vae, scheduler,
            args.epsilon, args.alpha, args.num_steps, args.num_restarts,
            args.t_min, args.t_max, args.eot_samples,
            device, dtype,
        )

        dst_path = dst_dir / src_path.name
        save_tensor_as_image(x_adv, dst_path)

        delta_final = (x_adv - x_clean).cpu().numpy()
        max_dev = np.abs(delta_final).max()
        reduction = (best_history[0] - best_history[-1]) / max(best_history[0], 1e-6) * 100

        all_results.append({
            "name": src_path.name,
            "init_loss": best_history[0],
            "final_loss": best_history[-1],
            "reduction_pct": reduction,
            "max_delta": max_dev,
            "best_restart": best_restart,
        })

        print(f"  >> kept restart {best_restart}: "
              f"loss {best_history[0]:.4f} -> {best_history[-1]:.4f}  "
              f"(reduction: {reduction:.1f}%, max|delta|: {max_dev:.4f})")
        print()

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/max(len(targets),1):.1f}s per image)")
    print("=" * 60)

    # ===== Summary =====
    if all_results:
        reds = [r["reduction_pct"] for r in all_results]
        n_positive = sum(1 for r in reds if r > 0)
        print()
        print("Summary (newly attacked images):")
        print(f"  Mean reduction:      {np.mean(reds):+.1f}%")
        print(f"  Median reduction:    {np.median(reds):+.1f}%")
        print(f"  Min / Max reduction: {min(reds):+.1f}% / {max(reds):+.1f}%")
        print(f"  Images with reduction > 0:   {n_positive}/{len(reds)}")
        print(f"  Images with reduction > 30%: {sum(1 for r in reds if r > 30)}/{len(reds)}")
        print()

    # ===== Metadata: include ALL images (existing + newly added) =====
    src_meta = src_dir / "metadata.csv"
    if src_meta.exists():
        # Rebuild metadata based on the actual contents of dst_dir
        existing_names = {p.name for p in dst_dir.glob("*.png")}
        with open(src_meta) as fin, open(dst_dir / "metadata.csv", "w", newline="") as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            writer.writerow(next(reader))
            for row in reader:
                if row and row[0] in existing_names:
                    writer.writerow(row)
        print(f"Wrote metadata.csv covering {len(existing_names)} images")

    # ===== Append per-image stats (don't overwrite previous runs' stats) =====
    if all_results:
        stats_path = dst_dir / "attack_stats.csv"
        write_header = not stats_path.exists()
        with open(stats_path, "a", newline="") as fout:
            writer = csv.writer(fout)
            if write_header:
                writer.writerow(["name", "init_loss", "final_loss", "reduction_pct", "max_delta", "best_restart"])
            for r in all_results:
                writer.writerow([r["name"], f"{r['init_loss']:.4f}", f"{r['final_loss']:.4f}",
                                 f"{r['reduction_pct']:.1f}", f"{r['max_delta']:.4f}", r["best_restart"]])
        print(f"Appended stats: {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
