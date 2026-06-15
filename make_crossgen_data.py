import os, csv, argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    prompt = f"An artwork by {a.artist}"
    out = f"data/{a.key}_crossgen"
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if "xl" in a.model:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(a.model, torch_dtype=dtype).to(device)
        gen_res = 1024
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=dtype, safety_checker=None).to(device)
        gen_res = 768 if "2-1" in a.model else 512
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for i in range(a.num):
        g = torch.Generator(device=device).manual_seed(a.seed + i)
        img = pipe(prompt, height=gen_res, width=gen_res, generator=g,
                   num_inference_steps=30, guidance_scale=7.5).images[0]
        img = img.resize((a.res, a.res))
        fn = f"{i}.png"; img.save(os.path.join(out, fn)); rows.append([fn, prompt])
        print("saved", fn)
    with open(os.path.join(out, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file_name", "prompt"]); w.writerows(rows)
    print(f"DONE: {out}  ({a.num} imgs from {a.model})")

if __name__ == "__main__":
    main()
