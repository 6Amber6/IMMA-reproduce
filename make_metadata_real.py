"""
Generate metadata.csv for data/vangogh_real/ in HuggingFace datasets format.
Each row: file_name, prompt
Format must match data/vangogh/metadata.csv exactly.
"""
import csv
from pathlib import Path

DATA_DIR = Path("data/vangogh_real")
PROMPT = "An artwork by Van Gogh"  # same as SD-generated set

# Collect all jpg files, sorted for reproducibility
files = sorted([p.name for p in DATA_DIR.glob("*.jpg")])
print(f"Found {len(files)} images in {DATA_DIR}")

# Write metadata.csv
out_path = DATA_DIR / "metadata.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "prompt"])
    for fn in files:
        writer.writerow([fn, PROMPT])

print(f"Wrote {out_path}")
print("\nFirst 5 lines:")
with open(out_path) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {line.rstrip()}")
