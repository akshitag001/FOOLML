"""Extra metrics for the final submission, computed from the saved PNGs only (the target model is not used).

- LPIPS (AlexNet backbone, `lpips` package) between each clean image and its adversarial PNG
- how many images have an L-inf change above 8/255 and above 16/255
- which wrong class each true class was flipped to (from results.csv)

Writes results/final/extra_metrics.json, adds an LPIPS column to results.csv and the summary numbers to
results/final/summary.json. Run from the repo root:  python report/extra_metrics.py
"""
import csv
import json
from collections import Counter

import lpips
import numpy as np
import torch
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
STEMS = [f"{c}_{i}" for c in CLASSES for i in range(100)]
device = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


clean = np.stack([load(f"p2_data/clean_images/{s}.png") for s in STEMS])
adv = np.stack([load(f"p2_data/adv_images/{s}_adv.png") for s in STEMS])

# L-inf in 1/255 steps (both images are 8-bit, so this is exact)
linf_steps = np.abs(adv.astype(int) - clean.astype(int)).reshape(len(STEMS), -1).max(1)

# LPIPS expects (N, 3, H, W) in [-1, 1]
to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
torch.manual_seed(0)
net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
with torch.no_grad():
    lp = torch.cat([net(to_t(clean[k:k + 100]), to_t(adv[k:k + 100])).flatten().cpu()
                    for k in range(0, len(STEMS), 100)]).numpy()

# results.csv: add / refresh the LPIPS column
rows = list(csv.DictReader(open("results.csv")))
assert [r["image"] for r in rows] == [f"{s}_adv.png" for s in STEMS]
for r, v in zip(rows, lp):
    r["LPIPS"] = f"{v:.6f}"
with open("results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# label flips: true class -> class the model now predicts
flips = {c: Counter() for c in CLASSES}
for r in rows:
    flips[r["true_label"]][r["adv_label"]] += 1
flip_table = {c: {d: flips[c][d] for d in CLASSES} for c in CLASSES}

extra = {
    "lpips_net": "alex",
    "avg_LPIPS": float(lp.mean()),
    "median_LPIPS": float(np.median(lp)),
    "max_LPIPS": float(lp.max()),
    "n_linf_gt_8_255": int((linf_steps > 8).sum()),
    "n_linf_gt_16_255": int((linf_steps > 16).sum()),
    "linf_steps_max": int(linf_steps.max()),
    "linf_steps_median": float(np.median(linf_steps)),
    "flip_table": flip_table,
    "most_common_target": Counter(r["adv_label"] for r in rows).most_common(),
}
json.dump(extra, open("results/final/extra_metrics.json", "w"), indent=2)

summary = json.load(open("results/final/summary.json"))
summary.update({k: extra[k] for k in ("avg_LPIPS", "n_linf_gt_8_255", "n_linf_gt_16_255")})
json.dump(summary, open("results/final/summary.json", "w"), indent=2)

print(f"LPIPS mean {extra['avg_LPIPS']:.5f}  median {extra['median_LPIPS']:.5f}  max {extra['max_LPIPS']:.5f}")
print(f"L-inf > 8/255: {extra['n_linf_gt_8_255']} images   > 16/255: {extra['n_linf_gt_16_255']} images   "
      f"(median {extra['linf_steps_median']:.0f}/255, max {extra['linf_steps_max']}/255)")
print("Predicted class after attack:", extra["most_common_target"])
