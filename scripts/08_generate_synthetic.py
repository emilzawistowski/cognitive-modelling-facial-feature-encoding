"""
Step 5: generate synthetic faces from the fitted model.

Follows Section 2.5. For a target rating a0 the unique image containing no
rating-irrelevant content is the one whose PC score vector is parallel to the
weight vector w:

    alpha = (a0 - delta) / ||w||^2        (2.27)
    j0    = alpha * w                     (2.28)

with delta the model intercept. The image is then reconstructed as the mean
image plus j0 projected back through the selected principal components.

Eleven faces are produced: ratings 1 to 5 (the levels used in Experiment 1) plus
0.5, 1.5, 2.5, 3.5, 4.5 and 5.5.

Outputs
  synthetic/synth_r0.5.png ...       one PNG per target, at Experiment 1 display size
  synthetic_strip.png                all eleven side by side, as in Figures 2.6 / 2.7
  exp2_conditions.csv                PsychoPy conditions file
  synthetic_report.txt               alpha, excursion and clipping per face

Usage:
    python generate_synthetic.py --data data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tag", default="")
    ap.add_argument("--display-size", type=int, default=400,
                    help="must match the size used for the Experiment 1 stimuli")
    ap.add_argument("--targets", type=float, nargs="+", default=TARGETS)
    args = ap.parse_args()

    data = Path(args.data)
    out_dir = data / f"synthetic{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = np.load(data / f"model{args.tag}.npz")
    mean_img = m["mean"]
    comps = m["components"]          # K x d, the pre-screened PCs
    sel = m["selected"]              # indices into comps
    coef = m["coef"]
    delta = float(m["intercept"])
    sd = m["score_sd"][sel]
    size = int(m["size"])
    scores = m["scores"][:, sel]     # observed scores on the selected PCs

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"model: PCs {[int(s) + 1 for s in sel]}, intercept (delta) {delta:.3f}")
    say(f"weights w: " + ", ".join(f"{c:+.4f}" for c in coef))

    w = np.asarray(coef, dtype=float)
    w_sq = float(w @ w)
    say(f"||w||^2 = {w_sq:.6f}")
    say(f"rating of the average face = delta = {delta:.2f}")
    say()

    # range the observed images actually cover along w
    obs_alpha = scores @ w / w_sq
    say(f"observed images span alpha from {obs_alpha.min():.1f} to "
        f"{obs_alpha.max():.1f}")
    say(f"which corresponds to ratings {delta + obs_alpha.min() * w_sq:.2f} "
        f"to {delta + obs_alpha.max() * w_sq:.2f}")
    say()

    say("target  alpha      max |score| in SD   pixels clipped")
    rows, imgs = [], []
    for a0 in args.targets:
        alpha = (a0 - delta) / w_sq
        j0 = alpha * w                                   # scores on selected PCs
        flat = mean_img + j0 @ comps[sel]                # back to image space
        clipped = float(((flat < 0) | (flat > 1)).mean())
        excursion = float(np.max(np.abs(j0) / sd))
        img = np.clip(flat, 0, 1).reshape(size, size)
        imgs.append(img)

        name = f"synth_r{a0:g}.png"
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        pil = pil.resize((args.display_size, args.display_size), Image.LANCZOS)
        pil.save(out_dir / name)

        flag = "   <- outside the data" if excursion > 2.5 else ""
        say(f"  {a0:4.1f}  {alpha:8.1f}   {excursion:6.2f}"
            f"            {100 * clipped:5.1f}%{flag}")
        rows.append({"image": f"synthetic{args.tag}/{name}",
                     "stim_id": f"synth_r{a0:g}",
                     "predicted_rating": a0,
                     "alpha": alpha,
                     "max_excursion_sd": excursion,
                     "pct_clipped": 100 * clipped})

    df = pd.DataFrame(rows)
    df.to_csv(data / f"exp2_conditions{args.tag}.csv", index=False)

    # --- side-by-side figure, as in Figures 2.6 and 2.7 --------------------
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(1.15 * n, 1.7))
    for ax, img, a0 in zip(axes, imgs, args.targets):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{a0:g}", fontsize=8)
        ax.axis("off")
    fig.suptitle("synthetic faces at predicted dominance ratings", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(data / f"synthetic_strip{args.tag}.png", dpi=180)
    plt.close(fig)

    # the mean face, for reference in the report
    Image.fromarray(
        (np.clip(mean_img, 0, 1).reshape(size, size) * 255).astype(np.uint8),
        mode="L"
    ).resize((args.display_size, args.display_size), Image.LANCZOS) \
     .save(out_dir / "mean_face.png")

    (data / f"synthetic_report{args.tag}.txt").write_text("\n".join(lines))
    say()
    say(f"wrote {len(imgs)} faces to {out_dir}")
    say(f"wrote synthetic_strip{args.tag}.png and exp2_conditions{args.tag}.csv")


if __name__ == "__main__":
    main()
