"""
Step 8: stimuli for Experiment 3, the adaptation after-effect experiment.

Generates two adapting faces at the endpoints of the continuum and three test
faces close to the neutral face, all by the same method as Experiment 2
(Equations 2.27 and 2.28), with a fixation point drawn at the centre of every
image.

The neutral face is the mean image, which the model rates at delta (the
intercept). The test faces are placed symmetrically around it, so that an
after-effect has room to move the rating in either direction.

Outputs
  exp3_stimuli/adapt_low.png, adapt_high.png
  exp3_stimuli/test_1.png, test_2.png, test_3.png
  exp3_conditions.csv        six conditions, two blocks
  exp3_strip.png             all five stimuli side by side, for the report
  exp3_report.txt

Usage:
    python generate_exp3_stimuli.py --data data
    python generate_exp3_stimuli.py --data data --adapt 2.0 5.0 --test 2.8 3.1 3.4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def synth(model, a0):
    """Image vector for target rating a0, by Eq. 2.27 / 2.28."""
    w = np.asarray(model["coef"], dtype=float)
    sel = model["selected"]
    delta = float(model["intercept"])
    alpha = (a0 - delta) / float(w @ w)
    j0 = alpha * w
    flat = model["mean"] + j0 @ model["components"][sel]
    excursion = float(np.max(np.abs(j0) / model["score_sd"][sel]))
    clipped = float(((flat < 0) | (flat > 1)).mean())
    return flat, alpha, excursion, clipped


def save_png(flat, size, display_size, path, fixation=True):
    img = Image.fromarray(
        (np.clip(flat, 0, 1).reshape(size, size) * 255).astype(np.uint8), mode="L"
    ).resize((display_size, display_size), Image.LANCZOS).convert("RGB")

    if fixation:
        d = ImageDraw.Draw(img)
        c = display_size / 2
        r_out, r_in = display_size * 0.022, display_size * 0.009
        # white ring with a black centre, visible on any background
        d.ellipse([c - r_out, c - r_out, c + r_out, c + r_out],
                  fill=(255, 255, 255))
        d.ellipse([c - r_in, c - r_in, c + r_in, c + r_in], fill=(0, 0, 0))
    img.save(path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tag", default="")
    ap.add_argument("--display-size", type=int, default=400,
                    help="must match Experiments 1 and 2")
    ap.add_argument("--adapt", type=float, nargs=2, default=None,
                    help="the two endpoint ratings; defaults to the ends of the "
                         "Experiment 2 range")
    ap.add_argument("--test", type=float, nargs=3, default=None,
                    help="three test ratings near neutral; defaults to "
                         "delta-0.3, delta, delta+0.3")
    ap.add_argument("--reps", type=int, default=5,
                    help="repetitions of each condition within its block")
    args = ap.parse_args()

    data = Path(args.data)
    stim = data / f"exp3_stimuli{args.tag}"
    stim.mkdir(parents=True, exist_ok=True)

    m = np.load(data / f"model{args.tag}.npz")
    delta = float(m["intercept"])
    size = int(m["size"])
    w = np.asarray(m["coef"], dtype=float)
    sel = m["selected"]
    obs = m["scores"][:, sel] @ w / float(w @ w)
    lo_pred = delta + obs.min() * float(w @ w)
    hi_pred = delta + obs.max() * float(w @ w)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"model intercept (neutral face rating) delta = {delta:.2f}")
    say(f"model predicts {lo_pred:.2f} to {hi_pred:.2f} for the training images")

    adapt = args.adapt if args.adapt else (round(lo_pred + 0.07, 1),
                                           round(hi_pred - 0.17, 1))
    test = args.test if args.test else (round(delta - 0.3, 1),
                                        round(delta, 1),
                                        round(delta + 0.3, 1))
    say(f"adapting stimuli: {adapt[0]:g} (low) and {adapt[1]:g} (high)")
    say(f"test stimuli: {', '.join(f'{t:g}' for t in test)}")
    say()

    say("stimulus      target  alpha    excursion (SD)  clipped")
    made = []
    for label, a0 in [("adapt_low", adapt[0]), ("adapt_high", adapt[1])]:
        flat, alpha, exc, clip = synth(m, a0)
        save_png(flat, size, args.display_size, stim / f"{label}.png")
        say(f"  {label:<12} {a0:5.2f}  {alpha:7.1f}   {exc:6.2f}"
            f"          {100*clip:4.1f}%")
        made.append((label, a0, flat))
    for i, a0 in enumerate(test, 1):
        flat, alpha, exc, clip = synth(m, a0)
        save_png(flat, size, args.display_size, stim / f"test_{i}.png")
        say(f"  test_{i:<8} {a0:5.2f}  {alpha:7.1f}   {exc:6.2f}"
            f"          {100*clip:4.1f}%")
        made.append((f"test_{i}", a0, flat))

    # --- conditions --------------------------------------------------------
    rows = []
    for block, (alab, arate) in enumerate(
            [("adapt_low", adapt[0]), ("adapt_high", adapt[1])], start=1):
        for i, t in enumerate(test, 1):
            for rep in range(1, args.reps + 1):
                rows.append({
                    "block": block,
                    "adapt_condition": alab,
                    "adapt_image": f"exp3_stimuli{args.tag}/{alab}.png",
                    "adapt_rating": arate,
                    "test_id": f"test_{i}",
                    "test_image": f"exp3_stimuli{args.tag}/test_{i}.png",
                    "test_rating": t,
                    "rep": rep,
                    "condition": f"{alab}__test_{i}",
                })
    cond = pd.DataFrame(rows)
    cond.to_csv(data / f"exp3_conditions{args.tag}.csv", index=False)
    say()
    say(f"{cond.condition.nunique()} conditions, {args.reps} reps each, "
        f"{len(cond)} trials total, in 2 blocks of {len(cond)//2}")

    # --- figure for the report --------------------------------------------
    fig, axes = plt.subplots(1, len(made), figsize=(1.5 * len(made), 2.0))
    for ax, (label, a0, flat) in zip(axes, made):
        ax.imshow(np.clip(flat, 0, 1).reshape(size, size), cmap="gray",
                  vmin=0, vmax=1)
        ax.plot(size / 2, size / 2, "o", ms=4, mfc="white", mec="black", mew=1)
        ax.set_title(f"{label}\n{a0:g}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Experiment 3 stimuli", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(data / f"exp3_strip{args.tag}.png", dpi=180)
    plt.close(fig)

    (data / f"exp3_report{args.tag}.txt").write_text("\n".join(lines))
    say(f"wrote stimuli to {stim} and exp3_conditions{args.tag}.csv")


if __name__ == "__main__":
    main()
