"""
Step 1: automatic quality screening, then a seeded random ordering of candidates
from the already-filtered folder (white females, 30-39).

Computes three cheap image-quality metrics for every file, rejects the obvious
failures, then shuffles what survives and writes candidates.csv - the order in
which images go through manual QC.

Metrics
  sharpness  variance of the Laplacian, divided by the image variance so it
             measures detail rather than contrast. Low values mean the original
             was small and has been upsampled, or the shot is out of focus.
  contrast   standard deviation of pixel intensity. Very low means washed out.
  exposure   mean intensity plus the fraction of clipped pixels. Catches images
             that are nearly black or blown out.

Watermarks and head pose are not screened here - no reliable cheap test - so
they stay part of the manual pass.

Workflow
  1. python draw_candidates.py --root <folder> --report-only
     Look at quality_examples.png and the printed percentiles, pick a threshold.
  2. python draw_candidates.py --root <folder> --blur-percentile 15
     Re-run with the threshold you chose.

UTKFace filenames: [age]_[gender]_[race]_[date&time].jpg
  gender: 0 = male, 1 = female
  race:   0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace


def parse_filename(path: Path):
    """Return (age, gender, race) or None if the name is malformed."""
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if gender not in (0, 1) or race not in (0, 1, 2, 3, 4):
        return None
    return age, gender, race


def quality_metrics(path: Path):
    """Sharpness, contrast and exposure for one image, on the 0..1 scale."""
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    # measure the centre only, so background clutter does not inflate detail
    h, w = g.shape
    c = g[h // 6: 5 * h // 6, w // 6: 5 * w // 6]
    var = c.var()
    lap = laplace(c)
    return {
        "sharpness": float(lap.var() / var) if var > 1e-6 else 0.0,
        "contrast": float(np.sqrt(var)),
        "mean_level": float(c.mean()),
        "clipped": float(((c < 0.02) | (c > 0.98)).mean()),
    }


def montage(paths, titles, out_path, suptitle):
    n = len(paths)
    fig, axes = plt.subplots(3, n // 3, figsize=(1.6 * (n // 3), 5.4))
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(Image.open(p))
        ax.set_title(t, fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder holding the pre-filtered .jpg files")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--n-candidates", type=int, default=450)
    ap.add_argument("--report-only", action="store_true",
                    help="compute metrics and write diagnostics, reject nothing")
    ap.add_argument("--blur-percentile", type=float, default=15.0,
                    help="reject this %% of the pool with the lowest sharpness")
    ap.add_argument("--blur-min", type=float, default=None,
                    help="absolute sharpness threshold; overrides the percentile")
    ap.add_argument("--min-contrast", type=float, default=0.08)
    ap.add_argument("--max-clipped", type=float, default=0.10)
    ap.add_argument("--level-range", type=float, nargs=2, default=(0.12, 0.88))
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, bad = [], []
    files = sorted(root.glob("*.jpg"))
    for i, p in enumerate(files, 1):
        parsed = parse_filename(p)
        if parsed is None:
            bad.append(p.name)
            continue
        age, gender, race = parsed
        rows.append({"filename": p.name, "age": age, "gender": gender,
                     "race": race, **quality_metrics(p)})
        if i % 100 == 0:
            print(f"  scored {i}/{len(files)}")

    pool = pd.DataFrame(rows)
    print(f"\nfound {len(pool)} usable files in {root}")
    if bad:
        print(f"WARNING: {len(bad)} malformed filenames skipped, e.g. {bad[:3]}")

    off = pool[~(pool.age.between(30, 39) & (pool.gender == 1) & (pool.race == 0))]
    if len(off):
        print(f"WARNING: {len(off)} files do not match the intended subset:")
        print(off.head(10).to_string(index=False))
    else:
        print("all files match the intended subset (white, female, 30-39)")

    # --- quality screening ------------------------------------------------
    qs = pool.sharpness
    print("\nsharpness percentiles:")
    for q in (1, 5, 10, 15, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(qs, q):.3f}")

    thr = args.blur_min if args.blur_min is not None \
        else float(np.percentile(qs, args.blur_percentile))

    reasons = pd.Series("", index=pool.index)
    reasons[qs < thr] += "blurry;"
    reasons[pool.contrast < args.min_contrast] += "low-contrast;"
    reasons[pool.clipped > args.max_clipped] += "clipped;"
    reasons[~pool.mean_level.between(*args.level_range)] += "exposure;"
    pool["reject_reason"] = reasons.str.rstrip(";")
    pool["auto_reject"] = pool.reject_reason != ""

    print(f"\nsharpness threshold {thr:.3f}")
    counts = pool.loc[pool.auto_reject, "reject_reason"].value_counts()
    print(counts.to_string() if len(counts) else "  nothing rejected")
    print(f"auto-rejected {pool.auto_reject.sum()} of {len(pool)}, "
          f"{(~pool.auto_reject).sum()} remain")

    pool.sort_values("filename").to_csv(out / "pool.csv", index=False)

    # visual calibration: worst, borderline, best
    srt = pool.sort_values("sharpness").reset_index(drop=True)
    near = int((srt.sharpness - thr).abs().idxmin())
    lo = max(0, min(near - 3, len(srt) - 6))
    picks = list(range(6)) + list(range(lo, lo + 6)) + list(range(len(srt) - 6, len(srt)))
    montage([root / srt.filename[i] for i in picks],
            [f"{srt.sharpness[i]:.2f}" for i in picks],
            out / "quality_examples.png",
            f"top: blurriest   middle: near threshold {thr:.2f}   bottom: sharpest")
    print(f"wrote {out/'quality_examples.png'} and {out/'pool.csv'}")

    if args.report_only:
        print("\nreport-only mode, no candidate list written")
        return

    # --- seeded random draw from the survivors ----------------------------
    keep = pool[~pool.auto_reject].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(keep))
    cand = keep.iloc[order].head(args.n_candidates).reset_index(drop=True)
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))
    cand["seed"] = args.seed
    cand.to_csv(out / "candidates.csv", index=False)

    digest = hashlib.sha1("\n".join(sorted(cand.filename)).encode()).hexdigest()[:12]
    print(f"\nwrote {len(cand)} candidates to {out/'candidates.csv'}")
    print(f"seed={args.seed}  sharpness_threshold={thr:.4f}  fingerprint={digest}")


if __name__ == "__main__":
    main()
