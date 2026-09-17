"""
Step 3: turn the final 200 kept images into (a) a data matrix for PCA and
(b) the stimulus files for Experiment 1 in PsychoPy.

Grayscale -> centre crop (drops background and most of the hairline/shoulders)
-> downsample -> flatten.

Reads the current state of qc_decisions.csv, so it picks up whatever the
glasses pass and the top-up pass left behind. Stale stimuli from earlier runs
are cleared before new ones are written, since the face_NNN numbering shifts
whenever the selection changes.

Outputs (TAG is empty by default, "_norm" with --normalize)
  facesTAG.npy         n x d float32 matrix in 0..1, the PCA input
  final_setTAG.csv     filenames, ages and the stimulus each maps to
  stimuliTAG/*.png     display versions of the same processed images
  exp1_conditionsTAG.csv  PsychoPy conditions file for the trial loop
  pca_componentsTAG.png   mean face and first nine eigenfaces
  pca_screeTAG.png        cumulative variance explained

The stimuli are produced from the same array that goes into the PCA, then
upscaled to display size. That keeps Experiment 1 and Experiment 2 visually
matched - the synthetic faces in Experiment 2 must be rendered the same way.

Usage:
    python preprocess.py --root /path/to/utkcropped_30_to_39
    python preprocess.py --root /path/to/utkcropped_30_to_39 --normalize
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA


def load_face(path, crop_frac, size):
    img = Image.open(path).convert("L")
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_display_png(arr, display_size, path):
    """Same pipeline Experiment 2 must use for the synthetic faces."""
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="L")
    img = img.resize((display_size, display_size), Image.LANCZOS)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--crop-frac", type=float, default=0.80,
                    help="fraction of the 200x200 image kept, centred")
    ap.add_argument("--size", type=int, default=96,
                    help="PCA resolution, side length in px")
    ap.add_argument("--display-size", type=int, default=400,
                    help="side length of the PNGs shown in PsychoPy")
    ap.add_argument("--normalize", action="store_true",
                    help="match each image to a common mean and contrast, "
                         "which suppresses lighting differences in the PCs")
    ap.add_argument("--tag", default=None,
                    help="suffix for the outputs; defaults to '_norm' with "
                         "--normalize, otherwise empty")
    ap.add_argument("--expect", type=int, default=200)
    args = ap.parse_args()

    tag = args.tag if args.tag is not None else ("_norm" if args.normalize else "")
    root, data = Path(args.root), Path(args.data)
    stim = data / f"stimuli{tag}"
    stim.mkdir(parents=True, exist_ok=True)

    # --- current state of the selection -----------------------------------
    dec = pd.read_csv(data / "qc_decisions.csv")
    kept = dec[dec.keep == 1].sort_values("rank").reset_index(drop=True)

    print(f"{len(dec)} images judged, {len(kept)} kept "
          f"({100 * len(kept) / len(dec):.0f}% keep rate)")
    if "reason" in dec.columns:
        r = dec.loc[dec.keep == 0, "reason"].fillna("").replace("", "quality")
        print("discard reasons: " + ", ".join(
            f"{k} {v}" for k, v in r.value_counts().items()))
    if len(kept) != args.expect:
        print(f"WARNING: expected {args.expect} kept images. "
              f"Run topup_review.py before continuing.")

    # clear stale stimuli - numbering shifts whenever the selection changes
    old_png = sorted(stim.glob("face_*.png"))
    for p in old_png:
        p.unlink()
    if old_png:
        print(f"cleared {len(old_png)} stimuli from a previous run")

    # --- build the matrix --------------------------------------------------
    imgs = np.stack([load_face(root / f, args.crop_frac, args.size)
                     for f in kept.filename])

    if args.normalize:
        flat = imgs.reshape(len(imgs), -1)
        mu, sd = flat.mean(1, keepdims=True), flat.std(1, keepdims=True)
        flat = (flat - mu) / np.maximum(sd, 1e-6) * 0.20 + 0.50
        imgs = np.clip(flat, 0, 1).reshape(imgs.shape)
        print("per-image intensity normalisation applied")

    X = imgs.reshape(len(imgs), -1)
    np.save(data / f"faces{tag}.npy", X)

    # --- stimuli for PsychoPy ---------------------------------------------
    stim_names = []
    for i, arr in enumerate(imgs, 1):
        name = f"face_{i:03d}.png"
        to_display_png(arr, args.display_size, stim / name)
        stim_names.append(name)

    kept["stimulus"] = stim_names
    kept.to_csv(data / f"final_set{tag}.csv", index=False)

    pd.DataFrame({
        "image": [f"stimuli{tag}/{n}" for n in stim_names],
        "stim_id": [Path(n).stem for n in stim_names],
        "source_file": kept.filename,
    }).to_csv(data / f"exp1_conditions{tag}.csv", index=False)

    digest = hashlib.sha1("\n".join(sorted(kept.filename)).encode()).hexdigest()[:12]
    print(f"matrix {X.shape}  ({args.size}x{args.size} px)  fingerprint={digest}")
    print(f"wrote {len(stim_names)} stimuli to {stim} at {args.display_size}px")
    print(f"wrote {data / f'exp1_conditions{tag}.csv'}")
    print("\nage distribution of the final set:")
    print(kept.age.value_counts().sort_index().to_string())

    # --- sanity check ------------------------------------------------------
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print()
    for k in (5, 10, 20, 50):
        if k < len(cum):
            print(f"  first {k:3d} PCs explain {100 * cum[k - 1]:.1f}% of variance")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    axes[0, 0].imshow(pca.mean_.reshape(args.size, args.size), cmap="gray")
    axes[0, 0].set_title("mean", fontsize=8)
    for j, ax in enumerate(axes.ravel()[1:]):
        ax.imshow(pca.components_[j].reshape(args.size, args.size), cmap="gray")
        ax.set_title(f"PC{j + 1} ({100 * pca.explained_variance_ratio_[j]:.1f}%)",
                     fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"normalised={args.normalize}", fontsize=9)
    fig.tight_layout()
    fig.savefig(data / f"pca_components{tag}.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(cum) + 1), 100 * cum)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative variance explained (%)")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(data / f"pca_scree{tag}.png", dpi=150)
    print(f"wrote figures to {data}")


if __name__ == "__main__":
    main()
