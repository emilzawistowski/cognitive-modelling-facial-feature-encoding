"""
Experiment 1 analysis: rating distributions, reliability, and the target
variable for the linear model.

Reads one CSV per participant (filename, rating_1, rating_2) and writes
  exp1_histograms.png   per-participant histograms, both presentations
  exp1_reliability.txt  the numbers to quote in the report
  exp1_ratings.csv      one row per image, per-participant means and the
                        grand mean used as the dependent variable

Usage:
    python analyse_exp1.py 1_exp1.csv 2_exp1.csv
    python analyse_exp1.py *.csv --normalise
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def spearman_brown(r, k):
    return k * r / (1 + (k - 1) * r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one CSV per participant")
    ap.add_argument("--out", default="data")
    ap.add_argument("--normalise", action="store_true",
                    help="min-max each participant onto 0..1 before averaging")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subs, means, lines = {}, {}, []
    for f in args.files:
        pid = Path(f).stem
        df = pd.read_csv(f).set_index("filename")
        subs[pid] = df
        means[pid] = df[["rating_1", "rating_2"]].mean(axis=1)

    ids = list(subs)
    ref = subs[ids[0]].index
    for pid, df in subs.items():
        if not df.index.equals(ref):
            raise SystemExit(f"{pid}: image list differs from {ids[0]}")

    # --- per-participant --------------------------------------------------
    for pid, df in subs.items():
        both = pd.concat([df.rating_1, df.rating_2])
        r = np.corrcoef(df.rating_1, df.rating_2)[0, 1]
        used = sorted(both.unique())
        lines += [
            f"{pid}",
            f"  mean {both.mean():.2f}  sd {both.std():.2f}  "
            f"range used {min(used)}-{max(used)}",
            f"  scale points used: {used}",
            f"  test-retest r = {r:.3f}  "
            f"(exact agreement {(df.rating_1 == df.rating_2).mean() * 100:.0f}%, "
            f"within 1 point {(abs(df.rating_1 - df.rating_2) <= 1).mean() * 100:.0f}%)",
            f"  reliability of this participant's mean (Spearman-Brown) "
            f"{spearman_brown(r, 2):.3f}",
            "",
        ]

    # --- between participants ---------------------------------------------
    M = pd.DataFrame(means)
    if len(ids) > 1:
        cm = M.corr()
        lines.append("inter-rater correlations (participant mean ratings)")
        lines.append(cm.round(3).to_string())
        off = cm.values[np.triu_indices(len(ids), 1)]
        rbar = float(off.mean())
        rel = spearman_brown(rbar, len(ids))
        lines += [
            "",
            f"mean inter-rater r = {rbar:.3f}",
            f"reliability of the {len(ids)}-rater mean = {rel:.3f}",
            f"upper bound on the Experiment 2 correlation ~ {np.sqrt(rel):.2f}",
            "",
            "projected reliability with more raters:",
        ]
        for k in (4, 6, 8, 10, 15):
            lines.append(f"  {k:2d} raters -> {spearman_brown(rbar, k):.3f}")
        lines.append("")

    # --- dependent variable ------------------------------------------------
    S = M.copy()
    if args.normalise:
        S = (S - S.min()) / (S.max() - S.min())
        lines.append("min-max normalisation applied per participant")
    S["mean_rating"] = S.mean(axis=1)
    S.index.name = "filename"
    S.to_csv(out / "exp1_ratings.csv")

    g = S["mean_rating"]
    lines += [
        f"dependent variable: mean {g.mean():.2f}  sd {g.std():.2f}  "
        f"range {g.min():.2f}-{g.max():.2f}",
    ]

    (out / "exp1_reliability.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # --- histograms --------------------------------------------------------
    n = len(ids)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    bins = np.arange(0.5, 6.5, 1)
    for i, (ax, pid) in enumerate(zip(axes[0], ids), start=1):
        df = subs[pid]
        ax.hist([df.rating_1, df.rating_2], bins=bins,
                label=["presentation 1", "presentation 2"])
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("dominance rating")
        ax.set_ylabel("number of images")
        ax.set_title(f"Participant {i}  (mean "
                     f"{pd.concat([df.rating_1, df.rating_2]).mean():.2f})",
                     fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "exp1_histograms.png", dpi=150)
    print(f"\nwrote {out / 'exp1_histograms.png'} and {out / 'exp1_ratings.csv'}")


if __name__ == "__main__":
    main()
