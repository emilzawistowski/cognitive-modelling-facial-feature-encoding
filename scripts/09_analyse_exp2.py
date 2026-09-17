"""
Step 7: analyse Experiment 2.

Takes one CSV per participant holding the trial-by-trial ratings of the
synthetic faces, joins them to the rating the model predicted for each face, and
produces the box plot and Spearman correlation the assignment asks for.

Input format, one row per trial, per participant. The script accepts either
  stim_id, rating              (long, one row per presentation)
  image,   rating
  stim_id, rating_1, rating_2, ...   (wide, one column per repetition)

Outputs
  exp2_boxplot.png     ratings against predicted rating, one panel per participant
  exp2_scatter.png     mean observed rating against predicted rating
  exp2_results.txt     Spearman rho, Pearson r and the per-face means

Usage:
    python analyse_exp2.py 1_exp2.csv 2_exp2.csv --data data
    python analyse_exp2.py 1_exp2.csv 2_exp2.csv --data data --suffix _v2
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def norm_id(s):
    """synth_r2_3.png / synthetic/synth_r2.3.png -> synth_r2.3"""
    s = str(s).split("/")[-1].split("\\")[-1]
    s = re.sub(r"\.png$", "", s)
    return re.sub(r"(?<=\d)_(?=\d)", ".", s)


def to_long(df):
    """Return a frame with columns stim_id, rating - one row per presentation."""
    cols = {c.lower(): c for c in df.columns}

    if "stim_id" in cols:
        key = cols["stim_id"]
    elif "image" in cols:
        key = cols["image"]
    elif "filename" in cols:
        key = cols["filename"]
    else:
        raise SystemExit(f"no stim_id / image / filename column in {list(df.columns)}")

    rating_cols = [c for c in df.columns if c.lower().startswith("rating")]
    if not rating_cols:
        raise SystemExit(f"no rating column in {list(df.columns)}")

    renamed = {c: f"__rep{i}" for i, c in enumerate(rating_cols)}
    out = df.rename(columns=renamed).melt(
        id_vars=[key], value_vars=list(renamed.values()),
        var_name="rep", value_name="rating")
    out = out.rename(columns={key: "stim_id"}).dropna(subset=["rating"])
    out["stim_id"] = out.stim_id.map(norm_id)
    out["rating"] = out.rating.astype(float)
    return out[["stim_id", "rating"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one CSV per participant")
    ap.add_argument("--data", default="data")
    ap.add_argument("--suffix", default="",
                    help="which stimulus set, e.g. _v2; must match "
                         "exp2_conditions<suffix>.csv")
    args = ap.parse_args()

    data = Path(args.data)
    cond = pd.read_csv(data / f"exp2_conditions{args.suffix}.csv")
    cond["stim_id"] = cond.stim_id.map(norm_id)
    pred = cond.set_index("stim_id").predicted_rating

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    subs = {}
    for f in args.files:
        pid = Path(f).stem
        d = to_long(pd.read_csv(f))
        missing = set(d.stim_id) - set(pred.index)
        if missing:
            raise SystemExit(f"{pid}: stimuli not in the conditions file: "
                             f"{sorted(missing)[:5]}")
        d["predicted"] = pred.loc[d.stim_id].to_numpy()
        subs[pid] = d
        n_per = d.groupby("stim_id").size()
        say(f"{pid}: {len(d)} trials, {d.stim_id.nunique()} stimuli, "
            f"{n_per.min()}-{n_per.max()} presentations each")
    say()

    # --- correlations ------------------------------------------------------
    say("per participant, on the mean rating per face")
    for pid, d in subs.items():
        m = d.groupby("predicted").rating.mean()
        rho, p_rho = spearmanr(m.index, m.values)
        r, p_r = pearsonr(m.index, m.values)
        say(f"  {pid}: Spearman rho = {rho:+.3f} (p = {p_rho:.4f}), "
            f"Pearson r = {r:+.3f} (p = {p_r:.4f})")

    say()
    say("per participant, on every individual trial")
    for pid, d in subs.items():
        rho, p_rho = spearmanr(d.predicted, d.rating)
        say(f"  {pid}: Spearman rho = {rho:+.3f} (p = {p_rho:.4g}), "
            f"n = {len(d)}")

    pooled = pd.concat(subs.values())
    pm = pooled.groupby("predicted").rating.mean()
    rho, p_rho = spearmanr(pm.index, pm.values)
    r, p_r = pearsonr(pm.index, pm.values)
    say()
    say(f"pooled across participants, mean rating per face:")
    say(f"  Spearman rho = {rho:+.3f} (p = {p_rho:.4f})")
    say(f"  Pearson  r   = {r:+.3f} (p = {p_r:.4f})")
    say()
    say("mean observed rating per face")
    say("  predicted   observed   sd")
    for t, g in pooled.groupby("predicted"):
        say(f"  {t:8.2f}   {g.rating.mean():8.2f}   {g.rating.std():.2f}")

    # --- box plot ----------------------------------------------------------
    n = len(subs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4), squeeze=False,
                             sharey=True)
    for ax, (pid, d) in zip(axes[0], subs.items()):
        levels = sorted(d.predicted.unique())
        ax.boxplot([d.rating[d.predicted == t].to_numpy() for t in levels],
                   positions=range(len(levels)), widths=0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{t:g}" for t in levels], rotation=45, fontsize=8)
        ax.set_xlabel("predicted rating")
        ax.set_ylabel("observed rating")
        rho, _ = spearmanr(d.predicted, d.rating)
        ax.set_title(f"{pid}   Spearman rho = {rho:+.2f}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(data / f"exp2_boxplot{args.suffix}.png", dpi=150)
    plt.close(fig)

    # --- scatter of the means ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for pid, d in subs.items():
        m = d.groupby("predicted").rating.mean()
        ax.plot(m.index, m.values, "o-", label=pid, alpha=0.8)
    lim = [min(pm.index.min(), pm.min()) - 0.3, max(pm.index.max(), pm.max()) + 0.3]
    ax.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    ax.set_xlabel("predicted rating")
    ax.set_ylabel("mean observed rating")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(data / f"exp2_scatter{args.suffix}.png", dpi=150)
    plt.close(fig)

    (data / f"exp2_results{args.suffix}.txt").write_text("\n".join(lines))
    say()
    say(f"wrote exp2_boxplot{args.suffix}.png, exp2_scatter{args.suffix}.png "
        f"and exp2_results{args.suffix}.txt")


if __name__ == "__main__":
    main()
