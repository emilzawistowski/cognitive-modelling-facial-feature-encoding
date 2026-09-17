"""
Step 8 analysis: the adaptation after-effect.

Reads one trial-level CSV per participant with columns
    trial_index, adapt_filename, test_filename, rating, rt_seconds
and produces the plot and the comparison the assignment asks for: ratings for
each test stimulus, separately for the two adaptation conditions.

Predicted after-effect: perception is repulsed away from the adapting stimulus,
so a test face should be rated MORE dominant after adapting to the low-dominance
endpoint, and LESS dominant after adapting to the high-dominance endpoint.

Outputs
  exp3_aftereffect.png   one panel per participant, three test stimuli, two conditions
  exp3_results.txt       cell means, differences, and tests

Usage:
    python analyse_exp3.py 1_exp3_aftereffect.csv 2_exp3_aftereffect.csv --out data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LOW, HIGH = "adapt_low.png", "adapt_high.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-ratings", type=float, nargs="*", default=None,
                    help="predicted rating of each test stimulus, in name order "
                         "(e.g. 2.8 3.1 3.4), used for axis labels only")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    data = {Path(f).stem: pd.read_csv(f) for f in args.files}
    ids = list(data)
    tests = sorted(set().union(*[set(d.test_filename) for d in data.values()]))

    labels = {}
    for i, t in enumerate(tests):
        if args.test_ratings and i < len(args.test_ratings):
            labels[t] = f"{Path(t).stem}\n(predicted {args.test_ratings[i]:g})"
        else:
            labels[t] = Path(t).stem

    all_diffs = []
    for pid, d in data.items():
        first = d.adapt_filename.iloc[0]
        say(f"=== {pid}   {len(d)} trials, first block: {Path(first).stem}")
        say("  test        after low    after high   difference   Mann-Whitney")
        for t in tests:
            lo = d[(d.test_filename == t) & (d.adapt_filename == LOW)].rating
            hi = d[(d.test_filename == t) & (d.adapt_filename == HIGH)].rating
            diff = lo.mean() - hi.mean()
            all_diffs.append(diff)
            try:
                u, p = stats.mannwhitneyu(lo, hi, alternative="greater")
                ptxt = f"U = {u:.0f}, p = {p:.3f}"
            except ValueError:
                ptxt = "n/a"
            say(f"  {Path(t).stem:<10} {lo.mean():5.2f} ({lo.std():.2f})  "
                f"{hi.mean():5.2f} ({hi.std():.2f})  {diff:+6.2f}      {ptxt}")

        lo_all = d[d.adapt_filename == LOW].rating
        hi_all = d[d.adapt_filename == HIGH].rating
        u, p = stats.mannwhitneyu(lo_all, hi_all, alternative="greater")
        say(f"  pooled over test stimuli: {lo_all.mean():.2f} vs "
            f"{hi_all.mean():.2f}, difference {lo_all.mean() - hi_all.mean():+.2f}, "
            f"U = {u:.0f}, p = {p:.2g}")
        say()

    n_pos = sum(1 for x in all_diffs if x > 0)
    say(f"direction of the effect: {n_pos} of {len(all_diffs)} comparisons in the "
        f"predicted direction (higher ratings after adapting to the low endpoint)")
    if len(all_diffs):
        say(f"mean difference across all comparisons: {np.mean(all_diffs):+.2f} "
            f"scale points")
    sign_p = stats.binomtest(n_pos, len(all_diffs), 0.5,
                             alternative="greater").pvalue
    say(f"sign test across the {len(all_diffs)} comparisons: p = {sign_p:.4f}")

    # --- plot --------------------------------------------------------------
    n = len(ids)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.2), squeeze=False, sharey=True)
    width = 0.34
    x = np.arange(len(tests))
    for ax, pid in zip(axes[0], ids):
        d = data[pid]
        for off, cond, colour, name in [(-width / 2, LOW, "tab:blue", "after low-dominance adaptor"),
                                        (+width / 2, HIGH, "tab:red", "after high-dominance adaptor")]:
            groups = [d[(d.test_filename == t) & (d.adapt_filename == cond)].rating
                      for t in tests]
            bp = ax.boxplot(groups, positions=x + off, widths=width * 0.8,
                            patch_artist=True, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=colour, alpha=0.35)
            for med in bp["medians"]:
                med.set(color="black")
            ax.plot(x + off, [g.mean() for g in groups], "o", color=colour,
                    ms=6, label=name, zorder=3)
            # individual trials, jittered
            for xi, g in zip(x + off, groups):
                ax.scatter(np.full(len(g), xi) + np.random.uniform(-0.05, 0.05, len(g)),
                           g, s=10, color=colour, alpha=0.55, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[t] for t in tests], fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("test stimulus")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper center")
    axes[0][0].set_ylabel("dominance rating of the test face")
    fig.tight_layout()
    fig.savefig(out / "exp3_aftereffect.png", dpi=150)
    plt.close(fig)

    (out / "exp3_results.txt").write_text("\n".join(lines))
    print(f"\nwrote {out / 'exp3_aftereffect.png'} and {out / 'exp3_results.txt'}")


if __name__ == "__main__":
    main()
