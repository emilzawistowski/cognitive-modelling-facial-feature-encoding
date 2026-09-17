"""
Steps 3 and 4: PCA on the stimulus images, then a linear model predicting the
dominance ratings from a small set of PC scores.

Inputs (all in --data)
  faces.npy          n x d matrix from preprocess.py  (use faces_norm.npy with --tag _norm)
  final_set.csv      maps matrix rows to stimulus names
  exp1_ratings.csv   from analyse_exp1.py, indexed by stimulus name

Outputs
  pca_variance.png        variance explained by every PC
  pc_effects.png          min / mean / max visualisation of the leading PCs
  pc_effects_selected.png the same for the PCs the model selected
  model_fit.png           predicted vs observed ratings
  model.npz               everything Experiment 2 needs to generate faces
  model_report.txt        the numbers to quote in the report

The PCA is centred but not scaled, which is what sklearn does by default and
what the assignment asks for. Do not add a StandardScaler.

Usage:
    python pca_model.py
    python pca_model.py --tag _norm --prescreen 25
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


# ----------------------------------------------------------------------------
# visualisation


def draw_effects(mean_img, comps, scores, idx, size, var_ratio, path, title=None):
    """Grid of PC effects: columns are the score level, rows are components.

    Column headers appear once, above the top row. Each row is labelled on the
    left with the component, the variance it explains and the range of observed
    scores. No internal title - everything else belongs in the caption.
    """
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(5.8, 1.78 * n), squeeze=False)
    headers = ["minimum score", "mean image", "maximum score"]

    for r, k in enumerate(idx):
        lo, hi = scores[:, k].min(), scores[:, k].max()
        for c, (ax, w) in enumerate(zip(axes[r], [lo, 0.0, hi])):
            img = (mean_img + w * comps[k]).reshape(size, size)
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(headers[c], fontsize=8.5, pad=4)
        axes[r][0].set_ylabel(f"PC{k + 1} ({100 * var_ratio[k]:.1f}%)\n"
                              f"scores {lo:+.0f} to {hi:+.0f}",
                              fontsize=8.5, labelpad=6)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06, wspace=0.06)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# forward selection


def forward_select(Z, y, cv, max_features, tol=0.002):
    """Greedy forward selection scored by cross-validated R^2."""
    chosen, remaining = [], list(range(Z.shape[1]))
    history, best = [], -np.inf
    while remaining and len(chosen) < max_features:
        scores = []
        for j in remaining:
            cols = chosen + [j]
            s = cross_val_score(LinearRegression(), Z[:, cols], y,
                                cv=cv, scoring="r2").mean()
            scores.append((s, j))
        s, j = max(scores)
        if s < best + tol:
            break
        best = s
        chosen.append(j)
        remaining.remove(j)
        history.append((list(chosen), s))
    return chosen, history


def nested_cv_r2(Z, y, outer_cv, inner_cv, max_features):
    """Honest estimate: selection is redone inside every outer fold."""
    preds = np.full(len(y), np.nan)
    for tr, te in outer_cv.split(Z):
        sel, _ = forward_select(Z[tr], y[tr], inner_cv, max_features)
        if not sel:
            preds[te] = y[tr].mean()
            continue
        m = LinearRegression().fit(Z[tr][:, sel], y[tr])
        preds[te] = m.predict(Z[te][:, sel])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, preds


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tag", default="", help="'_norm' for the normalised variant")
    ap.add_argument("--ratings", default=None,
                    help="defaults to <data>/exp1_ratings.csv")
    ap.add_argument("--target", default="mean_rating")
    ap.add_argument("--prescreen", type=int, default=25,
                    help="how many leading PCs forward selection may choose from")
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--show-pcs", type=int, default=5,
                    help="how many leading PCs to visualise")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    # --- load and align ----------------------------------------------------
    X = np.load(data / f"faces{args.tag}.npy")
    meta = pd.read_csv(data / f"final_set{args.tag}.csv")
    rat = pd.read_csv(args.ratings or (data / "exp1_ratings.csv"))
    key = rat.columns[0]
    rat = rat.set_index(key)

    if "stimulus" not in meta.columns:
        raise SystemExit("final_set.csv has no 'stimulus' column - re-run preprocess.py")
    if len(meta) != len(X):
        raise SystemExit(f"{len(meta)} metadata rows but {len(X)} image rows")

    missing = [s for s in meta.stimulus if s not in rat.index]
    if missing:
        raise SystemExit(f"{len(missing)} stimuli have no rating, e.g. {missing[:3]}")

    y = rat.loc[meta.stimulus, args.target].to_numpy(dtype=float)
    size = int(round(np.sqrt(X.shape[1])))
    say(f"{X.shape[0]} images, {size}x{size} px, target '{args.target}' "
        f"mean {y.mean():.2f} sd {y.std():.2f}")

    # --- PCA ---------------------------------------------------------------
    pca = PCA()                      # centres, does not scale
    S = pca.fit_transform(X)         # scores
    comps = pca.components_
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    say(f"PCs available: {S.shape[1]}")
    for k in (5, 10, 25, 50):
        if k <= len(cum):
            say(f"  first {k:3d} PCs explain {100 * cum[k - 1]:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(np.arange(1, len(var) + 1), 100 * var, width=1.0)
    ax.axvline(args.prescreen + 0.5, color="crimson", ls="--", lw=1,
               label=f"pre-screen cut ({args.prescreen} PCs, "
                     f"{100 * cum[args.prescreen - 1]:.0f}%)")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(data / f"pca_variance{args.tag}.png", dpi=150)
    plt.close(fig)

    draw_effects(pca.mean_, comps, S, range(args.show_pcs), size, var,
                 data / f"pc_effects{args.tag}.png")

    # --- forward selection -------------------------------------------------
    K = min(args.prescreen, S.shape[1])
    Z = S[:, :K]
    inner = KFold(5, shuffle=True, random_state=args.seed)
    outer = KFold(5, shuffle=True, random_state=args.seed + 1)

    sel, hist = forward_select(Z, y, inner, args.max_features)
    if not sel:
        raise SystemExit("forward selection kept nothing - no PC beats the "
                         "intercept-only model on cross-validation")

    say()
    say(f"forward selection over the first {K} PCs, max {args.max_features}")
    for cols, s in hist:
        say(f"  {len(cols)} PC(s) {[c + 1 for c in cols]}  cv R2 = {s:.3f}")

    model = LinearRegression().fit(Z[:, sel], y)
    pred_in = model.predict(Z[:, sel])
    r2_in = model.score(Z[:, sel], y)
    r2_cv = cross_val_score(LinearRegression(), Z[:, sel], y,
                            cv=inner, scoring="r2").mean()
    r2_nested, pred_nested = nested_cv_r2(Z, y, outer, inner, args.max_features)

    say()
    say(f"selected PCs: {[c + 1 for c in sel]}")
    say(f"coefficients: " + ", ".join(f"PC{c + 1} {b:+.4f}"
                                      for c, b in zip(sel, model.coef_)))
    say(f"intercept {model.intercept_:.3f}")
    say(f"in-sample R2      {r2_in:.3f}   (optimistic - selection saw this data)")
    say(f"cross-validated R2 {r2_cv:.3f}   (selection still saw this data)")
    say(f"nested CV R2       {r2_nested:.3f}   <- quote this one")
    say(f"correlation predicted vs observed r = "
        f"{np.corrcoef(pred_in, y)[0, 1]:.3f}")

    draw_effects(pca.mean_, comps, S, sel, size, var,
                 data / f"pc_effects_selected{args.tag}.png")

    # --- what Experiment 2 will have to do ---------------------------------
    w = np.zeros(K)
    w[sel] = model.coef_
    denom = float(w @ w)
    sd = S[:, :K].std(0)
    say()
    say("generating a face for a target rating: move from the mean image along")
    say("the coefficient vector, scores = t * w with t = (target - intercept) / |w|^2")
    say("  target   t        largest |score| in SD units")
    for target in (1, 2, 3, 4, 5):
        t = (target - model.intercept_) / denom
        worst = np.max(np.abs(t * w[sel]) / sd[sel])
        flag = "  <- beyond the data" if worst > 2.5 else ""
        say(f"  {target}      {t:8.1f}   {worst:.2f}{flag}")

    np.savez(data / f"model{args.tag}.npz",
             mean=pca.mean_, components=comps[:K], explained=var[:K],
             score_sd=sd, selected=np.array(sel), coef=model.coef_,
             intercept=model.intercept_, size=size,
             y=y, scores=S[:, :K])

    # --- diagnostics -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, p, ttl in zip(axes, [pred_in, pred_nested],
                          [f"in-sample (R2 {r2_in:.2f})",
                           f"nested CV (R2 {r2_nested:.2f})"]):
        ax.scatter(y, p, s=14, alpha=0.6)
        lim = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("observed mean rating")
        ax.set_ylabel("predicted rating")
        ax.set_title(ttl, fontsize=10)
    fig.tight_layout()
    fig.savefig(data / f"model_fit{args.tag}.png", dpi=150)
    plt.close(fig)

    (data / f"model_report{args.tag}.txt").write_text("\n".join(out))
    say()
    say(f"wrote model{args.tag}.npz and the figures to {data}")


if __name__ == "__main__":
    main()
