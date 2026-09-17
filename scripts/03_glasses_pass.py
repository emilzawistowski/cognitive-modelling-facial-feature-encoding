"""
Pass A - prune the current selection.

Walks only the images currently marked keep=1 in qc_decisions.csv and lets you
demote the ones wearing glasses (or anything else you missed the first time).
Writes back to the same qc_decisions.csv, so pass B can top the set back up.

Resumable: every keypress is flushed to disk, and already-checked images are
skipped when you restart.

Keys:
    k / right arrow  keep
    g / left arrow   discard - glasses
    x                discard - other (watermark, pose, occlusion...)
    u                undo last decision
    q                quit

Usage:
    python glasses_pass.py --root /path/to/utkcropped_30_to_39
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class GlassesPass:
    def __init__(self, root, dec_path):
        self.root = Path(root)
        self.dec_path = Path(dec_path)
        self.dec = pd.read_csv(self.dec_path)

        if "reason" not in self.dec.columns:
            self.dec["reason"] = ""
        self.dec["reason"] = self.dec["reason"].fillna("")
        if "checked_glasses" not in self.dec.columns:
            self.dec["checked_glasses"] = 0

        self.queue = list(self.dec.index[self.dec.keep == 1])
        self.history = []

        self.fig, self.ax = plt.subplots(figsize=(6, 6.6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    @property
    def n_keep(self):
        return int(self.dec.keep.sum())

    def next_idx(self):
        for i in self.queue:
            if self.dec.at[i, "checked_glasses"] == 0 and self.dec.at[i, "keep"] == 1:
                return i
        return None

    def n_left(self):
        return sum(
            1 for i in self.queue
            if self.dec.at[i, "checked_glasses"] == 0 and self.dec.at[i, "keep"] == 1
        )

    def flush(self):
        self.dec.to_csv(self.dec_path, index=False)

    def draw(self):
        i = self.next_idx()
        self.ax.clear()
        if i is None:
            self.ax.text(0.5, 0.5,
                         f"done - {self.n_keep} still kept\n"
                         f"need {max(0, 200 - self.n_keep)} more",
                         ha="center", va="center")
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            return
        row = self.dec.loc[i]
        self.ax.imshow(Image.open(self.root / row.filename))
        self.ax.axis("off")
        self.ax.set_title(
            f"{row.filename}   age {row.age}\n"
            f"{self.n_left()} left to check   currently keeping {self.n_keep}\n"
            f"[k]eep  [g]lasses  [x] other  [u]ndo  [q]uit",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "q":
            self.flush()
            plt.close(self.fig)
            return

        if event.key == "u":
            if not self.history:
                return
            i, prev = self.history.pop()
            self.dec.at[i, "keep"] = prev["keep"]
            self.dec.at[i, "reason"] = prev["reason"]
            self.dec.at[i, "checked_glasses"] = 0
        elif event.key in ("k", "right", "g", "left", "x"):
            i = self.next_idx()
            if i is None:
                return
            self.history.append(
                (i, {"keep": self.dec.at[i, "keep"],
                     "reason": self.dec.at[i, "reason"]})
            )
            self.dec.at[i, "checked_glasses"] = 1
            if event.key in ("g", "left"):
                self.dec.at[i, "keep"] = 0
                self.dec.at[i, "reason"] = "glasses"
            elif event.key == "x":
                self.dec.at[i, "keep"] = 0
                self.dec.at[i, "reason"] = "other"
        else:
            return

        self.flush()
        self.draw()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    p = GlassesPass(args.root, Path(args.data) / "qc_decisions.csv")
    print(f"checking {len(p.queue)} currently-kept images")
    p.draw()
    plt.show()

    removed = p.dec[(p.dec.keep == 0) & (p.dec.reason != "")]
    print(f"\nremoved {len(removed)} "
          f"({(removed.reason == 'glasses').sum()} glasses, "
          f"{(removed.reason == 'other').sum()} other)")
    print(f"{p.n_keep} kept, need {max(0, 200 - p.n_keep)} more - run topup_review.py")


if __name__ == "__main__":
    main()
