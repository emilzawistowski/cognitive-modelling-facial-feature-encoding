"""
Step 2: manual quality control.

Shows each candidate in the random order from draw_candidates.py and records a
keep/discard decision. Resumable - decisions are flushed to disk after every
keypress, so you can quit and pick up later. Stops once the target number of
keeps is reached.

Keys:
    k / right arrow  keep
    d / left arrow   discard
    u                undo last decision
    q                quit

Usage:
    python qc_review.py --root /path/to/utkcropped
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REASONS = "low resolution / not full-frontal / occluded / watermark / other"


class Reviewer:
    def __init__(self, root, cand, dec_path, target):
        self.root = Path(root)
        self.cand = cand
        self.dec_path = Path(dec_path)
        self.target = target
        self.decisions = []
        if self.dec_path.exists():
            self.decisions = pd.read_csv(self.dec_path).to_dict("records")
        self.fig, self.ax = plt.subplots(figsize=(6, 6.6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    @property
    def n_keep(self):
        return sum(d["keep"] for d in self.decisions)

    @property
    def i(self):
        return len(self.decisions)

    def flush(self):
        pd.DataFrame(self.decisions).to_csv(self.dec_path, index=False)

    def draw(self):
        if self.n_keep >= self.target or self.i >= len(self.cand):
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"done - {self.n_keep} kept", ha="center")
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            return
        row = self.cand.iloc[self.i]
        img = Image.open(self.root / row.filename)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis("off")
        self.ax.set_title(
            f"{row.filename}   age {row.age}\n"
            f"reviewed {self.i}/{len(self.cand)}   kept {self.n_keep}/{self.target}\n"
            f"[k]eep  [d]iscard  [u]ndo  [q]uit",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "q":
            self.flush()
            plt.close(self.fig)
            return
        if event.key == "u" and self.decisions:
            self.decisions.pop()
        elif event.key in ("k", "right", "d", "left"):
            if self.i < len(self.cand) and self.n_keep < self.target:
                row = self.cand.iloc[self.i]
                self.decisions.append(
                    {
                        "rank": int(row["rank"]),
                        "filename": row.filename,
                        "age": int(row.age),
                        "keep": int(event.key in ("k", "right")),
                    }
                )
        else:
            return
        self.flush()
        self.draw()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", type=int, default=200)
    args = ap.parse_args()

    data = Path(args.data)
    cand = pd.read_csv(data / "candidates.csv")
    r = Reviewer(args.root, cand, data / "qc_decisions.csv", args.target)
    print("discard if: " + REASONS)
    r.draw()
    plt.show()
    print(f"kept {r.n_keep}, reviewed {r.i} of {len(cand)}")


if __name__ == "__main__":
    main()
