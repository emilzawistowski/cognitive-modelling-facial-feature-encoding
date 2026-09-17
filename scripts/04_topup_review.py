"""
Pass B - top the selection back up to 200.

Picks up the candidate list where you left off, skipping everything already
recorded in qc_decisions.csv, and stops the moment the keep count hits the
target again. Same random order as before, so the set stays a random draw.

Resumable, and flushes after every keypress.

Keys:
    k / right arrow  keep
    d / left arrow   discard - quality (blurry, pose, occlusion, watermark)
    g                discard - glasses
    u                undo last decision
    q                quit

Usage:
    python topup_review.py --root /path/to/utkcropped_30_to_39
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

RULES = ("discard: glasses, blurry/upsampled, head turned off frontal, "
         "eyes closed, occluded, watermark")


class TopUp:
    def __init__(self, root, cand, dec_path, target):
        self.root = Path(root)
        self.dec_path = Path(dec_path)
        self.target = target

        self.dec = pd.read_csv(self.dec_path)
        if "reason" not in self.dec.columns:
            self.dec["reason"] = ""
        self.dec["reason"] = self.dec["reason"].fillna("")
        if "checked_glasses" not in self.dec.columns:
            self.dec["checked_glasses"] = 1

        seen = set(self.dec.filename)
        self.todo = (cand[~cand.filename.isin(seen)]
                     .sort_values("rank").reset_index(drop=True))
        self.pos = 0
        self.added = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6.6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    @property
    def n_keep(self):
        return int(self.dec.keep.sum())

    def flush(self):
        self.dec.to_csv(self.dec_path, index=False)

    def draw(self):
        self.ax.clear()
        done = self.n_keep >= self.target
        out_of = self.pos >= len(self.todo)
        if done or out_of:
            msg = f"done - {self.n_keep} kept" if done else (
                f"candidate list exhausted at {self.n_keep} kept\n"
                f"re-run draw_candidates.py with a larger --n-candidates"
            )
            self.ax.text(0.5, 0.5, msg, ha="center", va="center")
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            return
        row = self.todo.iloc[self.pos]
        self.ax.imshow(Image.open(self.root / row.filename))
        self.ax.axis("off")
        self.ax.set_title(
            f"{row.filename}   age {row.age}   rank {int(row['rank'])}\n"
            f"kept {self.n_keep}/{self.target}   "
            f"added this session {self.added}   "
            f"{len(self.todo) - self.pos} unseen candidates\n"
            f"[k]eep  [d]iscard  [g]lasses  [u]ndo  [q]uit",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "q":
            self.flush()
            plt.close(self.fig)
            return

        if event.key == "u":
            if self.pos == 0 or self.dec.empty:
                return
            last = self.dec.index[-1]
            if self.dec.at[last, "keep"] == 1:
                self.added -= 1
            self.dec = self.dec.drop(index=last)
            self.pos -= 1
        elif event.key in ("k", "right", "d", "left", "g"):
            if self.pos >= len(self.todo) or self.n_keep >= self.target:
                return
            row = self.todo.iloc[self.pos]
            keep = int(event.key in ("k", "right"))
            reason = "" if keep else ("glasses" if event.key == "g" else "quality")
            self.dec.loc[len(self.dec)] = {
                "rank": int(row["rank"]),
                "filename": row.filename,
                "age": int(row.age),
                "keep": keep,
                "reason": reason,
                "checked_glasses": 1,
            }
            self.added += keep
            self.pos += 1
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
    t = TopUp(args.root, cand, data / "qc_decisions.csv", args.target)

    print(RULES)
    print(f"currently {t.n_keep} kept, need {max(0, args.target - t.n_keep)} more")
    print(f"{len(t.todo)} unseen candidates available")
    t.draw()
    plt.show()
    print(f"\nadded {t.added}, now at {t.n_keep} kept")


if __name__ == "__main__":
    main()
