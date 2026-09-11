"""
Select images for Experiment 1 (dominance ratings, female faces aged 30-50).

Reads the 'Aligned & Cropped' UTKFace folder, filters on the labels encoded in
the filename, and draws a reproducible sample of 300 files.

UTKFace filename format:  [age]_[gender]_[race]_[date&time].jpg(.chip.jpg)
    gender: 0 = male, 1 = female
    race:   0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other

Usage:
    python select_images.py
"""

import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------- settings --
SRC_DIR = Path("C:/Users/kresi/Desktop/Cognitive Modelling/Project/archive/utkcropped")   # folder with the raw .jpg files
OUT_DIR = Path("C:/Users/kresi/Desktop/Cognitive Modelling/Project/selection_exp1")       # where the 300 chosen files are copied
MANIFEST = Path("C:/Users/kresi/Desktop/Cognitive Modelling/Project/selection_exp1.csv")  # audit trail for the report appendix

SEED = 20260910          # any fixed integer; write it down in the report
N_TOTAL = 300

AGE_MIN, AGE_MAX = 30, 50
GENDER = 1               # female
RACE = 0                 # White; set to None to keep all races

# Age bands used for stratification, and how many to draw from each.
# Must sum to N_TOTAL.
BANDS = {
    "30-33": (30, 33, 60),
    "34-37": (34, 37, 60),
    "38-41": (38, 41, 60),
    "42-45": (42, 45, 60),
    "46-50": (46, 50, 60),
}

STRATIFY = True          # False -> plain simple random sample of N_TOTAL

# ------------------------------------------------------------------ parsing --
# A handful of UTKFace filenames are malformed (missing the race field).
# The strict pattern below skips them instead of mislabelling them.
PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+.*\.jpg$", re.IGNORECASE)


def parse(path: Path):
    """Return (age, gender, race) or None if the filename is malformed."""
    m = PATTERN.match(path.name)
    if not m:
        return None
    age, gender, race = (int(g) for g in m.groups())
    if gender not in (0, 1) or race not in range(5):
        return None
    return age, gender, race


def band_of(age: int):
    for name, (lo, hi, _) in BANDS.items():
        if lo <= age <= hi:
            return name
    return None


# ------------------------------------------------------------------- filter --
records, skipped = [], 0
for path in sorted(SRC_DIR.iterdir()):          # sorted() -> deterministic order
    if not path.is_file():
        continue
    parsed = parse(path)
    if parsed is None:
        skipped += 1
        continue
    age, gender, race = parsed
    if not (AGE_MIN <= age <= AGE_MAX):
        continue
    if gender != GENDER:
        continue
    if RACE is not None and race != RACE:
        continue
    records.append({"file": path.name, "age": age, "gender": gender, "race": race})

print(f"malformed filenames skipped : {skipped}")
print(f"candidate pool              : {len(records)}")

# ------------------------------------------------------------------- sample --
rng = random.Random(SEED)

if STRATIFY:
    by_band = defaultdict(list)
    for rec in records:
        b = band_of(rec["age"])
        if b:
            by_band[b].append(rec)

    selected, reserve = [], []
    for name, (_, _, n) in BANDS.items():
        pool = sorted(by_band[name], key=lambda r: r["file"])
        if len(pool) < n:
            raise ValueError(f"band {name} has only {len(pool)} images, need {n}")
        shuffled = pool[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[:n])
        reserve.extend(shuffled[n:])          # spares for post-QC top-up
else:
    pool = sorted(records, key=lambda r: r["file"])
    shuffled = pool[:]
    rng.shuffle(shuffled)
    selected, reserve = shuffled[:N_TOTAL], shuffled[N_TOTAL:]

print(f"selected                    : {len(selected)}")
print("age distribution            :", sorted(Counter(r['age'] for r in selected).items()))

# --------------------------------------------------------------------- copy --
OUT_DIR.mkdir(exist_ok=True)
for rec in selected:
    shutil.copy2(SRC_DIR / rec["file"], OUT_DIR / rec["file"])

with MANIFEST.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["file", "age", "gender", "race", "band"])
    writer.writeheader()
    for rec in selected:
        writer.writerow({**rec, "band": band_of(rec["age"])})

# Keep the unused candidates on record so you can top up after quality control
# without re-running the sampling (which would change the whole draw).
with Path("selection_exp1_reserve.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["file", "age", "gender", "race", "band"])
    writer.writeheader()
    for rec in reserve:
        writer.writerow({**rec, "band": band_of(rec["age"])})

print(f"copied to {OUT_DIR}/ and wrote {MANIFEST}")