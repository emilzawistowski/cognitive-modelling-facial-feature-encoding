# Encoding of Facial Features: Perceived Dominance

A small psychophysics project that fits a linear encoding model relating facial
image structure to perceived dominance, then validates it by generating and
rating synthetic faces, and finally tests it with an adaptation after-effect
experiment.

Images are sourced from **UTKFace** (Aligned & Cropped), restricted to faces
labelled white, female, aged 30–39, to keep age/sex/ethnicity from confounding
the dominance ratings. 200 images survive automatic quality screening and
manual review (glasses, occlusion, pose, watermarks). PCA is run on the
processed 96×96 grayscale images, a small set of principal components is
selected by nested cross-validated forward selection, and a linear regression
on those components predicts mean dominance rating. The model is then used to
synthesize faces at target ratings and to run two follow-up experiments.

**Authors:** Krešimir Pavlov, Emil Zawistowski (DTU)

## Pipeline overview

Scripts in `report/scripts/` are numbered in the order they're meant to be run.

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_draw_candidates.py` | Automatic quality screening (sharpness, contrast, exposure) + seeded random ordering of candidates |
| 2 | `02_qc_review.py` | Manual keep/discard review (keyboard-driven, resumable) |
| 3 | `03_glasses_pass.py` | Prune remaining images wearing glasses from the kept set |
| 4 | `04_topup_review.py` | Top the kept set back up to 200 after pruning |
| 5 | `05_preprocess.py` | Grayscale → centre-crop → downsample → flatten into the PCA data matrix; also writes Experiment 1 stimuli |
| 6 | `06_analyse_exp1.py` | Experiment 1 analysis: rating distributions, reliability, per-image mean ratings (the regression target) |
| 7 | `07_pca_model.py` | PCA on the image matrix; nested-CV forward selection of components; fits the linear model |
| 8 | `08_generate_synthetic.py` | Generates synthetic faces at target predicted ratings from the fitted model |
| 9 | `09_analyse_exp2.py` | Experiment 2 analysis: validates the model against ratings of synthetic faces |
| 10 | `10_generate_exp3_stimuli.py` | Generates adapting and test stimuli for the adaptation after-effect experiment |
| 11 | `11_analyse_exp3.py` | Experiment 3 analysis: adaptation after-effect comparison |

The `report/experiments/facial_dominance_experiment_3.py` PsychoPy script runs
all three rating experiments (image presentation, response collection,
timing) and is the source of the CSVs consumed by steps 6, 9, and 11.

## Repository layout

```
report/
├── scripts/                        numbered pipeline scripts (see table above)
├── experiments/
│   ├── facial_dominance_experiment_3.py   PsychoPy experiment runner (all 3 experiments)
│   ├── data/                       raw per-participant rating CSVs
│   ├── stimuli/                    Experiment 1 stimuli (200 processed UTKFace images)
│   ├── stimuli_synthetic/          Experiment 2 stimuli (synthetic faces)
│   ├── stimuli_test/               Experiment 3 test-face stimuli
│   └── stimuli_adapt/              Experiment 3 adapting-face stimuli
└── figures/                        output figures referenced in the write-up
```

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `Pillow`
- [`psychopy`](https://www.psychopy.org/) — only required to run the experiment
  script (`facial_dominance_experiment_3.py`); not needed for the analysis
  pipeline

```bash
pip install numpy pandas scikit-learn scipy matplotlib Pillow psychopy
```

## Data

- Source images: [UTKFace](https://susanqq.github.io/UTKFace/) (Aligned & Cropped),
  not included in this repository — download separately and point
  `01_draw_candidates.py` at the local folder.
- `report/experiments/data/` contains the rating CSVs collected from the two
  participants for Experiments 1–3 (`1_exp1.csv`/`2_exp1.csv`,
  `1_exp2.csv`/`2_exp2.csv`, `1_exp3_aftereffect.csv`/`2_exp3_aftereffect.csv`).

## Usage

Run each script with `-h` for its exact CLI options. A typical end-to-end run:

```bash
cd report/scripts

python 01_draw_candidates.py --source /path/to/UTKFace ...
python 02_qc_review.py ...
python 03_glasses_pass.py ...
python 04_topup_review.py ...
python 05_preprocess.py ...

# collect Experiment 1 ratings with facial_dominance_experiment_3.py, then:
python 06_analyse_exp1.py ../experiments/data/1_exp1.csv ../experiments/data/2_exp1.csv

python 07_pca_model.py
python 08_generate_synthetic.py

# collect Experiment 2 ratings, then:
python 09_analyse_exp2.py ../experiments/data/1_exp2.csv ../experiments/data/2_exp2.csv

python 10_generate_exp3_stimuli.py

# collect Experiment 3 ratings, then:
python 11_analyse_exp3.py ../experiments/data/1_exp3_aftereffect.csv ../experiments/data/2_exp3_aftereffect.csv
```

## Method summary

- **Selection:** 780 UTKFace candidates (white, female, 30–39) → automatic
  quality screen (bottom 40% on sharpness/contrast/exposure discarded) →
  manual review for frontal pose, open eyes, no occlusion/watermark, no
  glasses → 200 final images.
- **Preprocessing:** grayscale, centre-cropped to 80%, downsampled to 96×96
  (9216-pixel data matrix).
- **Dimensionality reduction:** PCA without variance normalisation; candidate
  set pre-screened to the first 25 components (86.3% of variance) for
  statistical stability; forward selection scored by nested cross-validated
  R² retains 3 components (PC6, PC12, PC13).
- **Model:** linear regression on the 3 selected PC scores predicting mean
  dominance rating. In-sample R² = 0.302, nested CV R² = 0.259. The selected
  components track mouth/smile intensity rather than static structural
  features (brow, jaw) — a consequence of not controlling for expression at
  the selection stage.
- **Experiment 2 (validation):** synthetic faces generated at target
  predicted ratings; observed ratings correlate with model predictions at
  Spearman ρ ≈ 1.0 (face-level, pooled).
- **Experiment 3 (adaptation after-effect):** ratings of near-neutral test
  faces shift significantly depending on which end of the dominance
  continuum the participant adapted to beforehand, supporting the model as
  encoding a genuine perceptual dimension rather than a statistical artefact
  of the image set.
