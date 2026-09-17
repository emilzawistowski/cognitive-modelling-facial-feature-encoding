"""
Facial Dominance Encoding Project
==================================
Single PsychoPy script handling all three experiments:

  Experiment 1 - Rate original UTKFace images on a 1-5 dominance scale
                 (each image shown twice, random order)
  Experiment 2 - Rate synthetic images generated from the linear model
                 (each image shown >=10 times, random order)
  Experiment 3 - Adaptation after-effect: view an adapting (extreme) face,
                 then briefly rate a near-neutral test face

Run this script directly in PsychoPy (Coder view) or via
`python facial_dominance_experiment.py` with PsychoPy installed.

Author: Emil Zawistowski, Kresimir Pavlov (DTU)
"""

import os
import glob
import random
import csv
from datetime import datetime

from psychopy import visual, core, event, gui, logging
from PIL import Image

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# Folders (relative to this script's location)
STIMULI_DIR_EXP1 = "stimuli"            # 200 original UTKFace images
STIMULI_DIR_EXP2 = "stimuli_synthetic"  # 11 synthetic images from the model
STIMULI_DIR_EXP3_ADAPT = "stimuli_adapt"  # 2 adapting (endpoint) images
STIMULI_DIR_EXP3_TEST = "stimuli_test"    # 3 near-neutral test images

DATA_DIR = "data"  # where output CSVs are saved

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

RATING_KEYS = ["1", "2", "3", "4", "5"]  # top-row number keys, safe on any layout
QUIT_KEY = "escape"

# Exp1: each image shown this many times (repeats), across the whole set
EXP1_REPEATS_PER_IMAGE = 2

# Exp2: each synthetic image shown at least this many times
EXP2_MIN_REPEATS_PER_IMAGE = 10

# Exp3 timing (seconds)
ADAPT_DURATION = 25.0        # 20-30s adapting stimulus (fixed here at 25s; edit if needed)
TEST_DURATION = 0.75         # 0.5-1.0s test stimulus (fixed here at 0.75s; edit if needed)
FIXATION_DURATION = 1.0      # short fixation before each adapting stimulus

# Target size of the LONGER image dimension, in norm units (-1 to 1 scale).
# The other dimension is derived from the image's real aspect ratio and the
# window's pixel aspect ratio, so square images stay square and non-square
# images keep their true proportions regardless of window shape.
IMAGE_TARGET_SIZE = 0.6

# Cache of (native_pixel_w, native_pixel_h) per image path, so we only
# read each file's dimensions once.
_IMAGE_PIXEL_SIZE_CACHE = {}


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def get_participant_info():
    """Show a dialog box to collect the participant's student ID and
    which experiment to run. Returns a dict."""
    info = {"Student ID": "", "Experiment (1/2/3)": "1"}
    dlg = gui.DlgFromDict(
        dictionary=info,
        title="Facial Dominance Experiment",
        order=["Student ID", "Experiment (1/2/3)"],
    )
    if not dlg.OK:
        core.quit()  # user pressed Cancel
    info["Student ID"] = info["Student ID"].strip()
    if info["Student ID"] == "":
        raise ValueError("Student ID cannot be empty.")
    if info["Experiment (1/2/3)"] not in ("1", "2", "3"):
        raise ValueError("Experiment must be 1, 2, or 3.")
    return info


def load_image_list(folder):
    """Return a sorted list of full paths to all images in `folder`."""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    if not paths:
        raise FileNotFoundError(
            f"No images found in '{folder}'. Check the folder exists and "
            f"contains .jpg/.jpeg/.png files."
        )
    return sorted(paths)


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def check_quit(keys_pressed):
    if QUIT_KEY in keys_pressed:
        core.quit()


def get_native_pixel_size(image_path):
    """Return (width_px, height_px) of the image file, cached after the
    first read so repeated trials with the same image don't re-open it."""
    if image_path not in _IMAGE_PIXEL_SIZE_CACHE:
        with Image.open(image_path) as im:
            _IMAGE_PIXEL_SIZE_CACHE[image_path] = im.size  # (width, height)
    return _IMAGE_PIXEL_SIZE_CACHE[image_path]


def get_image_norm_size(win, image_path, target_size=IMAGE_TARGET_SIZE):
    """
    Compute an ImageStim `size` in norm units that preserves the image's
    true aspect ratio on screen, regardless of the window's pixel aspect
    ratio.

    In 'norm' units, an extent of 1.0 on x spans the full window WIDTH in
    pixels, while an extent of 1.0 on y spans the full window HEIGHT in
    pixels. These are independent scales, so passing the same number for
    both x and y size (e.g. size=(0.6, 0.6)) only looks square when the
    window itself is square. On a non-square window it stretches the
    image. This function corrects for that by scaling the shorter norm
    dimension using the ratio of window width to window height.
    """
    img_w_px, img_h_px = get_native_pixel_size(image_path)
    win_w_px, win_h_px = win.size

    img_aspect = img_w_px / img_h_px  # >1 = wider than tall

    if img_aspect >= 1.0:
        # Wider than tall (or square): fix the norm width, derive height.
        norm_w = target_size
        # Convert: height_px = width_px / img_aspect
        # norm_h * win_h_px = (norm_w * win_w_px) / img_aspect
        norm_h = (norm_w * win_w_px) / (img_aspect * win_h_px)
    else:
        # Taller than wide: fix the norm height, derive width.
        norm_h = target_size
        norm_w = (norm_h * win_h_px * img_aspect) / win_w_px

    return (norm_w, norm_h)


def show_instructions(win, text):
    instr = visual.TextStim(
        win, text=text, color="black", wrapWidth=1.4, height=0.045
    )
    instr.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", QUIT_KEY])
    check_quit(keys)


def get_rating(win, image_path, prompt_text="How dominant does this face look? (1 = very submissive, 5 = very dominant)"):
    """Display a single image plus a rating prompt, wait for a 1-5 key
    press, and return (rating:int, rt:float)."""
    image_stim = visual.ImageStim(
        win, image=image_path, size=get_image_norm_size(win, image_path)
    )
    prompt_stim = visual.TextStim(
        win, text=prompt_text, pos=(0, -0.75), color="black", height=0.04, wrapWidth=1.6
    )

    event.clearEvents()
    clock = core.Clock()

    image_stim.draw()
    prompt_stim.draw()
    win.flip()
    clock.reset()

    keys = event.waitKeys(keyList=RATING_KEYS + [QUIT_KEY], timeStamped=clock)
    check_quit([k for k, _ in keys])

    key, rt = keys[0]
    rating = int(key)
    return rating, rt


def show_fixation(win, duration):
    fix = visual.TextStim(win, text="+", color="black", height=0.08)
    fix.draw()
    win.flip()
    core.wait(duration)


# ----------------------------------------------------------------------
# EXPERIMENT 1 & 2 (shared logic: simple rating task)
# ----------------------------------------------------------------------

def run_rating_experiment(win, participant_id, stimuli_folder, repeats_per_image,
                           output_filename_suffix, instructions_text):
    """
    Generic rating-task runner used for both Experiment 1 (original images,
    2 repeats each) and Experiment 2 (synthetic images, >=10 repeats each).

    Saves one CSV row per image, with one rating column per repeat, in the
    format: filename, rating_rep1, rating_rep2, ...
    """
    ensure_data_dir()

    image_paths = load_image_list(stimuli_folder)

    # Build the full trial list: each image repeated `repeats_per_image` times
    trial_list = []
    for path in image_paths:
        for _ in range(repeats_per_image):
            trial_list.append(path)
    random.shuffle(trial_list)

    show_instructions(win, instructions_text)

    # ratings_by_image[filename] = list of ratings collected, in order obtained
    ratings_by_image = {os.path.basename(p): [] for p in image_paths}
    trial_log = []  # for a full trial-by-trial log (filename, rating, rt, trial_index)

    for trial_index, image_path in enumerate(trial_list):
        filename = os.path.basename(image_path)
        rating, rt = get_rating(win, image_path)
        ratings_by_image[filename].append(rating)
        trial_log.append((trial_index, filename, rating, rt))

        # brief inter-trial blank
        win.flip()
        core.wait(0.2)

    # ---- Save wide-format CSV: filename, rating_1, rating_2, ..., rating_N ----
    max_reps = max(len(v) for v in ratings_by_image.values())
    wide_path = os.path.join(
        DATA_DIR, f"{participant_id}_{output_filename_suffix}.csv"
    )
    with open(wide_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"rating_{i+1}" for i in range(max_reps)]
        writer.writerow(header)
        for filename, ratings in ratings_by_image.items():
            row = [filename] + ratings
            writer.writerow(row)

    # ---- Save a full trial-by-trial long-format log (useful for RT analysis) ----
    long_path = os.path.join(
        DATA_DIR, f"{participant_id}_{output_filename_suffix}_trials.csv"
    )
    with open(long_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_index", "filename", "rating", "rt_seconds"])
        for row in trial_log:
            writer.writerow(row)

    return wide_path, long_path


# ----------------------------------------------------------------------
# EXPERIMENT 3 (adaptation after-effect)
# ----------------------------------------------------------------------

def run_adaptation_experiment(win, participant_id):
    """
    Experiment 3: adaptation after-effect.

    Expects:
      - STIMULI_DIR_EXP3_ADAPT to contain exactly 2 images: the two endpoint
        (most submissive / most dominant) synthetic faces.
      - STIMULI_DIR_EXP3_TEST to contain exactly 3 near-neutral test images.

    Each of the 2 adapting stimuli is paired with each of the 3 test stimuli,
    giving 6 conditions. You may want to repeat this block multiple times
    per participant; edit N_REPEATS_PER_CONDITION below if so.
    """
    ensure_data_dir()

    N_REPEATS_PER_CONDITION = 5  # increase if you want multiple trials per condition

    adapt_paths = load_image_list(STIMULI_DIR_EXP3_ADAPT)
    test_paths = load_image_list(STIMULI_DIR_EXP3_TEST)

    if len(adapt_paths) != 2:
        raise ValueError(
            f"Expected exactly 2 adapting images in '{STIMULI_DIR_EXP3_ADAPT}', "
            f"found {len(adapt_paths)}."
        )
    if len(test_paths) != 3:
        raise ValueError(
            f"Expected exactly 3 test images in '{STIMULI_DIR_EXP3_TEST}', "
            f"found {len(test_paths)}."
        )

    instructions = (
        "Experiment 3\n\n"
        "You will first view a face for about 25 seconds. Keep looking at "
        "the central cross while the face is shown.\n\n"
        "Immediately afterwards, a second face will appear very briefly. "
        "Rate that second face on the same 1-5 dominance scale as before, "
        "as quickly as you can.\n\n"
        "Press SPACE to begin."
    )
    show_instructions(win, instructions)

    # Block by adapting stimulus. Adaptation carries over between trials, so
    # interleaving the two adaptors would cancel the effect. Block order is
    # counterbalanced across participants.
    blocks = list(adapt_paths)
    if str(participant_id)[-1] in "13579":
        blocks.reverse()

    conditions = []
    for block_i, adapt_path in enumerate(blocks):
        block_trials = []
        for test_path in test_paths:
            for _ in range(N_REPEATS_PER_CONDITION):
                block_trials.append((adapt_path, test_path, block_i + 1))
        random.shuffle(block_trials)
        conditions.extend(block_trials)

    trial_log = []

    for trial_index, (adapt_path, test_path, block_no) in enumerate(conditions):
        if trial_index == len(conditions) // 2:
            show_instructions(win, "Halfway. Take a break of a few minutes,\n"
                                   "then press SPACE to continue.")
        # Fixation before adaptation
        show_fixation(win, FIXATION_DURATION)

        # Adapting stimulus with central fixation cross overlaid
        adapt_stim = visual.ImageStim(
            win, image=adapt_path, size=get_image_norm_size(win, adapt_path)
        )

        adapt_stim.draw()
        win.flip()
        core.wait(ADAPT_DURATION)

        # Brief test stimulus
        test_stim = visual.ImageStim(
            win, image=test_path, size=get_image_norm_size(win, test_path)
        )
        test_stim.draw()
        win.flip()
        core.wait(TEST_DURATION)

        # Blank screen, then prompt for rating (response window is untimed,
        # and the test image is NOT shown again during the response phase)
        win.flip()
        rating, rt = get_rating_no_image(win)

        trial_log.append((
            trial_index,
            os.path.basename(adapt_path),
            os.path.basename(test_path),
            rating,
            rt,
        ))

        win.flip()
        core.wait(0.3)

    # ---- Save results ----
    out_path = os.path.join(DATA_DIR, f"{participant_id}_exp3_aftereffect.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_index", "adapt_filename", "test_filename", "rating", "rt_seconds"])
        for row in trial_log:
            writer.writerow(row)

    return out_path


def get_rating_no_image(win, prompt_text="How dominant did that face look? (1 = very submissive, 5 = very dominant)"):
    """Same as get_rating(), but shows only the rating prompt (no image),
    used for the post-test-stimulus response in Experiment 3."""
    prompt_stim = visual.TextStim(
        win, text=prompt_text, pos=(0, 0), color="black", height=0.05, wrapWidth=1.6
    )
    event.clearEvents()
    clock = core.Clock()
    prompt_stim.draw()
    win.flip()
    clock.reset()
    keys = event.waitKeys(keyList=RATING_KEYS + [QUIT_KEY], timeStamped=clock)
    check_quit([k for k, _ in keys])
    key, rt = keys[0]
    return int(key), rt


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    info = get_participant_info()
    participant_id = info["Student ID"]
    experiment_choice = info["Experiment (1/2/3)"]

    win = visual.Window(
        size=(1200, 800),
        color="white",
        units="norm",
        fullscr=False,  # set True for actual data collection
        allowGUI=True,
    )

    try:
        if experiment_choice == "1":
            instructions = (
                "Experiment 1\n\n"
                "You will see a series of face images, one at a time.\n"
                "For each face, rate how dominant it looks, from 1 (very "
                "submissive) to 5 (very dominant), using the number keys.\n\n"
                "Each image will be shown twice across the experiment.\n\n"
                "Press SPACE to begin."
            )
            wide_path, long_path = run_rating_experiment(
                win,
                participant_id,
                STIMULI_DIR_EXP1,
                EXP1_REPEATS_PER_IMAGE,
                output_filename_suffix="exp1",
                instructions_text=instructions,
            )
            print(f"Saved: {wide_path}\nSaved: {long_path}")

        elif experiment_choice == "2":
            instructions = (
                "Experiment 2\n\n"
                "You will see a series of face images, one at a time.\n"
                "For each face, rate how dominant it looks, from 1 (very "
                "submissive) to 5 (very dominant), using the number keys.\n\n"
                "Press SPACE to begin."
            )
            wide_path, long_path = run_rating_experiment(
                win,
                participant_id,
                STIMULI_DIR_EXP2,
                EXP2_MIN_REPEATS_PER_IMAGE,
                output_filename_suffix="exp2",
                instructions_text=instructions,
            )
            print(f"Saved: {wide_path}\nSaved: {long_path}")

        elif experiment_choice == "3":
            out_path = run_adaptation_experiment(win, participant_id)
            print(f"Saved: {out_path}")

    finally:
        win.close()
        core.quit()


if __name__ == "__main__":
    main()