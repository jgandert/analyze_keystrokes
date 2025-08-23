# Analyze Keystroke Data

The [original dataset](#dataset) used here has approximately 136 million presses and 168 thousand volunteers.

After filtering out suspicious (too fast) and incorrect data (e.g., uppercase letter, but no shift pressed), about 55 million presses and 77 thousand volunteers are left.

## Usage

### Step 1: Filter and convert the dataset (Optional)

1. Download the [136 Million Keystrokes dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/). (1,4 GiB)
2. Extract the downloaded dataset into the `extract_archive_in_here` directory. (16 GiB unzipped)
3. ▶️ Run [filter_and_convert_keystroke_dataset.py](filter_and_convert_keystroke_dataset.py).

---

If you'd like to skip this step, you can download the filtered version (`filtered_events.csv.gz`) from the [releases section](https://github.com/jgandert/analyze_keystrokes/releases) and move it to the `dataset` directory.

### Step 2: Analyze converted dataset and create training dataset

▶️ Run [analyze.py](analyze.py).

If you'd like to use the resulting training dataset to train prediction functions, head over to the [evolve_tap_hold_predictors](https://github.com/jgandert/evolve_tap_hold_predictors) repository.

Evolved functions are used in the [Predictive Tap-Hold community module for QMK](https://github.com/jgandert/qmk_modules/tree/main/predictive_tap_hold).

📊 Open [**analyze.md**](analyze.md) to see the output for the default settings.

# Dataset

The dataset used here can be found at: https://userinterfaces.aalto.fi/136Mkeystrokes/

```
Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta
Observations on Typing from 136 Million Keystrokes. 
In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM, 2018.

@inproceedings{dhakal2018observations,
author = {Dhakal, Vivek and Feit, Anna and Kristensson, Per Ola and Oulasvirta, Antti},
booktitle = {Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI '18)},
title = {{Observations on Typing from 136 Million Keystrokes}},
year = {2018}
publisher = {ACM}
doi = {https://doi.org/10.1145/3173574.3174220}
keywords = {text entry, modern typing behavior, large-scale study}
}
```

You are free to use this data for non-commercial use in your own research or projects with attribution to the authors.

As a result, the same applies to all files in the `/dataset` directory and the GitHub release files that are based on the dataset.