# Feature-Enhanced Buzzer for QA

This repository extends a QA “Buzzer” system with new **feature engineering** techniques, improving how the buzzer decides whether to trust a guess. Below is a brief overview of the features added by me and instructions on how to integrate additional features and run the code.

---

## Environment Setup

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---
   
## Features Added

My feature engineering focuses on signals that help a **logistic regression** buzzer determine correctness more reliably:

1. **Length Feature**  
   - **Guess:** Logarithms of (a) number of characters and (b) number of words.  
   - **Run (Question Snippet):** Logarithms of (a) number of characters and (b) number of words.  
   - **Why it Helps:** Longer guesses often appear when the system is more certain. Meanwhile, the question snippet’s length can indicate how far into the question we are; correct guesses typically surface later with more context.

2. **Guess History Feature**  
   - **Top Guess Count:** Tracks how many times the current guess (for the same question) appears if it’s a “top” repeated guess.  
   - **Number of Different Guesses:** Counts how many unique guesses have been made so far (could signal guesser uncertainty).  
   - **Total Guesses:** Total guesses made for this question so far (more runs might lead to higher chance of correct guess).  

3. **Last Guess Feature**  
   - A boolean indicator (1 or 0) if the current guess was seen in the last few runs (e.g., 4 runs ago).  
   - If it reappears frequently in recent runs, it may indicate guesser stability.

4. **Search (Common Word) Feature**  
   - Measures overlap of valid tokens (non-stopword, non-punctuation) between the guess and the question snippet.  
   - The ratio of shared words to total guess words is fed to the buzzer (logged for normalization).  
   - High overlap suggests a more relevant or correct guess.

5. **Dictionary Feature**  
   - **Guess Dictionary Ratio:** Fraction of guess words recognized by the NLTK English corpus.  
   - **Run Dictionary Ratio:** Fraction of question-snippet words recognized by the NLTK corpus.  
   - This helps the classifier identify whether the guess has many “standard” words vs. rare/proper nouns.

### Observed Improvements

- **Accuracy** rose from ~0.73 to ~0.77  
- **Buzz Ratio** improved from ~0.32 to ~0.3577  
- **“Best” Score** (correct guess + correct buzz) increased from ~0.40 to ~0.4387  

---

## How to Add New Features

1. **Create a Subclass in `features.py`**  
   - Inherit from the base `Feature` class:
     ```bash
     class MyNewFeature(Feature):
         def __init__(self, name):
             super().__init__(name)
         
         def __call__(self, question, run, guess, guess_history, other_guesses=None):
             # Compute your feature values here
             yield ("my_feature_name", value)
     ```
   - Return any number of `(feature_name, value)` pairs via `yield`.

2. **Integrate Your Feature in `parameters.py`**  
   - Locate `load_buzzer(flags, guesser_params, load=False)`.
   - Under the section where features are added, reference your new feature class:
     ```bash
     if ff == "MyFeature":
         from features import MyNewFeature
         feature = MyNewFeature(ff)
         buzzer.add_feature(feature)
     ```
   - Then include `"MyFeature"` in the `features` list or specify it via command-line (e.g., `--features MyFeature`).

3. **Retrain the Buzzer**  
   - Run the training command (see below). The newly added feature will be included in the logistic regression model.

---

## How to Run the Code

1. **Install Dependencies**  
   - Ensure you have a Python 3 environment (optionally a virtualenv).  
   - Install required libraries (e.g., `nltk`, `scikit-learn`, `tqdm`, `numpy`, etc.).

2. **Prepare/Load Guesser Cache** (if using GPT-based guesses)  
   ```bash
   python gpr_guesser.py --fold=buzztrain
   ```
   This loads or creates a cache of guesses for the buzztrain dataset.
3. **Train the Buzzer**
   ```bash
   python buzzer.py \
    --guesser_type=gpr \
    --gpr_guesser_filename=../models/buzztrain_gpr_cache \
    --questions=../data/qanta.buzztrain.json.gz \
    --buzzer_guessers gpr \
    --logistic_buzzer_filename=models/my_buzzer_model \
    --features Length GuessHistoryFeature ...
   ```
   Adjust the feature list to include whichever features you want.
4. **Evaluate the Model**
   ```bash
    python eval.py \
    --guesser_type=gpr \
    --gpr_guesser_filename=../models/buzzdev_gpr_cache \
    --questions=../data/qanta.buzzdev.json.gz \
    --buzzer_guessers gpr \
    --logistic_buzzer_filename=models/my_buzzer_model \
    --features Length GuessHistoryFeature ...
   ```
   Prints out performance metrics, including best/timid/aggressive/waiting outcomes, accuracy, buzz ratio, and feature weights.

---

### Note
- The features detailed above significantly improved the Buzzer’s performance. You can tweak or combine them further, or introduce entirely new concepts for even better results.
- Before running the scripts, ensure that all required files are located in relative paths consistent with the command-line arguments

### Acknowledgement

I extend my sincere gratitude to Professor Jordan Boyd-Graber for providing the foundational project files and resources for this CMSC723 assignment. His work and support have been instrumental in enabling the feature engineering and development efforts showcased in this repository.
