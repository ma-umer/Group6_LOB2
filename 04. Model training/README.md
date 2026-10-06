# 04 · Model Training

Laboratory of Bioinformatics 2 · Group 6 · University of Bologna

In this stage we build the two signal-peptide (SP) predictors and tune them with **5-fold cross-validation** on the training set from [`02. Data Preparation`](../02.%20Data%20Preparation/cross_validation). The held-out benchmark set is **not touched here**; it is used only once, in [`05. Model Evaluation`](../05.%20Model%20Evaluation).

| Method | Type | Input | Output |
| --- | --- | --- | --- |
| **von Heijne** | Position-specific weight matrix (PSWM) | 15-residue window around the cleavage site | SP score per protein + threshold |
| **SVM** | Support Vector Machine (RBF kernel) | Features of the N-terminal region | SP / no-SP prediction |

---

## Folder contents

<!-- TODO: replace with the real file list -->

| File / folder | What it does |
| --- | --- |
| `TODO_von_heijne.py` | Builds the PSWM, scores sequences, selects the threshold by cross-validation |
| `TODO_feature_extraction.py` | Computes the SVM feature vectors |
| `TODO_svm_training.py` | Grid search and cross-validation of the SVM, trains the final model |
| `TODO_results/` | Cross-validation tables, selected parameters, saved models |

---

## Input data

Both methods use the files in `02. Data Preparation/cross_validation/`:

- `pos_train.tsv`, `neg_train.tsv`: training proteins, with a `fold` column (1–5)
- sequences from the representative FASTA files of the MMseqs2 clustering

| Set | Positives | Negatives | Total |
| --- | --- | --- | --- |
| Training | 876 | 7,265 | 8,141 |
| ↳ each fold | ≈ 175 | ≈ 1,453 | ≈ 1,628 |

Only the first **90 residues** of each protein are used, since SPs are short (median 22 aa; see `03. Data Analysis`).

---

## Cross-validation scheme

For each of the 5 rounds, the folds play three roles:

| Role | Folds | Used for |
| --- | --- | --- |
| Training | 3 | learning the model (PSWM or SVM) |
| Validation | 1 | selecting the threshold / hyperparameters |
| Testing | 1 | unbiased estimate of performance for that round |

<!-- TODO: confirm this matches your implementation (3 train / 1 validation / 1 test), or describe your actual scheme -->

Metrics are reported as mean ± standard deviation over the 5 test folds. The main metric is the **Matthews correlation coefficient (MCC)**, because the classes are imbalanced (≈ 1 positive : 8 negatives).

---

## 1. von Heijne method

### How it works

1. **Collect motifs.** For every positive training protein, extract the 15 residues from position −13 to +2 around the annotated cleavage site.
2. **Count matrix.** Count how often each of the 20 amino acids appears at each of the 15 positions. <!-- TODO: pseudocount? -->
3. **Weights.** Convert counts into frequencies, then into log-odds scores relative to the background amino-acid frequencies <!-- TODO: SwissProt background? --> :

   `W(a, i) = log2( f(a, i) / b(a) )`

   where `f(a, i)` is the frequency of amino acid `a` at position `i` and `b(a)` its background frequency.
4. **Score a protein.** Slide the 15-residue window along the first 90 residues. The window score is the sum of `W` over its positions, and the **highest window score** is the protein's score.
5. **Classify.** A protein is predicted to have an SP if its score is above a threshold, chosen on the validation fold to maximise MCC.

### Results (cross-validation)

<!-- TODO: fill in -->

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| MCC | TODO |
| Threshold | TODO |

---

## 2. SVM

### Features

<!-- TODO: list the real features -->

The N-terminal region of each protein is converted into a numerical vector. Our analysis in stage 03 showed that SPs are hydrophobic and poor in charged residues, so the features are built around:

- amino-acid composition (TODO: which region, how many residues)
- hydrophobicity (TODO: which scale, e.g. Kyte–Doolittle)
- TODO: charge, other physicochemical scales, window-based features

Features are scaled before training (TODO: StandardScaler / other).

### Hyperparameter search

| Parameter | Values tried |
| --- | --- |
| Kernel | TODO (e.g. `rbf`) |
| `C` | TODO |
| `gamma` | TODO |

The best parameters of each round were chosen by validation MCC.

| Round | Kernel | C | gamma | Validation MCC |
| --- | --- | --- | --- | --- |
| 1 | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO | TODO |

**Final model:** TODO (chosen kernel, `C`, `gamma`; trained on the full training set; saved as `TODO.pkl`).

### Results (cross-validation)

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| MCC | TODO |

---

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib

# von Heijne
python "04. Model training/TODO_von_heijne.py"

# SVM
python "04. Model training/TODO_feature_extraction.py"
python "04. Model training/TODO_svm_training.py"
```

All random operations use seed **42**.

---

**Next step:** [`05. Model Evaluation`](../05.%20Model%20Evaluation): final test of both methods on the benchmark set.# 04 · Model Training

Laboratory of Bioinformatics 2 · Group 6 · University of Bologna

In this stage we build the two signal-peptide (SP) predictors and tune them with **5-fold cross-validation** on the training set from [`02. Data Preparation`](../02.%20Data%20Preparation/cross_validation). The held-out benchmark set is **not touched here**; it is used only once, in [`05. Model Evaluation`](../05.%20Model%20Evaluation).

| Method | Type | Input | Output |
| --- | --- | --- | --- |
| **von Heijne** | Position-specific weight matrix (PSWM) | 15-residue window around the cleavage site | SP score per protein + threshold |
| **SVM** | Support Vector Machine (RBF kernel) | Features of the N-terminal region | SP / no-SP prediction |

---

## Folder contents

<!-- TODO: replace with the real file list -->

| File / folder | What it does |
| --- | --- |
| `TODO_von_heijne.py` | Builds the PSWM, scores sequences, selects the threshold by cross-validation |
| `TODO_feature_extraction.py` | Computes the SVM feature vectors |
| `TODO_svm_training.py` | Grid search and cross-validation of the SVM, trains the final model |
| `TODO_results/` | Cross-validation tables, selected parameters, saved models |

---

## Input data

Both methods use the files in `02. Data Preparation/cross_validation/`:

- `pos_train.tsv`, `neg_train.tsv`: training proteins, with a `fold` column (1–5)
- sequences from the representative FASTA files of the MMseqs2 clustering

| Set | Positives | Negatives | Total |
| --- | --- | --- | --- |
| Training | 876 | 7,265 | 8,141 |
| ↳ each fold | ≈ 175 | ≈ 1,453 | ≈ 1,628 |

Only the first **90 residues** of each protein are used, since SPs are short (median 22 aa; see `03. Data Analysis`).

---

## Cross-validation scheme

For each of the 5 rounds, the folds play three roles:

| Role | Folds | Used for |
| --- | --- | --- |
| Training | 3 | learning the model (PSWM or SVM) |
| Validation | 1 | selecting the threshold / hyperparameters |
| Testing | 1 | unbiased estimate of performance for that round |

<!-- TODO: confirm this matches your implementation (3 train / 1 validation / 1 test), or describe your actual scheme -->

Metrics are reported as mean ± standard deviation over the 5 test folds. The main metric is the **Matthews correlation coefficient (MCC)**, because the classes are imbalanced (≈ 1 positive : 8 negatives).

---

## 1. von Heijne method

### How it works

1. **Collect motifs.** For every positive training protein, extract the 15 residues from position −13 to +2 around the annotated cleavage site.
2. **Count matrix.** Count how often each of the 20 amino acids appears at each of the 15 positions. <!-- TODO: pseudocount? -->
3. **Weights.** Convert counts into frequencies, then into log-odds scores relative to the background amino-acid frequencies <!-- TODO: SwissProt background? --> :

   `W(a, i) = log2( f(a, i) / b(a) )`

   where `f(a, i)` is the frequency of amino acid `a` at position `i` and `b(a)` its background frequency.
4. **Score a protein.** Slide the 15-residue window along the first 90 residues. The window score is the sum of `W` over its positions, and the **highest window score** is the protein's score.
5. **Classify.** A protein is predicted to have an SP if its score is above a threshold, chosen on the validation fold to maximise MCC.

### Results (cross-validation)

<!-- TODO: fill in -->

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| MCC | TODO |
| Threshold | TODO |

---

## 2. SVM

### Features

<!-- TODO: list the real features -->

The N-terminal region of each protein is converted into a numerical vector. Our analysis in stage 03 showed that SPs are hydrophobic and poor in charged residues, so the features are built around:

- amino-acid composition (TODO: which region, how many residues)
- hydrophobicity (TODO: which scale, e.g. Kyte–Doolittle)
- TODO: charge, other physicochemical scales, window-based features

Features are scaled before training (TODO: StandardScaler / other).

### Hyperparameter search

| Parameter | Values tried |
| --- | --- |
| Kernel | TODO (e.g. `rbf`) |
| `C` | TODO |
| `gamma` | TODO |

The best parameters of each round were chosen by validation MCC.

| Round | Kernel | C | gamma | Validation MCC |
| --- | --- | --- | --- | --- |
| 1 | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO | TODO |

**Final model:** TODO (chosen kernel, `C`, `gamma`; trained on the full training set; saved as `TODO.pkl`).

### Results (cross-validation)

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| MCC | TODO |

---

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib

# von Heijne
python "04. Model training/TODO_von_heijne.py"

# SVM
python "04. Model training/TODO_feature_extraction.py"
python "04. Model training/TODO_svm_training.py"
```

All random operations use seed **42**.

---

**Next step:** [`05. Model Evaluation`](../05.%20Model%20Evaluation): final test of both methods on the benchmark set.# 04. MODEL TRAINING

