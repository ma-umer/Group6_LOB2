# 05 – SVM classifier

`svm_model.py` trains RBF-kernel SVMs that separate signal-peptide proteins from negatives, using features computed from the N-terminal region.

```
python 05_svm/svm_model.py      # paths default to the project layout; takes about 3–4 minutes
```

## Features (39)
| Group | Region | Features |
|---|---|---|
| Amino acid composition | residues 1–40 | 20 frequencies |
| Kyte–Doolittle hydrophobicity (window 7) | 1–40 | mean, max, position of max |
| Aliphatic index (window 7) | 1–40 | mean, max, position of max |
| TM-helix propensity (window 7, triangular) | 1–80 | mean, max, position of max |
| α-helix / β-sheet propensity, Chou–Fasman (window 9, triangular) | 1–40 | mean, max, position of max (each) |
| Net charge (window 3) | 1–20 | max, position, min, position |

Scales come from [ProtScale](https://web.expasy.org/protscale/).

## Procedure
1. **5-fold rotating CV.** This is the same scheme as the von Heijne model: test = fold *i*, validation = fold *i+1*, training = the other three folds.
2. **All-features model.** A grid search over C ∈ {0.1, 1, 10, 100} and γ ∈ {scale, 0.01, 0.1, 1} picks the pair with the best validation MCC. Features are scaled with StandardScaler inside the pipeline.
3. **Selected-features model.** A Random Forest (400 trees, balanced class weights, training folds only) ranks the features by Gini importance. The top *k* features (k = 2…39) are scored on the validation MCC, the best *k* is kept, and C and γ are tuned again on that subset.
4. **Benchmark.** The CV round with the best test MCC of the selected-features model (run 1, MCC 0.885) is refit on its training and validation folds (2, 3, 4, 5). Both of its models, all features and selected features, are then evaluated on the benchmark set.

