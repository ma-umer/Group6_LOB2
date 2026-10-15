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

## Feature importance and selection

![Top-20 features, round 1](model_results/rounds/top_features_round1.png)

*Random Forest Gini importance of the top 20 features in round 1. The other rounds give almost the same ranking (see `rounds/`).*

The ranking is dominated by the hydrophobic core of the signal peptide. The six most important features are the maximum and mean hydrophobicity, the maximum aliphatic index and the three TM-propensity features. Together they account for about two thirds of the total importance, and the maximum hydrophobicity alone for about a quarter. Next come the *positions* of the most hydrophobic and most aliphatic windows: a signal peptide's h-region sits close to the N-terminus. Composition features rank low; Leu is the most informative of them.

![Validation MCC vs number of features](model_results/mcc_vs_k_all_rounds.png)

*Validation MCC of the SVM as the top-k features are added, one curve per round (same round colours as the von Heijne figures).*

The curves rise steeply up to k ≈ 5, when the hydrophobicity, aliphatic and TM features are in, and then keep climbing slowly as positional, charge and composition features are added. The best *k* is 26, 29, 31, 39 and 37 in rounds 1–5. No round has a clear plateau at a small *k*, so selection removes few features and barely changes performance.

| Run | Test fold | Best *k* | C / γ (all) | C / γ (selected) | Test MCC (all) | Test MCC (selected) |
|---|---|---|---|---|---|---|
| 1 | 1 | 26 | 1 / scale | 1 / scale | 0.873 | **0.885** |
| 2 | 2 | 29 | 1 / scale | 1 / scale | 0.882 | 0.879 |
| 3 | 3 | 31 | 1 / scale | 10 / 0.01 | 0.846 | 0.853 |
| 4 | 4 | 39 | 100 / scale | 100 / scale | 0.822 | 0.822 |
| 5 | 5 | 37 | 100 / scale | 10 / scale | 0.855 | 0.857 |

## Results
Cross-validation (test folds, mean ± SD over 5 runs):

| Model | MCC | Precision | Recall | Accuracy | F1 |
|---|---|---|---|---|---|
| All features | 0.855 ± 0.024 | 0.882 ± 0.032 | 0.860 ± 0.018 | 0.972 ± 0.005 | 0.871 ± 0.021 |
| Selected features | 0.859 ± 0.025 | 0.881 ± 0.032 | 0.868 ± 0.020 | 0.973 ± 0.005 | 0.874 ± 0.022 |
| *von Heijne (for comparison)* | *0.675 ± 0.025* | *0.739* | *0.686* | *0.938* | *0.705* |

<p>
<img src="model_results/confusion_matrix_combined_all.png" alt="Combined CV confusion matrix, all features" width="420">
<img src="model_results/confusion_matrix_combined_selected.png" alt="Combined CV confusion matrix, selected features" width="420">
</p>

*Confusion matrices summed over the 5 test folds: all features (left) and selected features (right). Each cell is shaded by its share of the actual class (its column), and that percentage is printed under the count, so the small positive class is as easy to read as the negatives.*

Benchmark (219 positives / 1817 negatives, refit from run 1):

| Model | TP | FP | FN | TN | MCC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| All features (39) | 190 | 21 | 29 | 1796 | 0.870 | 0.900 | 0.868 | 0.884 |
| Selected features (26) | 184 | 23 | 35 | 1794 | 0.848 | 0.889 | 0.840 | 0.864 |

<p>
<img src="model_results/confusion_matrix_benchmark_all.png" alt="Benchmark confusion matrix, all features" width="420">
<img src="model_results/confusion_matrix_benchmark_selected.png" alt="Benchmark confusion matrix, selected features" width="420">
</p>

*Benchmark confusion matrices: all 39 features (left) and the 26 selected features (right), shaded the same way.*

The benchmark MCC (0.870 for all features) is in line with the cross-validation mean (0.855 ± 0.024), so the model generalises well. Compared with von Heijne on the same benchmark (MCC 0.683), the SVM has about 3.4× fewer false positives (21 vs 72) and about half the false negatives (29 vs 56); see `06_comparison`.

False-positive rate on the benchmark negatives (all-features model): 6.4 % for proteins with an N-terminal transmembrane helix (13 / 204) and 0.5 % for the rest (8 / 1613). 13 of the 21 false positives have an N-terminal TM helix, which the hydrophobicity features can mistake for an h-region.

Feature selection gives no improvement: in CV the two models are within 0.004 MCC of each other, and on the benchmark the all-features model is slightly better (0.870 vs 0.848).

### Limitations
- The benchmark round is chosen by **test** MCC, so the benchmark estimate is slightly optimistic. The same applies to the von Heijne model, so the comparison between the two is still fair.

Figures use the same colour scheme as `04_von_heijne` (see its README, *Figure style*).

## Outputs (`model_results/`)
- `cv_runs.tsv`, `cv_summary.tsv`: per-run and mean/SD/SE metrics for both models
- `selected_features.tsv`: the chosen *k* and feature list for each round
- `benchmark_metrics.tsv`, `benchmark_predictions.tsv`, `benchmark_{true_positives,false_negatives,false_positives}.tsv`
- `confusion_matrix_combined_{all,selected}.png`, `confusion_matrix_benchmark_{all,selected}.png`, `mcc_vs_k_all_rounds.png`
- `svm_{all,selected}_final.pkl.gz`: the refit models (scikit-learn pipelines)
- `training_features.tsv`: the feature matrix for the training set
- `rounds/`: Gini rankings, top-20 feature plots, MCC-vs-k curves, confusion matrices and models for each round
