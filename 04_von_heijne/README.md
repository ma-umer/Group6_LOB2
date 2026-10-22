# 04 – von Heijne classifier

`von_heijne_method.py` builds a von Heijne-style position-specific weight matrix (PSWM) from the region around the signal peptide cleavage site. It scores each protein's N-terminus with the matrix and classifies it as a signal peptide when the best score passes a threshold.

```
python 04_von_heijne/von_heijne_method.py     # paths default to the project layout
```
Options: `--cv-dir`, `--pos-fasta`, `--neg-fasta`, `--output`.

## Method
1. **Data.** The cross-validation and benchmark TSVs from `02_data_preparation/cross_validation/` are joined to their sequences in the FASTA files from `01_data_collection/`. Each sequence is cut to its first 90 residues.
2. **Motifs.** For every positive in the training folds, the 15-residue window around the cleavage site *c* is taken: positions −13 to +2, i.e. `seq[c-13 : c+2]`.
3. **PSWM.** Residues are counted at each of the 15 positions with a pseudocount of 1 and normalised per column. Each frequency is turned into a log-odds score log₂(f / background) against the SwissProt amino acid background frequencies, giving a 20 × 15 matrix.
4. **Scoring.** A 15-residue window slides along the first 90 residues and the protein's score is the highest window score. Residues outside the 20 standard amino acids score 0.
5. **Threshold.** The threshold is set at the point of the precision–recall curve on the validation fold where F1 is highest.
6. **Cross-validation.** 5-fold rotation: test = fold *i*, validation = fold *i+1*, training = the remaining 3 folds.
7. **Final model.** As in the course slides (*Training the two final predictors*; data setup: subsets 0–3 train, subset 4 validates — folds 1–4 and 5 here):
   1. a PSWM is built on the positives of training folds 1–4;
   2. the threshold is fixed on validation fold 5, at the best-F1 point of its precision–recall curve (**8.63**);
   3. the PSWM is rebuilt on the whole training set (all 5 folds, 880 motifs).
8. **Benchmark.** The final PSWM and threshold are applied once to the benchmark set. The benchmark is never used for training or for any choice.

## The PSWM

![Final PSWM](model_results/pswm_final.png)

*Final PSWM (log₂ odds against the SwissProt background), built on all 5 training folds. Positions are relative to the cleavage site, which is marked by the vertical line between −1 and +1. Red = enriched in signal peptides, blue = depleted, grey ≈ background. The colour scale is clipped at ±3; the printed values are exact.*

The matrix reproduces the known structure of a eukaryotic signal peptide:
- **h-region (positions −13 to −6).** Leu has the highest score (+1.6 to +2.1 at −13 to −7), followed by Cys, Phe, Trp and Val. Charged residues (D, E, K, R) and Asn are strongly penalised (−1.5 to −5.7).
- **c-region and cleavage site.** The highest score in the matrix belongs to Ala at −1 (+2.59). At −3, Cys (+2.18), Ala (+1.80) and Val (+1.63) are favoured, and Cys (+1.51), Gly (+1.43) and Ser (+0.88) are also accepted at −1. This is the (A/V)-X-A rule. Large hydrophobic and aromatic residues are strongly disfavoured at −1 and −3 (Ile −4.16, Val −3.95 at −1; Phe −2.80, Tyr −4.72 at −3).
- **Mature protein (+1, +2).** Pro is favoured at +2 (+1.67), and charged/polar residues are no longer penalised.

## Results
### Cross-validation (test folds)
| Run | Test fold | Val fold | Threshold | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 2 | 9.84 | 0.939 | 0.733 | 0.688 | 0.710 | 0.676 |
| 2 | 2 | 3 | 9.60 | 0.929 | 0.699 | 0.608 | 0.650 | 0.613 |
| 3 | 3 | 4 | 9.20 | 0.939 | 0.702 | 0.761 | 0.730 | 0.697 |
| 4 | 4 | 5 | 8.77 | 0.926 | 0.634 | 0.739 | 0.682 | 0.643 |
| 5 | 5 | 1 | 8.73 | 0.939 | 0.704 | 0.744 | 0.724 | 0.690 |
| **Mean ± SE** | | | 9.23 ± 0.22 | 0.934 ± 0.003 | 0.695 ± 0.016 | 0.708 ± 0.028 | 0.699 ± 0.015 | **0.664 ± 0.016** |

Values are the mean over the 5 test folds ± the standard error (SD/√5), as required by the course. The standard deviations are in `cv_summary.tsv` (MCC SD = 0.035).

Combined confusion matrix over the 5 test folds: TP = 623, FP = 277, FN = 257, TN = 6988.

![Validation precision–recall curves](model_results/pr_curves_all_rounds.png)

*Precision–recall curves on the validation folds. The dot on each curve marks the threshold with the best validation F1, which is then applied to that round's test fold. The curves are close together, but the best-F1 points fall at different places along them (recall 0.68–0.86), so the threshold changes from round to round (8.73–9.84).*

![Pooled ROC, pooled precision–recall and average PSWM](model_results/combined_model_metrics.png)

*Left and centre: ROC and precision–recall curves pooled over the 5 test folds (ROC AUC = 0.96, average precision = 0.78). The high AUC but modest AP reflects the class imbalance (about 1 positive per 8 negatives): the PSWM ranks most positives above most negatives, but at any useful recall a sizeable share of the top-scoring proteins are negatives. The dashed grey lines are a random classifier (diagonal for ROC; the positive fraction, 0.11, for precision–recall). Right: PSWM averaged over the 5 rounds, on the same colour scale as the final matrix and very close to it (the largest differences, up to 0.7, are in rare residues).*

<img src="model_results/confusion_matrix_combined.png" alt="Combined CV confusion matrix" width="420">

*Confusion matrix summed over the 5 test folds, laid out as in the course slides (rows = observed class, columns = predicted class). Each cell is shaded by its share of the observed class (its row), and that percentage is printed under the count, so the small positive class is as easy to read as the negatives.*

### Score distributions

![Score distribution](model_results/score_distribution.png)

*PSWM score of positives and negatives (each class normalised to 1). Left: the 5 CV test folds pooled; the yellow band is the range of the five validation thresholds. Right: the benchmark with the final PSWM and θ = 8.63; negatives with an N-terminal TM helix are drawn separately (black outline). Drawn by `06_comparison/extra_figures.py`; medians in `score_distribution.tsv`.*

The two classes overlap mostly between scores 5 and 10, right where θ sits, which is why the threshold moves between runs and trades precision for recall. Negatives with an N-terminal TM helix score much higher than other negatives (median 6.3 vs 0.9 on the benchmark), so they make up a large part of the tail above θ.

### Final model: threshold
![Final-model precision–recall curve](model_results/pr_curve_final.png)

*Precision–recall curve of validation fold 5, scored by the PSWM built on folds 1–4. The dot is the best-F1 point, which sets the final threshold (8.63; validation precision 0.690, recall 0.773, F1 0.729, MCC 0.696).*

The threshold (8.63) is just below the lowest of the five CV-run thresholds (8.73–9.84) and close to the threshold that the pooled out-of-fold scores of a full 5-fold CV would give (8.71). `benchmark_threshold_check.tsv` reports, for information only, what the benchmark would give with the two other training-set rules: MCC 0.677 with the mean CV-run threshold (9.23) and 0.671 with the out-of-fold threshold (8.71), against 0.672 with the validation-fold threshold. The choice of rule therefore does not drive the result. None of these numbers was used to choose anything.

### Benchmark (221 positives / 1,817 negatives, threshold 8.63)
| TP | FP | FN | TN | Accuracy | Precision | Recall | Specificity | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 173 | 95 | 48 | 1722 | 0.930 | 0.646 | 0.783 | 0.948 | 0.708 | **0.672** |

<img src="model_results/confusion_matrix_benchmark.png" alt="Benchmark confusion matrix" width="420">

*Confusion matrix on the benchmark set (final PSWM, threshold 8.63), in the same layout.*

The false-positive rate is 20.7 % for negatives with an N-terminal transmembrane helix (39 / 188) and 3.4 % for all other negatives (56 / 1,629). 41 % of the 95 false positives (39) are transmembrane proteins, although these proteins are only 10 % of the negatives. A hydrophobic helix near the N-terminus followed by small residues looks like a signal peptide's h-region and cleavage site to the PSWM. The error analysis of the benchmark (false-positive TM share, logos of the missed cleavage sites, kingdoms and species) is in `06_comparison`.

The benchmark MCC (0.672) is in line with the cross-validation estimate (0.664 ± 0.016), so the model generalises without signs of overfitting. The threshold varies between CV rounds (8.73–9.84), which explains the spread in precision and recall there: a higher threshold trades recall for precision.

> **Revision notes.**
> - An earlier version used the CV round with the best *test* MCC for the benchmark, with that round's threshold. That choice used the test folds for model selection, which the course explicitly rules out.
> - The next version set the threshold on the pooled out-of-fold scores of a 5-fold CV. The slides on the final models prescribe a simpler protocol, now followed exactly: PSWM on subsets 0–3, threshold on validation subset 4, PSWM retrained on the whole training set, one test on the benchmark. The out-of-fold rule is kept as a reference value only.
> - All results were recomputed on the current data split (see `02_data_preparation`); the numbers of the earlier versions (benchmark of 220 positives) are not comparable, because the train/benchmark partition changed.
> - Confusion matrices now use the slide layout (rows = observed class).

### Limitations
- The scan covers all of the first 90 residues, as the method prescribes, so the PSWM can fire on a motif-like window far from where a cleavage site could be (see `06_comparison`). Restricting the window to a plausible cleavage range would likely reduce false positives.
- The PSWM scores only the cleavage-site motif. It has no notion of the h-region's position, so N-terminal transmembrane helices remain the main source of false positives.

### Per-round figures

<details>
<summary>PSWM, validation precision–recall curve and test-fold confusion matrix of each CV run (click to expand)</summary>

**PSWM of each run** (built on that run's 3 training folds; same colour scale as the final matrix)

<p>
<img src="model_results/rounds/pswm_round1.png" alt="pswm_round1.png" width="400">
<img src="model_results/rounds/pswm_round2.png" alt="pswm_round2.png" width="400">
<img src="model_results/rounds/pswm_round3.png" alt="pswm_round3.png" width="400">
<img src="model_results/rounds/pswm_round4.png" alt="pswm_round4.png" width="400">
<img src="model_results/rounds/pswm_round5.png" alt="pswm_round5.png" width="400">
</p>

**Validation precision–recall curves** (dot = best-F1 threshold, applied to the test fold)

<p>
<img src="model_results/rounds/pr_round1.png" alt="pr_round1.png" width="320">
<img src="model_results/rounds/pr_round2.png" alt="pr_round2.png" width="320">
<img src="model_results/rounds/pr_round3.png" alt="pr_round3.png" width="320">
<img src="model_results/rounds/pr_round4.png" alt="pr_round4.png" width="320">
<img src="model_results/rounds/pr_round5.png" alt="pr_round5.png" width="320">
</p>

**Test-fold confusion matrices** (rows = observed class, columns = predicted class)

<p>
<img src="model_results/rounds/cm_round1.png" alt="cm_round1.png" width="260">
<img src="model_results/rounds/cm_round2.png" alt="cm_round2.png" width="260">
<img src="model_results/rounds/cm_round3.png" alt="cm_round3.png" width="260">
<img src="model_results/rounds/cm_round4.png" alt="cm_round4.png" width="260">
<img src="model_results/rounds/cm_round5.png" alt="cm_round5.png" width="260">
</p>

</details>

## Figure style
All figures in 04, 05 and 06 share one colour scheme, defined at the top of `von_heijne_method.py` (and copied in `05_svm/svm_model.py`):
- **PSWM heatmaps:** diverging blue → grey → red centred on 0 (blue = depleted, red = enriched), with the colour scale clipped at ±3 log₂ odds.
- **Confusion matrices:** the layout of the course slides (rows = observed class, columns = predicted class), a single blue scale, light to dark.
- **Lines and bars:** a fixed categorical palette that is safe for colour-blind readers. Rounds 1–5 always use the same five colours (blue, orange, green, yellow, pink).

## Outputs (`model_results/`)
| File | Content |
|---|---|
| `cv_runs.tsv`, `cv_summary.tsv` | metrics for each run, and mean / SD / SE |
| `pswm_final.tsv`, `pswm_final.png` | the final PSWM, built on all 5 training folds |
| `final_model.tsv` | the final threshold, how it was chosen (validation fold 5, PSWM from folds 1–4), its validation precision / recall / F1 / MCC, and two reference thresholds |
| `pr_curve_final.png` | validation-fold precision–recall curve used to set the final threshold |
| `benchmark_threshold_check.tsv` | benchmark result with the reference thresholds (reported only, not used for any choice) |
| `benchmark_metrics.tsv` | benchmark confusion matrix, metrics and FPR with / without an N-terminal TM helix |
| `combined_model_metrics.png` | pooled ROC curve, pooled precision–recall curve and average PSWM |
| `pr_curves_all_rounds.png` | validation precision–recall curves with the chosen thresholds |
| `confusion_matrix_combined.png`, `confusion_matrix_benchmark.png` | combined CV and benchmark confusion matrices |
| `benchmark_true_positives.tsv`, `benchmark_false_negatives.tsv` | benchmark positives split by outcome, with their scores |
| `motifs_true_positives.tsv`, `motifs_false_negatives.tsv` | cleavage-site 15-mers of those proteins, e.g. for sequence logos |
| `score_distribution.png`, `score_distribution.tsv` | PSWM score histograms of positives vs negatives (pooled CV test folds and benchmark) with the thresholds, and the median score of each group (drawn by `06_comparison/extra_figures.py`) |
| `rounds/` | PSWM heatmap, validation precision–recall curve and test confusion matrix for each round |
