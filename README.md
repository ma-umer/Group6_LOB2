# Prediction of Secretory Signal Peptides in Eukaryotic Proteins

### Laboratory of Bioinformatics 2 · Group 6 · University of Bologna · A.Y. 2026–2027

**Course:** Laboratory of Bioinformatics 2, MSc in Bioinformatics, Alma Mater Studiorum – Università di Bologna
**Instructor:** Prof. Castrense Savojardo
**Authors:** <!-- TODO: list group members (name + GitHub handle) -->

> **Status:** work in progress. Stages 01–03 are complete; stages 04–05 are being finalised.

> ⚠️ **Disclaimer.** This is a student project, written for learning and educational purposes. Expect experiments, rough edges and code that could be improved.

---

## Table of contents

1. [Abstract](#abstract)
2. [Repository structure](#repository-structure)
3. [Quick start](#quick-start)
4. [Pipeline](#pipeline)
   - [01 · Data collection](#01--data-collection)
   - [02 · Data preparation](#02--data-preparation)
   - [03 · Data analysis](#03--data-analysis)
   - [04 · Model training](#04--model-training)
   - [05 · Model evaluation](#05--model-evaluation)
5. [Reproducibility](#reproducibility)
6. [References](#references)

---

## Abstract

Many proteins are destined to leave the cell. To enter the secretory pathway they carry a short N-terminal tag, the **signal peptide (SP)**, which is recognised and cleaved off after translocation. Predicting whether a protein carries an SP is a basic step in functional annotation and subcellular-localisation prediction.

In this project we build and compare two predictors that classify eukaryotic protein sequences as **with SP** or **without SP**:

| Method | Idea |
| ------ | ---- |
| **von Heijne** | A position-specific weight matrix (PSWM) learned from the 15-residue window around the SP cleavage site (−13 … +2). Each protein is scored with a sliding window and classified with an optimised threshold. |
| **SVM** | A Support Vector Machine trained on features computed from the N-terminal region of each protein. |

Both methods are tuned with 5-fold cross-validation and finally compared on a held-out benchmark set of proteins they have never seen.

---

## Repository structure

```
Group6_LOB2/
├── 01. Data Collection/     UniProt queries, download and filtering scripts, positive/negative FASTA + TSV
├── 02. Data Preparation/    MMseqs2 clustering, training/benchmark split, 5 cross-validation folds
├── 03. Data Analysis/       Exploratory analysis, plots and sequence logos
├── 04. Model training/      von Heijne PSWM and SVM training
├── 05. Model Evaluation/    Cross-validation and benchmark evaluation, method comparison
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/ma-umer/Group6_LOB2.git
cd Group6_LOB2

# Python dependencies
pip install pandas numpy matplotlib seaborn scikit-learn requests

# MMseqs2 (needed only to re-run the clustering step)
# https://github.com/soedinglab/MMseqs2
```

Run the stages in order:

```bash
python "01. Data Collection/data_collection.py"
python "02. Data Preparation/prepare_datasets.py"
python "03. Data Analysis/data_analysis.py"
# 04 and 05: see the respective folders
```

<!-- TODO: add exact Python version and a requirements.txt / environment.yml -->

---

## Pipeline

### 01 · Data collection

Proteins were downloaded from **UniProtKB** through the REST API (JSON). Script: [`data_collection.py`](01.%20Data%20Collection/data_collection.py)

**Common criteria (both sets):** reviewed (Swiss-Prot), eukaryotic (`taxonomy_id:2759`), non-fragment, length ≥ 40, evidence at protein level.

| | Positive set (with SP) | Negative set (without SP) |
| --- | --- | --- |
| **Specific criteria** | SP annotated with experimental evidence (`ft_signal_exp:*`) | No SP annotated at any evidence level (`NOT ft_signal:*`); experimental subcellular location in cytosol, nucleus, mitochondrion, plastid, peroxisome or cell membrane |
| **Extra filters (in script)** | SP evidence code ECO:0000269; SP length ≥ 14 residues | – |
| **UniProt results** | 2,972 | 20,975 |
| **After filtering** | 2,953 | 20,975 (2,531 with a transmembrane helix near the N-terminus) |
| **Files** | [`positive.fasta`](01.%20Data%20Collection/positives/positive.fasta), [`positive.tsv`](01.%20Data%20Collection/positives/positive.tsv) | [`negative.fasta`](01.%20Data%20Collection/negatives/negative.fasta), [`negative.tsv`](01.%20Data%20Collection/negatives/negative.tsv) |

<details>
<summary>UniProt queries</summary>

**Positive**

```
(fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)
```

**Negative**

```
(reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (existence:1)
NOT (ft_signal:*)
AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173)
  OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0039))
```

Subcellular-location terms: SL-0091 cytosol, SL-0191 nucleus, SL-0173 mitochondrion, SL-0209 plastid, SL-0204 peroxisome, SL-0039 cell membrane.

</details>

---

### 02 · Data preparation

Script: [`prepare_datasets.py`](02.%20Data%20Preparation/prepare_datasets.py) · Tool: [MMseqs2](https://github.com/soedinglab/MMseqs2)

**Step 1 – Redundancy reduction.** Near-identical proteins (e.g. orthologs in human and mouse) would leak information between training and test data and inflate the results. Positives and negatives were clustered **separately** and one representative per cluster was kept:

```bash
mmseqs easy-cluster positive.fasta cluster-results tmp \
    --min-seq-id 0.3 -c 0.4 --cov-mode 0 --cluster-mode 1
```

| Parameter | Value | Meaning |
| --- | --- | --- |
| `--min-seq-id` | 0.3 | ≥ 30 % sequence identity |
| `-c` | 0.4 | … over ≥ 40 % of the length |
| `--cov-mode` | 0 | coverage required on both sequences |
| `--cluster-mode` | 1 | connected-component clustering |

| | Before | After (representatives) | Removed |
| --- | --- | --- | --- |
| Positives | 2,946 | **1,095** | ≈ 63 % |
| Negatives | 20,974 | **9,082** | ≈ 57 % |

**Step 2 – Splitting.** Representatives were shuffled and split **80 % / 20 %** into a *training set* and a *benchmark set* (positives and negatives separately, preserving the ≈ 1 : 8 ratio). The training set was divided into **5 folds** for cross-validation. Random seed: **42**.

| Set | Positives | Negatives | Total |
| --- | --- | --- | --- |
| **Training** | 876 | 7,265 | 8,141 |
| ↳ each fold (1–5) | ≈ 175 | ≈ 1,453 | ≈ 1,628 |
| **Benchmark** | 219 | 1,817 | 2,036 |

The final datasets used by all later stages are in [`02. Data Preparation/cross_validation/`](02.%20Data%20Preparation/cross_validation) (`pos_train.tsv`, `neg_train.tsv` with a `fold` column, `pos_bench.tsv`, `neg_bench.tsv`). Each row holds the protein ID, class, organism, kingdom, length, and either the SP cleavage position (positives) or a flag for an N-terminal transmembrane helix (negatives).

---

### 03 · Data analysis

Script: [`data_analysis.py`](03.%20Data%20Analysis/data_analysis.py) · Output: [`plots/`](03.%20Data%20Analysis/plots)

Training and benchmark sets were described separately, to verify that the benchmark is a fair test and to guide feature design.

| Analysis | Main finding |
| --- | --- |
| **Protein length** | Proteins with SP are shorter (peak ≈ 100–250 aa) than those without (peak ≈ 300–500 aa). Our methods only use the first 90 residues, so this does not bias them. |
| **SP length** | Median 22 aa in both sets, most between 15 and 30 aa. |
| **Amino-acid composition** (vs SwissProt) | SPs are enriched in L (≈ 22 % vs 9 %) and A, V, F; strongly depleted in charged/polar D, E, K, N. Hydrophobicity is a strong feature for the SVM. |
| **Taxonomy** | Metazoa ≈ 55 %, Fungi ≈ 25 %, Viridiplantae ≈ 17 %. About three quarters of proteins come from five model organisms (human, *S. cerevisiae*, *A. thaliana*, mouse, *S. pombe*). Positives are mostly animal (≈ 80 %). |
| **Cleavage-site logo** (−13 … +2) | Clear **A‑x‑A** motif at −3 / −1; leucine-rich h-region at −13 … −6; almost no signal at +1 / +2. This is exactly the pattern the von Heijne PSWM captures. |

Training and benchmark distributions are very similar, so the benchmark is representative. The main caveat is the bias towards well-studied organisms.

![Amino-acid composition](03.%20Data%20Analysis/plots/comparative_aa_for_sp.png)
![Sequence logo (training)](03.%20Data%20Analysis/plots/logo_training.png)

---

### 04 · Model training

> 🚧 In progress.

**von Heijne method**

1. Extract the 15-residue window (−13 … +2) around the cleavage site of every training positive.
2. Build the count matrix (20 amino acids × 15 positions, with pseudocounts), convert to frequencies and then to log-odds weights against SwissProt background frequencies.
3. Slide the window over the first 90 residues of each protein; the best-scoring window is the protein's score.
4. Choose the classification threshold on validation folds by maximising MCC.

**SVM**

1. Encode the N-terminal region as numerical features (amino-acid composition and physicochemical scales, with sliding windows).
2. Scale the features and tune the hyperparameters (kernel, `C`, `γ`) by grid search with 5-fold cross-validation.
3. Train the final model on the full training set.

<!-- TODO: fill in the exact features, the hyperparameter grid and the script names once finalised -->

---

### 05 · Model evaluation

> 🚧 In progress.

Both methods are evaluated with 5-fold cross-validation (mean ± SD) and once on the held-out benchmark set. Metrics: accuracy, precision, recall, F1 and Matthews correlation coefficient (MCC), the latter being the main metric because the classes are imbalanced (≈ 1 : 8).

| Method | Accuracy | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- |
| von Heijne (CV) | – | – | – | – | – |
| SVM (CV) | – | – | – | – | – |
| von Heijne (benchmark) | – | – | – | – | – |
| SVM (benchmark) | – | – | – | – | – |

<!-- TODO: fill in results, add confusion matrices, and an error analysis (e.g. false positives with N-terminal transmembrane helices) -->

---

## Reproducibility

- All data were retrieved from UniProtKB; the UniProt release changes over time, so counts may differ slightly. <!-- TODO: add the download date -->
- Clustering uses MMseqs2 with the parameters listed above.
- All random splits use seed **42**.
- Every script writes its output to its own stage folder, so any stage can be re-run independently from the files of the previous one.

---

## References

1. The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* Nucleic Acids Research.
2. Steinegger M. & Söding J. (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology, 35, 1026–1028.
3. von Heijne G. (1986). *A new method for predicting signal sequence cleavage sites.* Nucleic Acids Research, 14(11), 4683–4690.
4. Owji H. et al. (2018). *A comprehensive review of signal peptides: structure, roles, and applications.* European Journal of Cell Biology, 97(6), 422–441.
