# Group6_LOB2
> ⚠️ **Disclaimer**
>
> This repository contains students work. It was written by students,
> not by a team of professional researchers locked in a basement with
> unlimited compute.
>
> Expect experiments, mistakes, questionable code, things that probably
> could have been done better, and the occasional piece of code that
> somehow works. The project is primarily for learning and educational
> purposes.

---

Some proteins are made to leave the cell. To get out, they carry a short tag at their front, called a **signal peptide**: a kind of boarding pass that the cell checks and then cuts off. Rude, but efficient.

**Our goal:** teach a computer to look at a protein sequence and tell whether it has that boarding pass.

This is the **Laboratory of Bioinformatics 2** project (University of Bologna, A.Y. 2026–2027) by **Group 6**.

---

## The project in short

We collect thousands of eukaryotic proteins from **UniProt**, some with a signal peptide and some without. We clean the data, explore it, and then build two predictors:

- **von Heijne method:** a classic approach that learns the typical amino-acid pattern around the signal-peptide cut site.
- **SVM (Support Vector Machine):** a machine-learning model trained on features calculated from each protein.

Finally, we test both methods on proteins they have never seen and compare which one works better.

---

**Course:** Laboratory of Bioinformatics 2, MSc in Bioinformatics, University of Bologna
**Instructor:** Prof. Castrense Savojardo

---

# LABORATORY OF BIONFORMATICS 2 PROJECT - BOLOGNA UNIVERSITY

# [01. DATA COLLECTION](https://github.com/ma-umer/Group6_LOB2/tree/main/01.%20Data%20Collection)

## Selection Criteria For The Positive Set of Proteins
The URL for Positive dataset was generated from the UniProtKB Advanced search. It selects: reviewed, eukaryotic, non-fragment proteins, length ≥ 40, evidence at protein level, with a signal peptide supported by experimental evidence.

The remaining criteria cannot be expressed in the query, so we check them with a filter function:
1. the signal peptide has experimental evidence (ECO:0000269)
2. the signal peptide is at least 14 residues long

## Selection Criteria For the Negative Set of Proteins
The URL for negative dataset selects the same common criteria as the positive, and in addition:
* no signal peptide annotated at any evidence level (`NOT ft_signal:*`)
* experimental subcellular location in cytosol (SL-0091), nucleus (SL-0191), mitochondrion (SL-0173), plastid (SL-0209), peroxisome (SL-0204) or cell membrane (SL-0039)

## Positive Query 

### UniProt Query For Positive Proteins
```text
 (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)
```
***UniProtKB 2,972 results***

### Positive query API URL:
```
https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28reviewed%3Atrue%29+AND+%28ft_signal_exp%3A*%29%29&size=500 #
```
***Format:JSON, Compressed:No***



## Negative Query 

### UniProt Query For Negative Proteins
```
(reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (existence:1) NOT (ft_signal:*) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039) OR (cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0173)	
```
***UniProtKB 20,975 results***


### Negative query API URL: 
***Format:JSON, Compressed:No***

```
https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND%28reviewed%3Atrue%29+AND+%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28existence%3A1%29+NOT+%28ft_signal%3A*%29+OR+%28cc_scl_term_exp%3ASL-0191%29+OR+%28cc_scl_term_exp%3ASL-0204%29+OR+%28cc_scl_term_exp%3ASL-0039%29+OR+%28cc_scl_term_exp%3ASL-0091%29+OR+%28cc_scl_term_exp%3ASL-0209%29+OR+%28cc_scl_term_exp%3ASL-0173%29%29&size=500
```
### Data Collection:
*** [data_collection.py](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/data_collection.py)
## DATA SUMMARY
---
|Data Set| Total Proteins|Filtered Proteins|Transmembrane Proteins|Files|
|--------|---------------|-----------|--------|----------------|
|Positive|  2972         |    2953   |    -   | [FASTA](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/positives/positive.fasta)  [TSV](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/positives/positive.tsv) |
|Negative|  20975        |    20975  |   2531 | [FASTA](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/negatives/negative.fasta)  [TSV](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/negatives/negative.tsv) |
---


## [02. DATA PREPARATION](https://github.com/ma-umer/Group6_LOB2/tree/main/02.%20Data%20Preparation)


In [`01. Data Collection`](../01.%20Data%20Collection) we downloaded our proteins from UniProt. Before we can use them to train and test a model, they need some cleaning and organising. This step does two things:

1. **Remove redundancy:** throw out proteins that are too similar to each other.
2. **Split the data:** divide the proteins into a **training set** (to build the models) and a **benchmark set** (to test them at the very end), and divide the training set into **5 folds** for cross-validation.

**Script:** [`prepare_datasets.py`](prepare_datasets.py) | **Clustering tool:** [MMseqs2](https://github.com/soedinglab/MMseqs2)

---

## Step 1: Removing redundancy

### Why?

UniProt contains many proteins that are almost copies of each other, for example the same protein in human and mouse. If one copy ends up in the training set and its "twin" in the test set, the model sees an easy question it has basically already studied. The results then look better than they really are.

To avoid this, we group similar proteins together (**clustering**) and keep only **one protein per group** (the **representative**).

### How?

We used **MMseqs2** with the settings from the course:

| Setting | Value | Meaning |
|---|---|---|
| `--min-seq-id` | 0.3 | Two proteins count as "similar" if they share at least **30% identity** |
| `-c` | 0.4 | ...over at least **40% of their length** |
| `--cov-mode` | 0 | The 40% must hold for **both** proteins |
| `--cluster-mode` | 1 | **Connected-component** clustering (if A is similar to B and B to C, all three end up in one cluster) |

Positives and negatives were clustered **separately**:

```bash
mmseqs easy-cluster positive.fasta cluster-results tmp \
    --min-seq-id 0.3 -c 0.4 --cov-mode 0 --cluster-mode 1
```

(and the same command for `negative.fasta`).

### Result

| | Before clustering | After clustering (representatives) | Removed |
|---|---|---|---|
| Positives (with SP) | 2,946 | **1,095** | about 63% |
| Negatives (without SP) | 20,974 | **9,082** | about 57% |

More than half of the proteins were redundant. What's left is a smaller but much more **diverse** and **honest** dataset.

---

## Step 2: Splitting the data

### Training vs benchmark

The representatives were shuffled and split **80% / 20%**:

- **Training set (80%):** used to build the models and tune their settings.
- **Benchmark set (20%):** kept aside and used **only once, at the very end**, as a final exam on proteins the models have never seen.

Positives and negatives were split **separately**, so both sets keep the same ratio of positives to negatives (about 1 to 8).

### 5 folds for cross-validation

The training set was further divided into **5 equal folds** (numbered 1–5). During cross-validation, each fold takes a turn as the test set while the others are used for training. This gives 5 results instead of 1, so we can see how **stable** a model's performance is.

### Result

| Set | Positives | Negatives | Total |
|---|---|---|---|
| **Training** | 876 | 7,265 | 8,141 |
| ↳ each fold (1–5) | about 175 | 1,453 | about 1,628 |
| **Benchmark** | 219 | 1,817 | 2,036 |

All splits use a fixed random seed (42), so running the script again gives exactly the same sets.

---

## Files in this folder

| File / folder | What it contains |
|---|---|
| [`prepare_datasets.py`](prepare_datasets.py) | Script that reads the representatives and creates the training/benchmark split and the 5 folds |
| [`positive_cluster/`](positive_cluster) | MMseqs2 output for the positives |
| [`negative_cluster/`](negative_cluster) | MMseqs2 output for the negatives |
| `*_cluster/cluster-results_cluster.tsv` | Which protein belongs to which cluster |
| `*_cluster/cluster-results_rep_seq.fasta` | Sequences of the representatives (one per cluster) |
| `*_cluster/rep_*.ids` | List of representative IDs |
| [`cross_validation/`](cross_validation) | **The final datasets used in the rest of the project** |
| `pos_train.tsv`, `neg_train.tsv` | Training proteins, with a `fold` column (1–5) |
| `pos_bench.tsv`, `neg_bench.tsv` | Benchmark proteins |

Each TSV row contains the protein ID, class, organism, kingdom, length, and either the **SP cleavage position** (positives) or whether it has a **transmembrane helix** near the start (negatives).

---

## Summary

- Similar proteins were removed with MMseqs2 (30% identity, 40% coverage), leaving **1,095 positives** and **9,082 negatives**.
- The data were split into **training (80%)** and **benchmark (20%)**, keeping the positive/negative ratio.
- The training set was divided into **5 folds** for cross-validation.

**Next step:** [`03. Data Analysis`](../03.%20Data%20Analysis) — exploring what these datasets look like.

## [03. DATA ANALYSIS](https://github.com/ma-umer/Group6_LOB2/tree/main/03.%20Data%20Analysis)

Before training any prediction method, we first look at our data. This step answers simple questions: How long are our proteins? How long are the signal peptides? Which amino acids do they contain? Which organisms do they come from? And what does the cleavage site look like?

Looking at the data first helps us:

check that the training and benchmark sets are similar (so the final test is fair),
spot biases in the data,
understand which features could help a model recognise a signal peptide (SP).

Script: [data_analysis.py](https://github.com/ma-umer/Group6_LOB2/blob/main/03.%20Data%20Analysis/data_analysis.py)

Output: [Plots](https://github.com/ma-umer/Group6_LOB2/tree/main/03.%20Data%20Analysis/plots)

## The data we analysed

The two datasets come from `02. Data Preparation`. Each one is described
**separately**, as asked in the course.

| Dataset | Proteins with SP (positives) | Proteins without SP (negatives) | Total |
|---|---:|---:|---:|
| Training | 876 | 7,265 | 8,141 |
| Benchmark | 219 | 1,817 | 2,036 |

> **Quick reminder:** a *signal peptide* is a short "address label"
> (about 20–30 amino acids) at the start (N-terminus) of a protein.
> It sends the protein into the secretory pathway and is then cut off at
> the *cleavage site*. It has three parts: a positively charged
> **n-region**, a hydrophobic (water-avoiding) **h-region**, and a
> **c-region** that contains the cleavage site.

## 1. Protein length

![Protein length](plots/protein_length_comparison.png)

**What the plot shows:** how long the proteins are, comparing proteins with an SP (blue) and without an SP (green). Each colour is scaled to its own group, so the two shapes can be compared even though there are many more negatives. Very long proteins (over 3,000 amino acids) are left out to keep the plot readable.

**What we see:**

- Proteins **with** an SP tend to be **shorter**: most are under 500 amino acids, with a peak around 100–250.
- Proteins **without** an SP are spread over longer lengths, with a peak around 300–500.
- The training and benchmark plots look alike.

**Why it matters:** total length is different between the two groups, but our methods only look at the **start of the protein** (the first 90 amino acids), so this difference does not affect them.

---

## 2. Signal-peptide length

![SP length](plots/signal_peptide_graph.png)

**What the plot shows:** how many amino acids long each signal peptide is.

**What we see:**

- Most SPs are **15–30 amino acids** long.
- The **median is 22 amino acids** in both sets.
- A few unusual SPs are longer (up to 83 in training and 64 in benchmark).
- Training and benchmark have almost the same shape.

**Why it matters:** this matches the expected SP length from the lectures (20–30 residues). Because SPs are short, it is enough to search only the **first 90 amino acids** of each protein.

---

## 3. Amino-acid composition: signal peptides vs SwissProt

![Amino-acid composition](plots/comparative_aa_for_sp.png)

**What the plot shows:** the percentage of each amino acid inside signal peptides (training in blue, benchmark in orange), compared with all proteins in SwissProt (green, the "normal" background). The amino acids are grouped by type: apolar, aromatic, polar and charged.

**What we see:**

| Amino acid | In SPs | In SwissProt | Meaning |
|---|---|---|---|
| **L** (leucine) | about 22% | about 9% | Much more common: L forms the **hydrophobic h-region** |
| **A** (alanine) | about 14–15% | about 8% | More common: A is typical of the **cleavage site** |
| **V, F, M** | higher | lower | Hydrophobic residues are enriched (M is high because every protein starts with M) |
| **D, E, K, N** | about 1–2% | about 4–6% | Much rarer: **charged and polar** residues are avoided |

**Why it matters:** SPs have a very clear "chemical fingerprint": **lots of hydrophobic residues, few charged ones**. This tells us that amino-acid composition and hydrophobicity are good features for a classifier such as the SVM.

---

## 4. Taxonomy: which organisms the proteins come from

![Taxonomy training](plots/taxonomic_classification_training.png)
![Taxonomy benchmark](plots/taxonomic_classification_benchmark.png)

**What the plots show:** the left pie chart groups proteins by **kingdom**; the right pie chart shows the most common **species**.

**What we see:**

| Kingdom | Training | Benchmark |
|---|---|---|
| Metazoa (animals) | 55.5% | 55.4% |
| Fungi | 25.9% | 24.6% |
| Viridiplantae (plants) | 16.5% | 17.6% |
| Other | 2.1% | 2.5% |

- About **three quarters** of all proteins come from just five model organisms: human (*Homo sapiens*, about 27%), baker's yeast (*S. cerevisiae*, about 16%), *Arabidopsis thaliana* (about 13%), mouse (*Mus musculus*, about 12%) and fission yeast (*S. pombe*, about 6–7%).
- Training and benchmark have almost the same proportions.

**Why it matters:** the data are **biased toward well-studied organisms**, because these are the proteins with experimental evidence in UniProt. A model trained on them may work less well on rarely studied organisms. Also, most proteins with an SP come from animals (about 80% of positives), so we should keep this in mind when we interpret the results.

---

## 5. Sequence logos of the cleavage site

![Logo training](plots/logo_training.png)
![Logo benchmark](plots/logo_benchmark.png)

**How we built it:** for every protein with an SP, we cut out the **15 amino acids around the cleavage site**: 13 before the cut (positions −13 to −1) and 2 after it (+1 and +2). These windows are saved in `motifs_training.*` and `motifs_benchmark.*`. Lining them up gives a sequence logo.

**How to read a logo:**

- Each column is one position. The dashed line is where the SP is cut.
- **Big letters** = that amino acid appears very often at that position.
- **Tall columns** = the position is very conserved (it carries a lot of information).

**What we see:**

- **Position −1:** a large **A** (alanine), with G and S below it.
- **Position −3:** **A** and **V**.
- Together these form the well-known **"A-x-A" motif**, the pattern recognised by the enzyme that cuts the SP.
- **Positions −13 to −6:** many **L** (leucine). This is the end of the hydrophobic **h-region**.
- **Positions +1 and +2:** almost no pattern. After the cut, the mature protein can start with almost anything.

**Why it matters:** this is exactly the pattern the **von Heijne method** learns. It builds a table (the weight matrix) of which amino acids are typical at each of these 15 positions, then uses it to find cleavage sites in new proteins.

---

## Summary

| Question | Answer |
|---|---|
| Are training and benchmark similar? | **Yes.** All plots look alike, so the benchmark is a fair final test. |
| How long are SPs? | About **22 amino acids** (most 15–30). |
| What are SPs made of? | Mostly **hydrophobic** residues (L, A, V), very few **charged** ones. |
| What does the cleavage site look like? | **A-x-A** motif at positions −3 and −1. |
| Any biases? | Most proteins come from **five model organisms**, and SP proteins are mostly from **animals**. |

**Next step:** use these findings to build the predictors in [`04. Model training`](../04.%20Model%20training): first the von Heijne weight matrix, then the SVM.

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python "03. Data Analysis/data_analysis.py"
```

The plots are saved in `03. Data Analysis/plots/`.
