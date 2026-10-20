# 02. Data Preparation

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
