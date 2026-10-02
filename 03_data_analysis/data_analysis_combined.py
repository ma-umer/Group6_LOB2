#!/usr/bin/env python3
"""
Describe and visualize the positive (signal peptide) / negative datasets:
  - SP length distribution (train vs bench)
  - amino-acid composition of SPs vs SwissProt reference
  - sequence motifs around the SP cleavage site (exported as TSV)
  - taxonomic composition (kingdom + species)
  - protein length distributions (train vs bench, positive vs negative)

Plotting/analysis functions come from the original standalone scripts
(comparison_aa, motif_logo, signal_peptide_graph, taxonomic_classification,
proteinlenght); the layout (paths, load_dataset, main) follows data_analysis.py.

Requires: pandas, numpy, matplotlib, seaborn
    pip install pandas numpy matplotlib seaborn
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths — adjust to match your project layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CROSS_VAL_DIR = BASE_DIR.parent / "02_data_preparation" / "cross_validation"
POSITIVE_FASTA = BASE_DIR.parent / "01_data_collection" / "positives" / "positive.fasta"

OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_TRAIN_TSV = OUTPUT_DIR / "motifs_training.tsv"
OUTPUT_BENCH_TSV = OUTPUT_DIR / "motifs_benchmark.tsv"

WINDOW_COLUMN = "SP cleavage"   # was "Signal_Peptide" in the old combined dataset
WINDOW_LEFT = 12
WINDOW_RIGHT = 3

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# SwissProt AA composition (fractions, same order as AA_LIST)
SWISSPROT_FREQ = [0.078, 0.013, 0.053, 0.063, 0.039, 0.073, 0.022, 0.062, 0.058, 0.092,
                  0.023, 0.045, 0.051, 0.039, 0.052, 0.070, 0.058, 0.065, 0.013, 0.032]


# ---------------------------------------------------------------------------
# 1. Load & label datasets
# ---------------------------------------------------------------------------

def load_dataset(pos_file, neg_file):
    """Load a positive + negative TSV pair, concatenate, and label by class."""
    pos = pd.read_csv(pos_file, sep="\t")
    neg = pd.read_csv(neg_file, sep="\t")
    df = pd.concat([pos, neg], axis=0)
    df["class"] = df["SP cleavage"].isna().map({True: "Negative", False: "Positive"})
    return pos, neg, df


def load_fasta(fasta_file):
    """Return {header: sequence} (multi-line sequences are joined)."""
    sequences, header, chunks = {}, None, []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        sequences[header] = "".join(chunks)
    return sequences


def valid_sp(pos_df):
    """Keep only rows with a valid signal peptide position."""
    return pos_df[pos_df[WINDOW_COLUMN].notnull() & (pos_df[WINDOW_COLUMN] > 0)]


# ---------------------------------------------------------------------------
# 2. SP length distribution  (from signal_peptide_graph.py)
# ---------------------------------------------------------------------------

def plot_sp_length_graph(pos_train, pos_bench, out_dir):
    train_sp = pd.to_numeric(pos_train[WINDOW_COLUMN])
    train_sp = train_sp[train_sp > 0]
    bench_sp = pd.to_numeric(pos_bench[WINDOW_COLUMN])
    bench_sp = bench_sp[bench_sp > 0]

    print("Training SP count:", len(train_sp))
    print(train_sp.describe())
    print("Benchmark SP count:", len(bench_sp))
    print(bench_sp.describe())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_sp, bins=30, stat="probability", alpha=0.3, color="blue", label="Train")
    sns.kdeplot(train_sp, color="blue", linewidth=2)
    plt.title("Signal Peptide Lengths (Training)")
    plt.xlabel("Signal Peptide Length (aa)")
    plt.ylabel("Probability")
    plt.text(53, 0.105, f"Training count: {len(train_sp)}", color="blue")

    plt.subplot(1, 2, 2)
    sns.histplot(bench_sp, bins=30, stat="probability", alpha=0.3, color="green", label="Benchmark")
    sns.kdeplot(bench_sp, color="green", linewidth=2)
    plt.title("Signal Peptide Lengths (Benchmark)")
    plt.xlabel("Signal Peptide Length (aa)")
    plt.ylabel("Probability")
    plt.text(45, 0.1, f"Benchmark count: {len(bench_sp)}", color="green")

    plt.tight_layout()
    plt.savefig(out_dir / "signal_peptide_graph.png", dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3. Amino-acid composition  (from comparison_aa.py)
# ---------------------------------------------------------------------------

def aa_freq(df_subset, sequences):
    counts = {aa: 0 for aa in AA_LIST}
    total = 0

    for _, row in df_subset.iterrows():
        sequence = sequences[row["seq_id"]]
        sp_seq = sequence[:int(row[WINDOW_COLUMN])]

        for aa in sp_seq:
            if aa in counts:
                counts[aa] += 1
                total += 1

    freqs = [counts[aa] / total if total > 0 else 0 for aa in AA_LIST]
    return freqs


def plot_aa_composition(train_freq, bench_freq, swissprot_freq, out_dir):
    x = np.arange(len(AA_LIST))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, train_freq, width, label="Training SPs")
    ax.bar(x, bench_freq, width, label="Benchmark SPs")
    ax.bar(x + width, swissprot_freq, width, label="SwissProt")
    ax.set_xticks(x)
    ax.set_xticklabels(AA_LIST)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Amino Acid")
    ax.set_title("Comparative Amino-Acid Composition of SPs")
    ax.legend()
    fig.savefig(out_dir / "comparative_aa_for_sp.png", dpi=300)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Cleavage-site motifs  (from motif_logo.py)
# ---------------------------------------------------------------------------

def extract_motifs(df_subset, sequences):
    motifs = []
    for index, row in df_subset.iterrows():
        cleavage_site = int(row[WINDOW_COLUMN])
        sequence = sequences[row["seq_id"]]

        cleavage_index = cleavage_site - 1
        start = max(0, cleavage_index - WINDOW_LEFT)
        end = cleavage_index + WINDOW_RIGHT
        motif = sequence[start:end].ljust(WINDOW_LEFT + WINDOW_RIGHT, "-")
        motifs.append(motif)
    return motifs


def save_motifs(motifs_train, motifs_bench):
    pd.DataFrame(motifs_train, columns=["Motif"]).to_csv(OUTPUT_TRAIN_TSV, sep="\t", index=False)
    pd.DataFrame(motifs_bench, columns=["Motif"]).to_csv(OUTPUT_BENCH_TSV, sep="\t", index=False)

    print(f" Saved {len(motifs_train)} training motifs to {OUTPUT_TRAIN_TSV}")
    print(f" Saved {len(motifs_bench)} benchmark motifs to {OUTPUT_BENCH_TSV}")


# ---------------------------------------------------------------------------
# 5. Taxonomy  (from taxonomic_classification.py)
# ---------------------------------------------------------------------------

def plot_side_by_side(data_train, data_bench, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Training Distribution
    kingdom_counts = data_train["Kingdom"].value_counts()
    axes[0].pie(
        kingdom_counts,
        labels=kingdom_counts.index,
        autopct="%1.1f%%",
        colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(kingdom_counts)))
    )
    axes[0].set_title("Training Set – Kingdom Distribution")

    species_counts = data_train["Organism"].value_counts()
    top_species = species_counts.head(5)
    others = species_counts.iloc[5:].sum()
    species_data = pd.concat([top_species, pd.Series({"Others": others})])

    axes[1].pie(
        species_data,
        labels=species_data.index,
        autopct="%1.1f%%",
        colors=plt.cm.Greens(np.linspace(0.4, 0.9, len(species_data)))
    )
    axes[1].set_title("Training Set – Species Distribution")

    plt.tight_layout()
    plt.savefig(out_dir / "taxonomic_classification_training.png", dpi=300)
    plt.show()
    plt.close(fig)

    # Benchmarking Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    kingdom_counts = data_bench["Kingdom"].value_counts()
    axes[0].pie(
        kingdom_counts,
        labels=kingdom_counts.index,
        autopct="%1.1f%%",
        colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(kingdom_counts)))
    )
    axes[0].set_title("Benchmarking Set – Kingdom Distribution")

    species_counts = data_bench["Organism"].value_counts()
    top_species = species_counts.head(5)
    others = species_counts.iloc[5:].sum()
    species_data = pd.concat([top_species, pd.Series({"Others": others})])

    axes[1].pie(
        species_data,
        labels=species_data.index,
        autopct="%1.1f%%",
        colors=plt.cm.Greens(np.linspace(0.4, 0.9, len(species_data)))
    )
    axes[1].set_title("Benchmarking Set – Species Distribution")

    plt.tight_layout()
    plt.savefig(out_dir / "taxonomic_classification_benchmark.png", dpi=300)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Protein length distributions  (from proteinlenght.py)
# ---------------------------------------------------------------------------

def plot_normalized_hist_subplot(df_subset, title, pos):
    plt.subplot(1, 2, pos)
    for label, color in zip(["Positive", "Negative"], ["blue", "green"]):
        data = df_subset.loc[df_subset["class"] == label, "Length"]
        weights = np.ones_like(data) / len(data)   # normalize class-wise
        sns.histplot(
            x=data,
            bins=150,
            element="bars",
            fill=True,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.4,
            kde=False,
            color=color,
            stat="count",
            weights=weights,
            label=label
        )
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Protein Length (aa)", fontsize=12)
    plt.ylabel("Probability (within class)", fontsize=12)
    plt.xlim(0, 3000)
    plt.legend(title="Label")


def plot_protein_length_comparison(train_df, bench_df, out_dir):
    # Seaborn theme (global — kept here and run last so it doesn't restyle the other plots)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 5))

    # Left: Train
    plot_normalized_hist_subplot(train_df, "(Train) Distribution of Protein Lengths", 1)

    # Right: Benchmark
    plot_normalized_hist_subplot(bench_df, "(Benchmark) Distribution of Protein Lengths", 2)

    plt.tight_layout()
    plt.savefig(out_dir / "protein_length_comparison.png", dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load and label
    pos_train, neg_train, train_df = load_dataset(
        CROSS_VAL_DIR / "pos_train.tsv", CROSS_VAL_DIR / "neg_train.tsv"
    )
    pos_bench, neg_bench, bench_df = load_dataset(
        CROSS_VAL_DIR / "pos_bench.tsv", CROSS_VAL_DIR / "neg_bench.tsv"
    )
    sequences = load_fasta(POSITIVE_FASTA)

    # Only positives with a valid SP position are used for SP-based analyses
    sp_train = valid_sp(pos_train)
    sp_bench = valid_sp(pos_bench)

    # 2. SP length distribution
    plot_sp_length_graph(pos_train, pos_bench, OUTPUT_DIR)

    # 3. Amino-acid composition of SPs
    train_freq = aa_freq(sp_train, sequences)
    bench_freq = aa_freq(sp_bench, sequences)
    plot_aa_composition(train_freq, bench_freq, SWISSPROT_FREQ, OUTPUT_DIR)

    # 4. Cleavage-site motifs
    motifs_train = extract_motifs(sp_train, sequences)
    motifs_bench = extract_motifs(sp_bench, sequences)
    save_motifs(motifs_train, motifs_bench)

    # 5. Taxonomy
    plot_side_by_side(train_df, bench_df, OUTPUT_DIR)

    # 6. Protein length distributions (last: sets a global seaborn theme)
    plot_protein_length_comparison(train_df, bench_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
