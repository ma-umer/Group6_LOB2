#!/usr/bin/env python3
"""
Describe and visualize the positive (signal peptide) / negative datasets:
  - protein length distributions (train vs bench, positive vs negative)
  - SP length distribution (train vs bench)
  - amino-acid composition of SPs vs SwissProt reference
  - taxonomic composition (kingdom + species)
  - sequence logos of SP cleavage sites

Sequence logos are NOT generated in Python here — per the course material, this
script exports the aligned cleavage-site windows as a FASTA file that you
upload to WebLogo (https://weblogo.berkeley.edu) to produce the logo.

Requires: pandas, matplotlib, seaborn
    pip install pandas matplotlib seaborn
"""

from pathlib import Path
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

PALETTE = ["blue", "lime", "cyan", "pink", "purple", "magenta", "#7FFFD4", "#550A35"]

# Official SwissProt AA composition (%) — from Expasy
AA_SWISSPROT = {
    "A": 8.25, "C": 1.38, "D": 5.46, "E": 6.71, "F": 3.86,
    "G": 7.07, "H": 2.27, "I": 5.90, "K": 5.79, "L": 9.64,
    "M": 2.41, "N": 4.06, "P": 4.74, "Q": 3.93, "R": 5.52,
    "S": 6.65, "T": 5.36, "V": 6.85, "W": 1.10, "Y": 2.92,
}

sns.set(style="whitegrid")


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


# ---------------------------------------------------------------------------
# 2. Distributions
# ---------------------------------------------------------------------------

def plot_protein_length_histograms(train_df, bench_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, df, title in [(axes[0], train_df, "training"), (axes[1], bench_df, "benchmark")]:
        sns.histplot(df, x="Length", hue="class", kde=True,
                     stat="density", common_norm=False, palette=PALETTE[:2], ax=ax)
        ax.set(title=f"Protein lengths in the {title} set", xlabel="Protein length", xlim=(0, 4000))
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "density_protein_lengths.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_protein_length_boxplots(train_df, bench_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, df, title, ylim in [
        (axes[0], train_df, "training", 6000),
        (axes[1], bench_df, "benchmark", 5000),
    ]:
        sns.boxplot(data=df, y="Length", hue="class", ax=ax)
        ax.set(title=f"Protein lengths in the {title} set", ylim=(0, ylim))
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_protein_length.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_sp_length_distribution(pos_train, pos_bench, out_dir):
    """SP-cleavage-position distribution, positive entries only, train vs bench."""
    pos_train = pos_train.assign(**{"class": "Training"})
    pos_bench = pos_bench.assign(**{"class": "Benchmark"})
    df = pd.concat([pos_train, pos_bench], axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df, x="SP cleavage", hue="class", kde=True,
                 stat="density", common_norm=False, palette=PALETTE[3:5], ax=ax)
    ax.set(title="SP lengths in the training and benchmark sets",
           xlabel="SP length", ylabel="Frequency")
    fig.savefig(out_dir / "SP_lengths.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Amino-acid composition
# ---------------------------------------------------------------------------

def extract_subsequences(fasta_file, pos_df, out_file, slicer, fasta_output=False):
    """Walk a FASTA file and, for every header matching an entry in pos_df,
    write slicer(sequence, cleavage_position) to out_file.

    If fasta_output is True, each result is written as a proper FASTA
    record (">seq_id\\nSUBSEQ\\n") instead of a bare line."""
    id_to_end = pos_df.set_index("seq_id")["SP cleavage"].to_dict()

    with open(fasta_file) as fasta, open(out_file, "w") as out:
        header = None
        for line in fasta:
            if line.startswith(">"):
                header = line[1:].strip()
            elif header in id_to_end:
                end = int(id_to_end[header])
                subseq = slicer(line.strip(), end)
                if fasta_output:
                    out.write(f">{header}\n{subseq}\n")
                else:
                    out.write(subseq + "\n")
                header = None  # avoid reusing a stale header on odd input


def aa_composition(seq_file):
    """Return {amino_acid: percentage} for all residues in seq_file."""
    counts, total = {}, 0
    with open(seq_file) as f:
        for line in f:
            for char in line.strip():
                counts[char] = counts.get(char, 0) + 1
                total += 1
    return {aa: round(count / total * 100, 2) for aa, count in counts.items()}


def plot_aa_composition(aa_train, aa_bench, out_dir):
    aa_order = list(AA_SWISSPROT.keys())
    df = pd.DataFrame({
        "AA": aa_order,
        "Training": [aa_train.get(a, 0) for a in aa_order],
        "Benchmark": [aa_bench.get(a, 0) for a in aa_order],
        "SwissProt": [AA_SWISSPROT[a] for a in aa_order],
    }).set_index("AA")

    ax = df.plot(kind="bar", figsize=(14, 6))
    ax.set(ylabel="Percentage (%)", title="Residue composition")
    fig = ax.get_figure()
    fig.savefig(out_dir / "residues_composition.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Taxonomy
# ---------------------------------------------------------------------------

def plot_kingdom_pies(train_df, bench_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, df, title in [(axes[0], train_df, "training"), (axes[1], bench_df, "benchmark")]:
        df["Kingdom"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax, title=f"Kingdoms in {title} set", colors=PALETTE[3:]
        )
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "kingdom_pie.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_species_barplots(train_df, bench_df, out_dir, top_n=7):
    for df, title, color, fname in [
        (train_df, "training", "#7FFFD4", "barplot_species_train.png"),
        (bench_df, "benchmark", "skyblue", "barplot_species_bench.png"),
    ]:
        counts = df["Organism"].value_counts()
        top = counts.head(top_n)
        others = counts[top_n:].sum()
        result = pd.concat([top, pd.Series({"Others": others})])

        fig, ax = plt.subplots(figsize=(10, 4))
        result.plot(kind="barh", color=color, edgecolor="black", ax=ax)
        ax.set(title=f"Species in {title} set", xlabel="Count", ylabel="Species")
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Sequence logos — export only (logo itself is built on weblogo.berkeley.edu)
# ---------------------------------------------------------------------------

def export_cleavage_windows_fasta(pos_df, fasta_file, out_fasta, window=(13, 2)):
    """Extract the [-N, +M] cleavage-site window for each positive entry and
    write it as FASTA, ready to upload to WebLogo."""
    before, after = window
    extract_subsequences(
        fasta_file, pos_df, out_fasta,
        slicer=lambda seq, end: seq[end - before:end + after],
        fasta_output=True,
    )


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

    # 2. Length distributions
    plot_protein_length_histograms(train_df, bench_df, OUTPUT_DIR)
    plot_protein_length_boxplots(train_df, bench_df, OUTPUT_DIR)
    plot_sp_length_distribution(pos_train, pos_bench, OUTPUT_DIR)

    # 3. Amino-acid composition of SPs
    extract_subsequences(POSITIVE_FASTA, pos_train, "train_SP.seq",
                          slicer=lambda seq, end: seq[:end + 1])
    extract_subsequences(POSITIVE_FASTA, pos_bench, "bench_SP.seq",
                          slicer=lambda seq, end: seq[:end + 1])
    aa_train = aa_composition("train_SP.seq")
    aa_bench = aa_composition("bench_SP.seq")
    plot_aa_composition(aa_train, aa_bench, OUTPUT_DIR)

    # 4. Taxonomy
    plot_kingdom_pies(train_df, bench_df, OUTPUT_DIR)
    plot_species_barplots(train_df, bench_df, OUTPUT_DIR)

    # 5. Export cleavage-site windows [-13, +2] for WebLogo (weblogo.berkeley.edu)
    export_cleavage_windows_fasta(pos_train, POSITIVE_FASTA, OUTPUT_DIR / "train_cleavage_windows.fasta")
    export_cleavage_windows_fasta(pos_bench, POSITIVE_FASTA, OUTPUT_DIR / "bench_cleavage_windows.fasta")
    print(f"Cleavage-window FASTA files written to {OUTPUT_DIR} — "
          f"upload them to https://weblogo.berkeley.edu to build the logos.")


if __name__ == "__main__":
    main()
