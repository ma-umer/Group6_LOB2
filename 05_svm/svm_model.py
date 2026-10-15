#!/usr/bin/env python3
import argparse
import gzip
import math
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CV_DIR = PROJECT_DIR / "02_data_preparation" / "cross_validation"
POS_FASTA = PROJECT_DIR / "01_data_collection" / "positives" / "positive.fasta"
NEG_FASTA = PROJECT_DIR / "01_data_collection" / "negatives" / "negative.fasta"
OUTPUT_DIR = BASE_DIR / "model_results"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FRAG_LEN = 90                       # N-terminal fragment kept per sequence
C_GRID = [0.1, 1, 10, 100]
GAMMA_GRID = ["scale", 0.01, 0.1, 1.0]
K_RANGE = range(2, 40)              # number of top-ranked features tried in selection
RANDOM_STATE = 42

TM_COLUMNS = ("N-term TM", "Transmembrane_Helix", "TransM_Helix_90")
METRICS = ["MCC", "Precision", "Recall", "Accuracy", "F1-score", "Specificity"]


# ---------------------------------------------------------------------------
# Colours (shared look for 04, 05 and 06)
#   SERIES     categorical, fixed order; validated colour-blind safe
#              (worst adjacent CVD dE 9.1, normal-vision dE 19.6)
#   SEQ_BLUE   one-hue light->dark ramp for magnitudes (confusion matrices)
#   DIVERGING  blue <- grey -> red for signed scores (PSWM log-odds),
#              always centred on 0
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]   # blue, orange, aqua, yellow, magenta, green, violet, red
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ_BLUE = mpl.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f3a59d", "#d63c3a", "#8a1c1c"])
PSWM_LIMIT = 3.0   # colour scale of PSWM heatmaps is clipped to +-3 (log2 odds)


def setup_style():
    """Seaborn/matplotlib theme (Liberation Serif if installed, else DejaVu Serif)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "Liberation Serif" if "Liberation Serif" in available else "DejaVu Serif"
    sns.set_theme(context="notebook", style="white", font=font, font_scale=1.1)
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def draw_confusion_matrix(ax, TP, FP, FN, TN):
    """Rows = predicted class, columns = actual class. Each cell is shaded by its
    share of the actual class, so the rare positive class is as readable as the
    negatives."""
    counts = np.array([[TP, FP], [FN, TN]])
    names = np.array([["TP", "FP"], ["FN", "TN"]])
    totals = counts.sum(axis=0)
    frac = counts / np.where(totals == 0, 1, totals)
    for r in range(2):
        for c in range(2):
            f = frac[r, c]
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=SEQ_BLUE(f),
                                       edgecolor="white", linewidth=4))
            ink = "white" if f > 0.45 else INK
            ax.text(c + 0.5, 1.57 - r, f"{counts[r, c]:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=ink)
            ax.text(c + 0.5, 1.33 - r, f"{names[r, c]} \u00b7 {f:.1%}", ha="center",
                    va="center", fontsize=9, color=ink)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Actual Positive", "Actual Negative"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Predicted Negative", "Predicted Positive"], rotation=90, va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1, -0.06, "% = share of the actual class", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MUTED)


setup_style()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------


def load_fasta(path):
    """Return {accession: sequence} (multi-line sequences are joined)."""
    seqs, header, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks)
                header, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if header is not None:
        seqs[header] = "".join(chunks)
    return seqs


def _build_table(tsv, fasta_seqs, class_value):
    df = pd.read_csv(tsv, sep="\t")
    df["seq_id"] = df["seq_id"].astype(str).str.strip()
    missing = ~df["seq_id"].isin(fasta_seqs)
    if missing.any():
        print(f"  Warning: {missing.sum()} IDs in {Path(tsv).name} not in FASTA - dropped")
        df = df[~missing]
    seqs = df["seq_id"].map(fasta_seqs)
    out = pd.DataFrame({
        "UniProt_ID": df["seq_id"].to_numpy(),
        "Class": class_value,
        "Frag_90": seqs.str[:FRAG_LEN].to_numpy(),
        "Label": df["fold"].astype(int).to_numpy() if "fold" in df.columns else 0,
    })
    for col in ("Organism", "Kingdom", "Length", "SP cleavage", "N-term TM"):
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


def load_data(cv_dir, pos_fasta, neg_fasta):
    """(train_df, bench_df). Label 1..N = CV fold, 0 = benchmark."""
    cv_dir = Path(cv_dir)
    pos_seqs, neg_seqs = load_fasta(pos_fasta), load_fasta(neg_fasta)
    train_df = pd.concat([_build_table(cv_dir / "pos_train.tsv", pos_seqs, 1),
                          _build_table(cv_dir / "neg_train.tsv", neg_seqs, 0)],
                         ignore_index=True)
    bench_df = pd.concat([_build_table(cv_dir / "pos_bench.tsv", pos_seqs, 1),
                          _build_table(cv_dir / "neg_bench.tsv", neg_seqs, 0)],
                         ignore_index=True)
    overlap = set(train_df["UniProt_ID"]) & set(bench_df["UniProt_ID"])
    if overlap:
        print(f"  WARNING: {len(overlap)} IDs in both CV folds and benchmark")
    return train_df, bench_df


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

ALPHABET = "AQLSREKTNGMWDHFYCIPV"   # order used for the composition features

KD_SCALE = {  # Kyte & Doolittle hydrophobicity
    'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5, 'E': -3.5, 'H': -3.2,
    'P': -1.6, 'Y': -1.3, 'W': -0.9, 'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8,
    'M': 1.9, 'C': 2.5, 'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5}

# aliphatic index weights: X_A + 2.9 X_V + 3.9 (X_I + X_L)
AI_WEIGHTS = {'A': 1.0, 'V': 2.9, 'I': 3.9, 'L': 3.9}

ALPHA_SCALE = {  # Chou & Fasman alpha-helix propensity
    'G': 0.57, 'P': 0.57, 'Y': 0.69, 'C': 0.70, 'S': 0.77, 'T': 0.83, 'N': 0.67,
    'R': 0.98, 'H': 1.00, 'D': 1.01, 'I': 1.08, 'W': 1.08, 'Q': 1.11, 'F': 1.13,
    'K': 1.16, 'V': 1.06, 'L': 1.21, 'A': 1.42, 'M': 1.45, 'E': 1.51}

BETA_SCALE = {  # Chou & Fasman beta-sheet propensity
    'E': 0.37, 'D': 0.54, 'P': 0.55, 'G': 0.75, 'S': 0.75, 'K': 0.74, 'H': 0.87,
    'N': 0.89, 'R': 0.93, 'A': 0.83, 'M': 1.05, 'Q': 1.10, 'C': 1.19, 'T': 1.19,
    'L': 1.30, 'F': 1.38, 'W': 1.37, 'Y': 1.47, 'I': 1.60, 'V': 1.70}

TM_SCALE = {  # transmembrane tendency (Zhao & London)
    'F': 1.98, 'I': 1.97, 'L': 1.82, 'W': 1.53, 'V': 1.46, 'M': 1.40, 'Y': 0.49,
    'A': 0.38, 'G': -0.19, 'C': -0.30, 'T': -0.32, 'S': -0.53, 'H': -1.44,
    'P': -1.44, 'N': -1.62, 'Q': -1.84, 'R': -2.57, 'E': -2.90, 'D': -3.27, 'K': -3.46}

CHARGE_SCALE = {'K': 1.0, 'R': 1.0, 'H': 0.5, 'D': -1.0, 'E': -1.0}


def window_profile(seq, scale, window, triangular=False):
    """Per-residue sliding-window average of `scale` along `seq`.

    The sequence is padded with (window // 2) empty positions at both ends so
    every residue gets the score of the window centred on it. Residues not in
    the scale (X, U, padding) contribute 0. With triangular=True the window is
    weighted 1 at the centre and 0 at the edges.
    """
    values = np.array([scale.get(aa, 0.0) for aa in seq])
    d = window // 2
    padded = np.concatenate([np.zeros(d), values, np.zeros(d)])
    if triangular:
        c = (window - 1) / 2
        weights = 1 - np.abs(np.arange(window) - c) / c
    else:
        weights = np.ones(window)
    return np.convolve(padded, weights[::-1], mode="valid") / window


def summary(profile):
    """mean, max and position (0-based residue index) of the max."""
    return [profile.mean(), profile.max(), int(np.argmax(profile))]


def aa_composition(seq):
    seq = seq[:40]
    return [seq.count(aa) / len(seq) for aa in ALPHABET]


def hydrophobicity_aliphatic(seq, window=7):
    seq = seq[:40]
    return (summary(window_profile(seq, KD_SCALE, window)) +
            summary(window_profile(seq, AI_WEIGHTS, window)))


def tm_propensity(seq, window=7):
    return summary(window_profile(seq[:80], TM_SCALE, window, triangular=True))


def secondary_structure(seq, window=9):
    seq = seq[:40]
    return (summary(window_profile(seq, ALPHA_SCALE, window, triangular=True)) +
            summary(window_profile(seq, BETA_SCALE, window, triangular=True)))


def charge(seq, window=3):
    p = window_profile(seq[:20], CHARGE_SCALE, window)
    return [p.max(), int(np.argmax(p)), p.min(), int(np.argmin(p))]


FEATURE_NAMES = np.array(
    [f"composition_({aa})" for aa in ALPHABET] +
    ["mean_hydrophobicity", "max_hydrophobicity", "most_hydrophobic_position",
     "mean_aliphatic_index", "max_aliphatic_index", "most_aliphatic_position",
     "tm_mean", "tm_max", "tm_position",
     "mean_alpha_helix", "max_alpha_helix", "alpha_helix_position",
     "mean_beta_sheet", "max_beta_sheet", "beta_sheet_position",
     "max_charge", "max_charge_position", "min_charge", "min_charge_position"])


def extract_features(seqs):
    rows = [aa_composition(s) + hydrophobicity_aliphatic(s) + tm_propensity(s) +
            secondary_structure(s) + charge(s) for s in seqs]
    X = np.array(rows, dtype=float)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X


############## TODO



if __name__ == "__main__":
    main()
