#!/usr/bin/env python3
"""
SVM-based signal peptide classifier.

Pipeline
  1. Load the cross-validation / benchmark TSVs and attach sequences (same
     layout used by 04_von_heijne/model_creation_combined.py).
  2. Extract 39 sequence-derived features from the N-terminal region:
       - amino acid composition (20)                       first 40 residues
       - Kyte-Doolittle hydrophobicity + aliphatic index (6) first 40 residues
       - transmembrane helix propensity (3)                 first 80 residues
       - alpha-helix / beta-sheet propensity (6)            first 40 residues
       - net charge (4)                                     first 20 residues
     Scales from ProtScale (https://web.expasy.org/protscale/).
  3. 5-fold rotating cross-validation (test = fold i, validation = fold i+1,
     training = remaining 3 folds - same scheme as the von Heijne model):
       a. grid search over C and gamma (RBF kernel) maximising validation MCC
          -> "all features" model
       b. Random Forest Gini importance (training folds only) ranks features;
          the number k of top features is chosen on the validation MCC curve,
          then C / gamma are re-tuned on the selected subset
          -> "selected features" model
       c. both models evaluated on the test fold.
  4. The best CV round (test MCC of the selected-features model) is refit on
     its training + validation folds and evaluated on the benchmark set.

Run:  python svm_model.py            (paths default to the project layout)
"""

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


# ---------------------------------------------------------------------------
# 3. Model helpers
# ---------------------------------------------------------------------------

def svm_pipeline(C, gamma):
    return Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_STATE))])


def compute_metrics(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else 0.0
    precision, recall = div(TP, TP + FP), div(TP, TP + FN)
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {"TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
            "MCC": div(TP * TN - FP * FN, denom), "Precision": precision,
            "Recall": recall, "Accuracy": div(TP + TN, TP + TN + FP + FN),
            "F1-score": div(2 * precision * recall, precision + recall),
            "Specificity": div(TN, TN + FP)}


def grid_search(X_tr, y_tr, X_va, y_va):
    """Best (C, gamma) by validation MCC."""
    best_mcc, best = -1.0, None
    for C in C_GRID:
        for gamma in GAMMA_GRID:
            model = svm_pipeline(C, gamma).fit(X_tr, y_tr)
            mcc = compute_metrics(y_va, model.predict(X_va))["MCC"]
            if mcc > best_mcc:
                best_mcc, best = mcc, {"C": C, "gamma": gamma}
    return best_mcc, best


def gini_ranking(X_tr, y_tr):
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return (pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def feature_index(names):
    return [int(np.where(FEATURE_NAMES == f)[0][0]) for f in names]


def save_model(model, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(m, title, path):
    """Rows = predicted class, columns = actual class (same layout as the VH model)."""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    draw_confusion_matrix(ax, m["TP"], m["FP"], m["FN"], m["TN"])
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    save_figure(fig, path)


def plot_top_features(gini_df, title, path, top=20):
    head = gini_df.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(head["feature"], head["importance"], color=SERIES[0], height=0.7)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Gini importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, path)


def plot_mcc_curve(ks, mccs, best_k, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, mccs, marker="o", ms=4, color=SERIES[0])
    ax.axvline(best_k, ls="--", color=MUTED, label=f"best k = {best_k}")
    ax.set_xlabel("k (top features by RF Gini importance)")
    ax.set_ylabel("Validation MCC (SVM)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, path)


def plot_all_mcc_curves(results, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        ax.plot(r["ks"], r["k_mcc"], marker="o", ms=3, color=SERIES[i % len(SERIES)],
                label=f"Round {r['run']} (best k={r['k']})")
    ax.set_xlabel("k (top features by RF Gini importance)")
    ax.set_ylabel("Validation MCC (SVM)")
    ax.set_title("MCC vs number of selected features (validation folds)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# 5. Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(train_df, X_all, rounds_dir):
    folds = sorted(int(f) for f in train_df["Label"].unique())
    labels = train_df["Label"].to_numpy()
    y_all = train_df["Class"].to_numpy()
    results = []

    for i, test_fold in enumerate(folds):
        val_fold = folds[(i + 1) % len(folds)]
        train_folds = [f for f in folds if f not in (test_fold, val_fold)]
        run = i + 1
        print("/" * 60)
        print(f"Run {run}: test fold = {test_fold}, validation fold = {val_fold}, "
              f"training folds = {train_folds}")

        tr, va, te = np.isin(labels, train_folds), labels == val_fold, labels == test_fold
        X_tr, X_va, X_te = X_all[tr], X_all[va], X_all[te]
        y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

        # a. all features
        val_mcc_all, params_all = grid_search(X_tr, y_tr, X_va, y_va)
        model_all = svm_pipeline(**params_all).fit(X_tr, y_tr)
        m_all = compute_metrics(y_te, model_all.predict(X_te))
        print(f"  All features   : params {params_all}, val MCC {val_mcc_all:.3f}, "
              f"test MCC {m_all['MCC']:.3f}")

        # b. feature selection (RF ranking on training folds, k chosen on validation)
        gini_df = gini_ranking(X_tr, y_tr)
        gini_df.to_csv(rounds_dir / f"features_gini_round{run}.tsv", sep="\t", index=False)
        ks = [k for k in K_RANGE if k <= X_all.shape[1]]
        k_mcc = []
        for k in ks:
            idx = feature_index(gini_df["feature"].head(k))
            model = svm_pipeline(**params_all).fit(X_tr[:, idx], y_tr)
            k_mcc.append(compute_metrics(y_va, model.predict(X_va[:, idx]))["MCC"])
        best_k = ks[int(np.argmax(k_mcc))]
        selected = gini_df["feature"].head(best_k).tolist()
        idx = feature_index(selected)

        val_mcc_sel, params_sel = grid_search(X_tr[:, idx], y_tr, X_va[:, idx], y_va)
        model_sel = svm_pipeline(**params_sel).fit(X_tr[:, idx], y_tr)
        m_sel = compute_metrics(y_te, model_sel.predict(X_te[:, idx]))
        print(f"  Selected ({best_k:2d})  : params {params_sel}, val MCC {val_mcc_sel:.3f}, "
              f"test MCC {m_sel['MCC']:.3f}")
        print(f"  Selected features: {', '.join(selected)}")

        save_model(model_all, rounds_dir / f"svm_all_round{run}.pkl.gz")
        save_model(model_sel, rounds_dir / f"svm_selected_round{run}.pkl.gz")
        plot_top_features(gini_df, f"Random Forest Gini importance (top 20, round {run})",
                          rounds_dir / f"top_features_round{run}.png")
        plot_mcc_curve(ks, k_mcc, best_k, f"MCC vs selected features (round {run}, validation)",
                       rounds_dir / f"mcc_vs_k_round{run}.png")
        for tag, m, name in (("all", m_all, "all features"),
                             ("selected", m_sel, f"{best_k} selected features")):
            plot_confusion_matrix(
                m, f"Confusion Matrix - {name} (round {run})\n"
                   f"Acc={m['Accuracy']:.3f}, F1={m['F1-score']:.3f}, MCC={m['MCC']:.3f}",
                rounds_dir / f"cm_{tag}_round{run}.png")

        results.append({"run": run, "test_fold": test_fold, "val_fold": val_fold,
                        "train_folds": train_folds, "params_all": params_all,
                        "params_sel": params_sel, "k": best_k, "selected": selected,
                        "ks": ks, "k_mcc": k_mcc, "m_all": m_all, "m_sel": m_sel})
    return results


def summarize_cv(results, out_dir):
    rows, summary = [], []
    for model_key, model_name in (("m_all", "All features"), ("m_sel", "Selected features")):
        for r in results:
            params = r["params_all"] if model_key == "m_all" else r["params_sel"]
            rows.append({"Model": model_name, "Run": r["run"], "Test fold": r["test_fold"],
                         "Val fold": r["val_fold"],
                         "N features": len(FEATURE_NAMES) if model_key == "m_all" else r["k"],
                         "C": params["C"], "gamma": params["gamma"],
                         **{m: r[model_key][m] for m in METRICS}})
        for m in METRICS:
            vals = np.array([r[model_key][m] for r in results])
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            summary.append({"Model": model_name, "Metric": m, "Mean": vals.mean(),
                            "Std": std, "SE": std / np.sqrt(len(vals))})
    runs_df, summary_df = pd.DataFrame(rows), pd.DataFrame(summary)
    runs_df.to_csv(out_dir / "cv_runs.tsv", sep="\t", index=False)
    summary_df.to_csv(out_dir / "cv_summary.tsv", sep="\t", index=False)
    pd.DataFrame([{"Run": r["run"], "k": r["k"], "Selected features": ", ".join(r["selected"])}
                  for r in results]).to_csv(out_dir / "selected_features.tsv", sep="\t", index=False)
    print("/" * 60)
    print("\nCross-validation runs (test folds):")
    print(runs_df.round(3).to_string(index=False))
    print("\nCross-validation summary (mean, std, SE):")
    print(summary_df.round(3).to_string(index=False))
    return summary_df


def plot_combined_confusion(results, summary_df, out_dir):
    for model_key, model_name, tag in (("m_all", "All features", "all"),
                                       ("m_sel", "Selected features", "selected")):
        tot = {k: sum(r[model_key][k] for r in results) for k in ("TP", "FP", "FN", "TN")}
        mean = summary_df[summary_df["Model"] == model_name].set_index("Metric")["Mean"]
        plot_confusion_matrix(
            tot, f"Combined Confusion Matrix - {model_name}\n"
                 f"Avg Acc={mean['Accuracy']:.3f}, F1={mean['F1-score']:.3f}, MCC={mean['MCC']:.3f}",
            out_dir / f"confusion_matrix_combined_{tag}.png")


# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------

def transmembrane_fpr(bench_df, y_true, y_pred):
    tm_col = next((c for c in TM_COLUMNS if c in bench_df.columns), None)
    if tm_col is None:
        return
    is_tm = bench_df[tm_col].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()
    neg, fp = y_true == 0, (y_true == 0) & (y_pred == 1)
    for name, mask in (("Transmembrane", is_tm), ("Non-transmembrane", ~is_tm)):
        n = neg[mask].sum()
        print(f"  FPR ({name}) = {fp[mask].sum() / n * 100:.2f}%  ({fp[mask].sum()}/{n})"
              if n else f"  FPR ({name}) = n/a")


def evaluate_benchmark(train_df, X_train_all, bench_df, X_bench, results, out_dir):
    best = max(results, key=lambda r: r["m_sel"]["MCC"])
    folds = best["train_folds"] + [best["val_fold"]]
    mask = train_df["Label"].isin(folds).to_numpy()
    y_tr = train_df["Class"].to_numpy()[mask]
    y_true = bench_df["Class"].to_numpy()
    print("\n" + "/" * 60)
    print(f"Best CV run: {best['run']} (selected-features test MCC {best['m_sel']['MCC']:.3f})")
    print(f"Refitting on folds {folds} (test fold {best['test_fold']} left out)")

    rows = []
    for tag, name, idx, params in (
            ("all", "All features", list(range(len(FEATURE_NAMES))), best["params_all"]),
            ("selected", f"Selected features ({best['k']})", feature_index(best["selected"]),
             best["params_sel"])):
        model = svm_pipeline(**params).fit(X_train_all[mask][:, idx], y_tr)
        save_model(model, out_dir / f"svm_{tag}_final.pkl.gz")
        y_pred = model.predict(X_bench[:, idx])
        scores = model.decision_function(X_bench[:, idx])
        m = compute_metrics(y_true, y_pred)
        rows.append({"Model": name, "C": params["C"], "gamma": params["gamma"], **m})
        print(f"\n{name} - params {params}")
        print(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
        print("  " + ", ".join(f"{k}={m[k]:.3f}" for k in METRICS))
        transmembrane_fpr(bench_df, y_true, y_pred)
        plot_confusion_matrix(
            m, f"Benchmark Confusion Matrix - {name}\n"
               f"Acc={m['Accuracy']:.3f}, F1={m['F1-score']:.3f}, MCC={m['MCC']:.3f}",
            out_dir / f"confusion_matrix_benchmark_{tag}.png")

        if tag == "selected":   # export predictions of the final (selected) model
            pred = bench_df.drop(columns=["Label"]).copy()
            pred["svm_score"], pred["y_pred"] = scores, y_pred
            pred.to_csv(out_dir / "benchmark_predictions.tsv", sep="\t", index=False)
            for fname, sel in (("false_negatives", (y_true == 1) & (y_pred == 0)),
                               ("true_positives", (y_true == 1) & (y_pred == 1)),
                               ("false_positives", (y_true == 0) & (y_pred == 1))):
                pred[sel].to_csv(out_dir / f"benchmark_{fname}.tsv", sep="\t", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "benchmark_metrics.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train/cross-validate SVM signal peptide classifiers.")
    ap.add_argument("--cv-dir", type=Path, default=CV_DIR)
    ap.add_argument("--pos-fasta", type=Path, default=POS_FASTA)
    ap.add_argument("--neg-fasta", type=Path, default=NEG_FASTA)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.output
    rounds_dir = out_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    train_df, bench_df = load_data(args.cv_dir, args.pos_fasta, args.neg_fasta)
    print(f"Training rows: {len(train_df)} | benchmark rows: {len(bench_df)}")

    X_train_all = extract_features(train_df["Frag_90"])
    X_bench = extract_features(bench_df["Frag_90"])
    feats = pd.DataFrame(X_train_all, columns=FEATURE_NAMES)
    feats.insert(0, "UniProt_ID", train_df["UniProt_ID"])
    feats.insert(1, "Class", train_df["Class"])
    feats.insert(2, "Fold", train_df["Label"])
    feats.to_csv(out_dir / "training_features.tsv", sep="\t", index=False)
    print(f"Extracted {X_train_all.shape[1]} features per sequence")

    results = run_cross_validation(train_df, X_train_all, rounds_dir)
    summary_df = summarize_cv(results, out_dir)
    plot_combined_confusion(results, summary_df, out_dir)
    plot_all_mcc_curves(results, out_dir / "mcc_vs_k_all_rounds.png")

    evaluate_benchmark(train_df, X_train_all, bench_df, X_bench, results, out_dir)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
