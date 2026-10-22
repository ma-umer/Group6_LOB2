#!/usr/bin/env python3
"""
SVM-based signal peptide classifier (nested cross-validation).

Pipeline
  1. Load the cross-validation / benchmark TSVs and attach sequences (same
     layout used by 04_von_heijne/von_heijne_method.py).
  2. Extract 39 sequence-derived features from the N-terminal region:
       - amino acid composition (20)                       first n residues,
         n = median SP length of the current training set
       - Kyte-Doolittle hydrophobicity + aliphatic index (6) first 40 residues
       - transmembrane helix propensity (3)                 first 80 residues
       - alpha-helix / beta-sheet propensity (6)            first 40 residues
       - net charge (4)                                     first 20 residues
     Scales from ProtScale (https://web.expasy.org/protscale/).
  3. Nested cross-validation on the 5 predefined subsets:
       outer loop  each subset is the test set once; the other 4 are training
       inner loop  4-fold CV on those 4 subsets (3 train + 1 validation);
                   for every (k, C, gamma) the mean validation MCC over the 4
                   inner folds is computed (grid extended automatically if the
                   best C or gamma is on its edge)
       choice      (k, C, gamma) with the highest mean validation MCC, as in
                   the course protocol (Practical session IV)
       refit       n, the Random Forest ranking, the scaler and the SVM are
                   refitted on all 4 training subsets, then tested once.
     Three models per run, all from the same inner search:
       "selected features"  the pipeline above (RF top-k -> SVM), the main model
       "one-SE rule"        variant: the smallest k whose best mean MCC is within
                            one standard error of the best (feature-selection
                            slides, "Read the shape, not the peak"), best C, gamma
                            for that k
       "all features"       baseline: k = 39, only C and gamma tuned
  4. Final model (course slides, "Training the two final predictors"): the same
     search as a 5-fold CV over all hyperparameters on the whole training set;
     the best combination is refitted on all of it and evaluated once on the
     benchmark set. The benchmark is never used for any choice.
  5. Feature selection is reported as selection frequency over the 5 outer runs,
     not as one run's list.

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
from joblib import Parallel, delayed
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
C_GRID = [1, 4, 16]                 # course "minimal grid"; extended if the best C is on an edge
GAMMA_FACTORS = [0.5, 1, 2]         # gamma = factor / k  (1/k = 'scale' on k standardised features)
MAX_GRID_EXTENSIONS = 3
RF_TREES = 500                      # Random Forest kept fixed (not tuned)
RANDOM_STATE = 42

TM_COLUMNS = ("N-term TM", "Transmembrane_Helix", "TransM_Helix_90")
METRICS = ["MCC", "Precision", "Recall", "Accuracy", "F1-score", "Specificity"]
MODEL_NAMES = {"sel": "Selected features", "1se": "Selected features, one-SE rule",
               "all": "All features"}


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
    """Layout of the course slides: rows = observed class, columns = predicted
    class. Each cell is shaded by its share of the observed class (its row), so
    the rare positive class is as readable as the negatives."""
    counts = np.array([[TP, FN], [FP, TN]])
    names = np.array([["TP", "FN"], ["FP", "TN"]])
    totals = counts.sum(axis=1, keepdims=True)
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
    ax.set_xticklabels(["Predicted positive", "Predicted negative"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Observed negative", "Observed positive"], rotation=90, va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1, -0.06, "% = share of the observed class (row)", transform=ax.transAxes,
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


def aa_composition(seq, n):
    seq = seq[:n]
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


N_COMPOSITION = len(ALPHABET)          # composition features come first
D = len(FEATURE_NAMES)                  # 39


def fixed_features(seqs):
    """The 19 features whose region does not depend on n."""
    return np.array([hydrophobicity_aliphatic(s) + tm_propensity(s) +
                     secondary_structure(s) + charge(s) for s in seqs], dtype=float)


def composition_features(seqs, n):
    return np.array([aa_composition(s, n) for s in seqs], dtype=float)


def extract_features(seqs, n, fixed=None):
    """Full 39-feature matrix; composition over the first n residues."""
    seqs = list(seqs)
    if fixed is None:
        fixed = fixed_features(seqs)
    X = np.hstack([composition_features(seqs, n), fixed])
    assert X.shape[1] == D
    return X


def median_sp_length(df):
    """n = median signal peptide length of the positives in df."""
    sp = pd.to_numeric(df.loc[df["Class"] == 1, "SP cleavage"], errors="coerce").dropna()
    return int(round(sp.median()))


# ---------------------------------------------------------------------------
# 3. Model helpers
# ---------------------------------------------------------------------------

def k_grid(d=D):
    """Numbers of top-ranked features tried: the course grid {D/4, D/2, D}
    plus steps of 5 in between, so the MCC-vs-k curve can be drawn."""
    return sorted({math.ceil(d / 4), math.ceil(d / 2), d, *range(5, d, 5)})


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


def gini_ranking(X_tr, y_tr):
    """Random Forest Gini importance, fitted on training data only (RF parameters fixed)."""
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return (pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def feature_index(names):
    return [int(np.where(FEATURE_NAMES == f)[0][0]) for f in names]


def save_model(model, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def bundle_features(bundle, seqs, fixed=None):
    """Feature matrix for a saved model bundle (uses the bundle's own n and features)."""
    X = extract_features(seqs, bundle["n"], fixed)
    return X[:, feature_index(bundle["features"])]


def fit_model(df, fixed, idx_rows, k, C, gamma_factor):
    """Fit n, RF ranking, scaler and SVM on the rows idx_rows of df.
    Returns a bundle {model, n, features, k, C, gamma, ranking}."""
    sub = df.iloc[idx_rows]
    n = median_sp_length(sub)
    X = extract_features(sub["Frag_90"], n, fixed[idx_rows])
    y = sub["Class"].to_numpy()
    ranking = gini_ranking(X, y)
    names = ranking["feature"].head(k).tolist()
    idx = feature_index(names)
    gamma = gamma_factor / k
    model = svm_pipeline(C, gamma).fit(X[:, idx], y)
    return {"model": model, "n": n, "features": names, "k": k, "C": C,
            "gamma": gamma, "gamma_factor": gamma_factor, "ranking": ranking}


# ---------------------------------------------------------------------------
# 4. Model selection (inner cross-validation)
# ---------------------------------------------------------------------------

def _prepare_inner_fold(df, fixed, labels, fit_folds, val_fold):
    """n, features and RF ranking for one inner split - computed once and
    reused for every (k, C, gamma)."""
    fit = np.isin(labels, fit_folds)
    val = labels == val_fold
    n = median_sp_length(df[fit])
    X_fit = extract_features(df["Frag_90"][fit], n, fixed[fit])
    X_val = extract_features(df["Frag_90"][val], n, fixed[val])
    y_fit, y_val = df["Class"].to_numpy()[fit], df["Class"].to_numpy()[val]
    order = feature_index(gini_ranking(X_fit, y_fit)["feature"])
    return {"X_fit": X_fit, "y_fit": y_fit, "X_val": X_val, "y_val": y_val,
            "order": order, "n": n}


def _score(fold, k, C, gf):
    idx = fold["order"][:k]
    model = svm_pipeline(C, gf / k).fit(fold["X_fit"][:, idx], fold["y_fit"])
    return compute_metrics(fold["y_val"], model.predict(fold["X_val"][:, idx]))["MCC"]


def inner_search(df, fixed, folds):
    """CV over `folds` (each one is validation once). Returns the mean validation
    MCC of every (k, C, gamma_factor) tried, the per-fold MCCs (needed for the
    one-SE rule) and the n of every inner fold. The C and gamma grids are
    extended when the best value lies on their edge."""
    labels = df["Label"].to_numpy()
    prepared = [_prepare_inner_fold(df, fixed, labels, [f for f in folds if f != v], v)
                for v in folds]
    ks, Cs, gfs = k_grid(), list(C_GRID), list(GAMMA_FACTORS)
    scores, per_fold = {}, {}

    def evaluate(combos):
        todo = [c for c in combos if c not in scores]
        jobs = [(c, fold) for c in todo for fold in prepared]
        mccs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_score)(fold, *c) for c, fold in jobs)
        for i, c in enumerate(todo):
            per_fold[c] = [float(v) for v in mccs[i * len(prepared):(i + 1) * len(prepared)]]
            scores[c] = float(np.mean(per_fold[c]))

    for _ in range(MAX_GRID_EXTENSIONS + 1):
        evaluate([(k, C, g) for k in ks for C in Cs for g in gfs])
        _, C_best, g_best = max(scores, key=scores.get)
        grew = False
        if C_best == max(Cs):
            Cs.append(C_best * 4); grew = True
        elif C_best == min(Cs) and C_best > 0.1:
            Cs.insert(0, C_best / 4); grew = True
        if g_best == max(gfs):
            gfs.append(g_best * 2); grew = True
        elif g_best == min(gfs):
            gfs.insert(0, g_best / 2); grew = True
        if not grew:
            break
    return scores, per_fold, [p["n"] for p in prepared]


def best_of(scores, k=None):
    """Best (k, C, gamma_factor) by mean validation MCC, optionally with k fixed."""
    cands = {c: s for c, s in scores.items() if k is None or c[0] == k}
    best = max(cands, key=cands.get)
    return best, cands[best]


def mcc_by_k(scores):
    """For each k, the best mean validation MCC over C and gamma."""
    ks = sorted({c[0] for c in scores})
    return ks, [max(s for c, s in scores.items() if c[0] == k) for k in ks]


def one_se_choice(scores, per_fold):
    """One-standard-error rule for k (course slides, "Read the shape, not the
    peak"): take the best combination, compute the standard error of its
    validation MCC over the inner folds, and keep the smallest k whose best
    mean MCC is within one SE of the best. C and gamma are then the best ones
    for that k. Returns ((k, C, gamma_factor), mean MCC, details)."""
    best, best_mcc = best_of(scores)
    folds = np.array(per_fold[best])
    se = float(folds.std(ddof=1) / np.sqrt(len(folds)))
    ks, k_mcc = mcc_by_k(scores)
    k_1se = min(k for k, m in zip(ks, k_mcc) if m >= best_mcc - se)
    combo, mcc = best_of(scores, k=k_1se)
    return combo, mcc, {"k_best": best[0], "best_mcc": best_mcc, "se": se,
                        "cutoff": best_mcc - se}


def save_inner_scores(scores, per_fold, path):
    """Every (k, C, gamma) tried, with its mean and per-fold validation MCC."""
    rows = [{"k": k, "C": C, "gamma_factor": g, "mean_MCC": scores[(k, C, g)],
             "SE": float(np.std(per_fold[(k, C, g)], ddof=1) / np.sqrt(len(per_fold[(k, C, g)]))),
             **{f"fold{j + 1}_MCC": v for j, v in enumerate(per_fold[(k, C, g)])}}
            for (k, C, g) in sorted(scores)]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(m, title, path):
    """Rows = observed class, columns = predicted class (same layout as the VH model)."""
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


def plot_mcc_curve(ks, mccs, rule, chosen_k, k_1se, title, path):
    """MCC vs k. Filled dot = k chosen (best mean validation MCC); hollow dot =
    the k the one-SE rule would keep (smallest k on or above the dashed line)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, mccs, marker="o", ms=5, color=SERIES[0], label="best C, γ for each k")
    ax.axhline(rule["cutoff"], ls="--", color=MUTED,
               label=f"best − 1 SE = {rule['cutoff']:.3f}")
    j = ks.index(chosen_k)
    ax.scatter([chosen_k], [mccs[j]], s=110, color=SERIES[1], edgecolor=INK, zorder=5,
               label=f"chosen k = {chosen_k} (best mean MCC)")
    j = ks.index(k_1se)
    ax.scatter([k_1se], [mccs[j]], s=110, facecolors="none", edgecolors=INK, lw=1.4, zorder=6,
               label=f"one-SE rule: k = {k_1se}")
    ax.set_xlabel("k (top features by RF Gini importance)")
    ax.set_ylabel("Mean inner-validation MCC")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, path)


def plot_all_mcc_curves(results, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        color = SERIES[i % len(SERIES)]
        ax.plot(r["ks"], r["k_mcc"], marker="o", ms=4, color=color,
                label=f"Run {r['run']} (chosen k={r['sel']['k']}, one-SE k={r['1se']['k']})")
        j = r["ks"].index(r["sel"]["k"])
        ax.scatter(r["ks"][j], r["k_mcc"][j], s=90, color=color, edgecolor=INK, zorder=5)
        j = r["ks"].index(r["1se"]["k"])
        ax.scatter(r["ks"][j], r["k_mcc"][j], s=90, facecolors="none", edgecolors=color,
                   lw=1.5, zorder=5)
    ax.set_xlabel("k (top features by RF Gini importance)")
    ax.set_ylabel("Mean inner-validation MCC (best C, γ for each k)")
    ax.set_title("Inner CV: MCC vs number of selected features")
    ax.legend(title="filled dot = chosen k (best mean MCC); hollow = one-SE rule",
              title_fontsize=9, loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, path)


def selection_frequency(results, final):
    """How often each feature is selected over the 5 outer runs, with its
    Gini importance and rank averaged over the 5 outer rankings."""
    rows = []
    for f in FEATURE_NAMES:
        imps = [float(r["sel"]["ranking"].set_index("feature").loc[f, "importance"]) for r in results]
        ranks = [int(r["sel"]["ranking"].index[r["sel"]["ranking"]["feature"] == f][0]) + 1
                 for r in results]
        rows.append({"feature": f,
                     "times_selected": sum(f in r["sel"]["features"] for r in results),
                     "of_runs": len(results),
                     "in_final_model": f in final["features"],
                     "mean_importance": np.mean(imps), "sd_importance": np.std(imps, ddof=1),
                     "mean_rank": np.mean(ranks), "min_rank": min(ranks), "max_rank": max(ranks)})
    return (pd.DataFrame(rows).sort_values(["mean_importance"], ascending=False)
            .reset_index(drop=True))


def plot_selection_frequency(freq, path, top=25):
    """Mean Gini importance (± SD over the 5 outer rankings); bar colour and
    label = in how many of the 5 outer runs the feature was selected."""
    head = freq.head(top)[::-1]
    n_runs = int(freq["of_runs"].iloc[0])
    colors = [SEQ_BLUE(0.15 + 0.85 * t / n_runs) if t else "#e1e0d9"
              for t in head["times_selected"]]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.barh(head["feature"], head["mean_importance"], xerr=head["sd_importance"],
            color=colors, height=0.72, error_kw={"ecolor": INK_2, "lw": 0.8, "capsize": 2})
    for yi, (imp, sd, t) in enumerate(zip(head["mean_importance"], head["sd_importance"],
                                          head["times_selected"])):
        ax.text(imp + sd + 0.003, yi, f"{t}/{n_runs}", va="center", fontsize=8, color=INK_2)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Gini importance (mean ± SD over the 5 outer runs)")
    ax.set_title(f"Feature selection over the {n_runs} outer runs (top {top})\n"
                 "label = runs in which the feature was selected", fontsize=12)
    fig.tight_layout()
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# 6. Nested cross-validation
# ---------------------------------------------------------------------------

def run_nested_cv(train_df, fixed, rounds_dir):
    folds = sorted(int(f) for f in train_df["Label"].unique())
    labels = train_df["Label"].to_numpy()
    results = []

    for i, test_fold in enumerate(folds):
        run = i + 1
        train_folds = [f for f in folds if f != test_fold]
        print("/" * 60)
        print(f"Run {run}: test subset = {test_fold}, inner CV on subsets {train_folds}")

        scores, per_fold, inner_n = inner_search(train_df, fixed, train_folds)
        save_inner_scores(scores, per_fold, rounds_dir / f"inner_scores_round{run}.tsv")
        (k_s, C_s, g_s), val_s = best_of(scores)                      # course rule
        (k_1, C_1, g_1), val_1, rule = one_se_choice(scores, per_fold)  # variant
        (_, C_a, g_a), val_a = best_of(scores, k=D)                    # baseline
        print(f"  inner n (median SP length) = {inner_n}, combinations tried: {len(scores)}")
        print(f"  best mean validation MCC {val_s:.3f} at k={k_s}, C={C_s:g}, gamma={g_s:g}/k; "
              f"one-SE rule (SE {rule['se']:.3f}, cut-off {rule['cutoff']:.3f}) would keep k={k_1}")

        tr_rows = np.where(np.isin(labels, train_folds))[0]
        te = labels == test_fold
        y_te = train_df["Class"].to_numpy()[te]
        out = {}
        for tag, k, C, g, val in (("all", D, C_a, g_a, val_a), ("sel", k_s, C_s, g_s, val_s),
                                  ("1se", k_1, C_1, g_1, val_1)):
            b = fit_model(train_df, fixed, tr_rows, k, C, g)
            X_te = bundle_features(b, train_df["Frag_90"][te], fixed[te])
            m = compute_metrics(y_te, b["model"].predict(X_te))
            out[tag] = {**b, "val_mcc": val, "m": m}
            print(f"  {MODEL_NAMES[tag]:30s}: k={k:2d}, C={C:g}, gamma={g:g}/k, n={b['n']}, "
                  f"inner MCC {val:.3f}, test MCC {m['MCC']:.3f}")

        ks, k_mcc = mcc_by_k(scores)
        sel = out["sel"]
        sel["rule"] = rule
        sel["ranking"].to_csv(rounds_dir / f"features_gini_round{run}.tsv", sep="\t", index=False)
        save_model(out["all"], rounds_dir / f"svm_all_round{run}.pkl.gz")
        save_model(sel, rounds_dir / f"svm_selected_round{run}.pkl.gz")
        plot_top_features(sel["ranking"], f"Random Forest Gini importance (top 20, run {run})",
                          rounds_dir / f"top_features_round{run}.png")
        plot_mcc_curve(ks, k_mcc, rule, sel["k"], k_1, f"Inner CV: MCC vs k (run {run})",
                       rounds_dir / f"mcc_vs_k_round{run}.png")
        for tag, name in (("all", "all features"), ("selected", f"{sel['k']} selected features")):
            m = out["all" if tag == "all" else "sel"]["m"]
            plot_confusion_matrix(
                m, f"Confusion Matrix - {name} (run {run})\n"
                   f"Acc={m['Accuracy']:.3f}, F1={m['F1-score']:.3f}, MCC={m['MCC']:.3f}",
                rounds_dir / f"cm_{tag}_round{run}.png")

        results.append({"run": run, "test_fold": test_fold, "train_folds": train_folds,
                        "all": out["all"], "sel": sel, "1se": out["1se"], "ks": ks,
                        "k_mcc": k_mcc, "rule": rule, "n_combos": len(scores)})
    return results


def summarize_cv(results, out_dir):
    rows, summary = [], []
    for key in ("sel", "1se", "all"):
        model_name = MODEL_NAMES[key]
        for r in results:
            b = r[key]
            rows.append({"Model": model_name, "Run": r["run"], "Test fold": r["test_fold"],
                         "n (composition)": b["n"], "N features": b["k"],
                         "k at MCC peak": r["rule"]["k_best"],
                         "C": b["C"], "gamma": b["gamma"], "gamma x k": b["gamma_factor"],
                         "Inner val MCC": b["val_mcc"],
                         **{m: b["m"][m] for m in METRICS}})
        for m in METRICS:
            vals = np.array([r[key]["m"][m] for r in results])
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            summary.append({"Model": model_name, "Metric": m, "Mean": vals.mean(),
                            "Std": std, "SE": std / np.sqrt(len(vals))})
    runs_df, summary_df = pd.DataFrame(rows), pd.DataFrame(summary)
    runs_df.to_csv(out_dir / "cv_runs.tsv", sep="\t", index=False)
    summary_df.to_csv(out_dir / "cv_summary.tsv", sep="\t", index=False)
    print("/" * 60)
    print("\nNested CV, outer test folds:")
    print(runs_df.round(3).to_string(index=False))
    print("\nSummary (mean, std, SE over the 5 outer folds):")
    print(summary_df.round(3).to_string(index=False))
    return summary_df


def plot_combined_confusion(results, summary_df, out_dir):
    for key, model_name, tag in (("all", "All features", "all"),
                                 ("sel", "Selected features", "selected")):
        tot = {k: sum(r[key]["m"][k] for r in results) for k in ("TP", "FP", "FN", "TN")}
        s = summary_df[summary_df["Model"] == model_name].set_index("Metric")
        plot_confusion_matrix(
            tot, f"Combined Confusion Matrix - {model_name}\n"
                 f"MCC={s.loc['MCC', 'Mean']:.3f} ± {s.loc['MCC', 'SE']:.3f} (mean ± SE)",
            out_dir / f"confusion_matrix_combined_{tag}.png")


# ---------------------------------------------------------------------------
# 7. Final model and benchmark
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


def final_models(train_df, fixed, out_dir):
    """Course slides, "Training the two final predictors": a complete 5-fold CV
    over all hyperparameters on the whole training set (the same search as the
    inner loop), then the best combination is refitted on all of it."""
    folds = sorted(int(f) for f in train_df["Label"].unique())
    print("\n" + "/" * 60)
    print(f"Final model: {len(folds)}-fold CV on the whole training set")
    scores, per_fold, _ = inner_search(train_df, fixed, folds)
    save_inner_scores(scores, per_fold, out_dir / "inner_scores_final.tsv")
    best_combo, best_val = best_of(scores)
    se_combo, se_val, rule = one_se_choice(scores, per_fold)
    print(f"  best mean CV MCC {best_val:.3f} at k={best_combo[0]}; one-SE rule "
          f"(SE {rule['se']:.3f}, cut-off {rule['cutoff']:.3f}) would keep k={se_combo[0]}")
    all_rows = np.arange(len(train_df))
    bundles = {}
    for tag, (combo, val) in (("selected", (best_combo, best_val)),
                              ("all", best_of(scores, k=D)),
                              ("one_se", (se_combo, se_val))):
        k, C, g = combo
        b = fit_model(train_df, fixed, all_rows, k, C, g)
        b["val_mcc"] = val
        b["rule"] = rule
        bundles[tag] = b
        print(f"  {tag:8s}: k={k}, C={C:g}, gamma={g:g}/k, n={b['n']}, CV MCC {val:.3f}")
        if tag != "one_se":
            save_model(b, out_dir / f"svm_{tag}_final.pkl.gz")
    ks, k_mcc = mcc_by_k(scores)
    plot_mcc_curve(ks, k_mcc, rule, bundles["selected"]["k"], bundles["one_se"]["k"],
                   "Final model: CV MCC vs k", out_dir / "mcc_vs_k_final.png")
    plot_top_features(bundles["selected"]["ranking"],
                      "Random Forest Gini importance (top 20, final model)",
                      out_dir / "top_features_final.png")
    bundles["selected"]["ranking"].to_csv(out_dir / "features_gini_final.tsv", sep="\t", index=False)
    return bundles


def evaluate_benchmark(bundles, bench_df, out_dir):
    """Apply both final models once to the benchmark. Per-protein predictions of
    both are saved; the TP / FN / FP lists are for the selected-features model
    (the full pipeline: RF top-k -> SVM)."""
    y_true = bench_df["Class"].to_numpy()
    fixed_b = fixed_features(bench_df["Frag_90"])
    pred = bench_df.drop(columns=["Label"]).copy()
    tm_col = next((c for c in TM_COLUMNS if c in bench_df.columns), None)
    is_tm = (bench_df[tm_col].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()
             if tm_col else np.zeros(len(bench_df), bool))
    rows = []
    print("\n" + "/" * 60)
    print("Benchmark")
    for tag in ("selected", "all", "one_se"):
        b = bundles[tag]
        name = {"selected": f"Selected features ({b['k']})", "all": "All features",
                "one_se": f"Selected features, one-SE rule ({b['k']})"}[tag]
        X = bundle_features(b, bench_df["Frag_90"], fixed_b)
        y_pred = b["model"].predict(X)
        m = compute_metrics(y_true, y_pred)
        neg = y_true == 0
        fpr_tm = ((neg & is_tm & (y_pred == 1)).sum() / (neg & is_tm).sum()
                  if (neg & is_tm).any() else np.nan)
        fpr_no = ((neg & ~is_tm & (y_pred == 1)).sum() / (neg & ~is_tm).sum()
                  if (neg & ~is_tm).any() else np.nan)
        rows.append({"Model": name, "n (composition)": b["n"], "k": b["k"], "C": b["C"],
                     "gamma": b["gamma"], **m, "FPR": m["FP"] / (m["FP"] + m["TN"]),
                     "FPR N-term TM": fpr_tm, "FPR no TM": fpr_no})
        print(f"\n{name}: n={b['n']}, C={b['C']:g}, gamma={b['gamma']:.4f}")
        print(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
        print("  " + ", ".join(f"{k}={m[k]:.3f}" for k in METRICS))
        transmembrane_fpr(bench_df, y_true, y_pred)
        plot_confusion_matrix(
            m, f"Benchmark Confusion Matrix - {name}\n"
               f"Acc={m['Accuracy']:.3f}, F1={m['F1-score']:.3f}, MCC={m['MCC']:.3f}",
            out_dir / f"confusion_matrix_benchmark_{tag}.png")
        pred[f"svm_{tag}_score"] = b["model"].decision_function(X)
        pred[f"svm_{tag}_pred"] = y_pred
    pred.to_csv(out_dir / "benchmark_predictions.tsv", sep="\t", index=False)
    y_sel = pred["svm_selected_pred"].to_numpy()
    for fname, mask in (("false_negatives", (y_true == 1) & (y_sel == 0)),
                        ("true_positives", (y_true == 1) & (y_sel == 1)),
                        ("false_positives", (y_true == 0) & (y_sel == 1))):
        pred[mask].to_csv(out_dir / f"benchmark_{fname}.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "benchmark_metrics.tsv", sep="\t", index=False)


def write_selected_features(results, bundles, out_dir):
    rows = [{"Model": "Run " + str(r["run"]), "n": r["sel"]["n"], "k (chosen)": r["sel"]["k"],
             "Best mean MCC": r["rule"]["best_mcc"], "SE": r["rule"]["se"],
             "k (one-SE rule)": r["1se"]["k"],
             "Selected features": ", ".join(r["sel"]["features"])}
            for r in results]
    b = bundles["selected"]
    rows.append({"Model": "Final", "n": b["n"], "k (chosen)": b["k"],
                 "Best mean MCC": b["rule"]["best_mcc"], "SE": b["rule"]["se"],
                 "k (one-SE rule)": bundles["one_se"]["k"],
                 "Selected features": ", ".join(b["features"])})
    pd.DataFrame(rows).to_csv(out_dir / "selected_features.tsv", sep="\t", index=False)
    freq = selection_frequency(results, b)
    freq.to_csv(out_dir / "feature_selection_frequency.tsv", sep="\t", index=False)
    plot_selection_frequency(freq, out_dir / "feature_selection_frequency.png")
    print("\nSelection frequency over the outer runs (top 20 by mean Gini importance):")
    print(freq.head(20).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Nested-CV SVM signal peptide classifier.")
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
    fixed = fixed_features(train_df["Frag_90"])

    n_all = median_sp_length(train_df)
    feats = pd.DataFrame(extract_features(train_df["Frag_90"], n_all, fixed), columns=FEATURE_NAMES)
    feats.insert(0, "UniProt_ID", train_df["UniProt_ID"])
    feats.insert(1, "Class", train_df["Class"])
    feats.insert(2, "Fold", train_df["Label"])
    feats.to_csv(out_dir / "training_features.tsv", sep="\t", index=False)
    print(f"{D} features per sequence; composition over the first n = {n_all} residues "
          f"(median SP length of the whole training set; recomputed in every fit)")

    results = run_nested_cv(train_df, fixed, rounds_dir)
    summary_df = summarize_cv(results, out_dir)
    plot_combined_confusion(results, summary_df, out_dir)
    plot_all_mcc_curves(results, out_dir / "mcc_vs_k_all_rounds.png")

    bundles = final_models(train_df, fixed, out_dir)
    write_selected_features(results, bundles, out_dir)
    evaluate_benchmark(bundles, bench_df, out_dir)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
