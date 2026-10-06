import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve,
)

# ---------------------------------------------------------------------------
# Paths — adjust to match your project layout (or pass --cv-dir / --pos-fasta / --neg-fasta)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CV_DIR = PROJECT_DIR / "02_data_preparation" / "cross_validation"     # pos/neg_train/bench.tsv
POS_FASTA = PROJECT_DIR / "01_data_collection" / "positives" / "positive.fasta"
NEG_FASTA = PROJECT_DIR / "01_data_collection" / "negatives" / "negative.fasta"

OUTPUT_DIR = BASE_DIR / "model_results"

SHOW_PLOTS = False   # True: also open the summary figures in a window

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(ALPHABET)}
WINDOW_LENGTH = 15      # PSWM width (positions -13 ... +2 around the cleavage site)
SCAN_LIMIT = 90         # scan the first N residues (model_creation.py used 70, notebook 90)
PSEUDOCOUNT = 1

# Expasy SwissProt background frequencies, in ALPHABET order
BACKGROUND_FREQ = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0545, 'C': 0.0137,
    'Q': 0.0393, 'E': 0.0675, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
    'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
    'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0687,
}
BACKGROUND = np.array([BACKGROUND_FREQ[aa] for aa in ALPHABET])

TM_COLUMNS = ("Transmembrane_Helix", "TransM_Helix_90", "N-term TM")

# Residue lookup for fast scoring (unknown residues -> extra index with weight 0)
_LUT = np.full(256, len(ALPHABET), dtype=np.int64)
for _i, _aa in enumerate(ALPHABET):
    _LUT[ord(_aa)] = _i

CM_COLORS = np.array([["#fde725", "#483677"],
                      ["#2d708e", "#3cbb75"]])


def setup_style():
    """Seaborn/matplotlib theme (Liberation Serif if installed, else DejaVu Serif)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "Liberation Serif" if "Liberation Serif" in available else "DejaVu Serif"
    sns.set_theme(context="notebook", style="white", palette="viridis",
                  font=font, font_scale=1.1)
    mpl.rcParams["axes.unicode_minus"] = False


setup_style()


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

FRAG_LEN = SCAN_LIMIT          # N-terminal fragment used for scoring
SP_LEFT, SP_RIGHT = 13, 2      # window seq[c-13 : c+2] around the cleavage site c


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
    """Read one cross-validation TSV and attach sequences.

    Output columns: UniProt_ID, Class (1 = signal peptide, 0 = negative),
    Frag_90, SP_15, Label (CV fold, 0 if the file has no 'fold' column), + metadata.
    """
    df = pd.read_csv(tsv, sep="\t")
    if "seq_id" not in df.columns:
        raise ValueError(f"{tsv} needs a 'seq_id' column (found {list(df.columns)})")
    df["seq_id"] = df["seq_id"].astype(str).str.strip()

    missing = ~df["seq_id"].isin(fasta_seqs)
    if missing.any():
        print(f"  Warning: {missing.sum()} IDs in {Path(tsv).name} not found in the FASTA - dropped")
        df = df[~missing]
    seqs = df["seq_id"].map(fasta_seqs)

    out = pd.DataFrame({
        "UniProt_ID": df["seq_id"].to_numpy(),
        "Class": class_value,
        "Frag_90": seqs.str[:FRAG_LEN].to_numpy(),
    })

    if class_value == 1:
        if "SP cleavage" not in df.columns:
            raise ValueError(f"{tsv} needs an 'SP cleavage' column")
        cleavage = pd.to_numeric(df["SP cleavage"], errors="coerce").fillna(0).astype(int)
        out["SP_15"] = [s[c - SP_LEFT:c + SP_RIGHT] if c >= SP_LEFT else ""
                        for s, c in zip(seqs, cleavage)]
    else:
        out["SP_15"] = ""

    out["Label"] = df["fold"].astype(int).to_numpy() if "fold" in df.columns else 0

    for col in ("Organism", "Kingdom", "Length", "SP cleavage", "N-term TM"):
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


def load_data(cv_dir, pos_fasta, neg_fasta):
    """Return (train_df, bench_df) built from the cross-validation TSVs + FASTA files.

    Label 1..N = CV folds (training set), Label 0 = benchmark set.
    """
    cv_dir = Path(cv_dir)
    for f in ("pos_train", "neg_train", "pos_bench", "neg_bench"):
        if not (cv_dir / f"{f}.tsv").exists():
            raise FileNotFoundError(f"{cv_dir / (f + '.tsv')} not found (use --cv-dir)")
    for f in (pos_fasta, neg_fasta):
        if not Path(f).exists():
            raise FileNotFoundError(f"{f} not found (use --pos-fasta / --neg-fasta)")

    pos_seqs, neg_seqs = load_fasta(pos_fasta), load_fasta(neg_fasta)

    train_df = pd.concat([
        _build_table(cv_dir / "pos_train.tsv", pos_seqs, 1),
        _build_table(cv_dir / "neg_train.tsv", neg_seqs, 0),
    ], ignore_index=True)
    bench_df = pd.concat([
        _build_table(cv_dir / "pos_bench.tsv", pos_seqs, 1),
        _build_table(cv_dir / "neg_bench.tsv", neg_seqs, 0),
    ], ignore_index=True)

    if (train_df["Label"] <= 0).any():
        raise ValueError("Training files need a 'fold' column with values 1..N")

    overlap = set(train_df["UniProt_ID"]) & set(bench_df["UniProt_ID"])
    if overlap:
        print(f"  WARNING: {len(overlap)} IDs appear in both the CV folds and the "
              f"benchmark set (data leakage), e.g. {sorted(overlap)[:3]}")
    return train_df, bench_df


def get_motifs(df):
    """15-mers around the cleavage site of the positive rows."""
    motifs = df.loc[df["Class"] == 1, "SP_15"]
    good = motifs[motifs.str.len() == WINDOW_LENGTH]
    skipped = len(motifs) - len(good)
    if skipped:
        print(f"  Warning: skipped {skipped} motifs not of length {WINDOW_LENGTH}")
    return good.tolist()


# ---------------------------------------------------------------------------
# 2. PSWM construction and scoring
# ---------------------------------------------------------------------------

def build_pswm(motifs, pseudocount=PSEUDOCOUNT):
    """Position-specific weight matrix (len(ALPHABET) x WINDOW_LENGTH), log2 odds."""
    counts = np.full((len(ALPHABET), WINDOW_LENGTH), float(pseudocount))
    for motif in motifs:
        for j, aa in enumerate(motif):
            i = AA_INDEX.get(aa)
            if i is not None:
                counts[i, j] += 1
    # per-column normalisation: stays correct when a motif has a non-standard residue
    freqs = counts / counts.sum(axis=0, keepdims=True)
    return np.log2(freqs / BACKGROUND[:, None])


def best_window_score(seq, W_padded, window=WINDOW_LENGTH, limit=SCAN_LIMIT):
    """Highest PSWM score over all windows in the first `limit` residues."""
    idx = _LUT[np.frombuffer(seq[:limit].encode("ascii", "ignore"), dtype=np.uint8)]
    if len(idx) < window:
        return 0.0   # sequence shorter than the window: neutral score
    windows = sliding_window_view(idx, window)
    return float(W_padded[windows, np.arange(window)].sum(axis=1).max())


def score_sequences(seqs, W):
    """Score every sequence in `seqs` with PSWM W."""
    W_padded = np.vstack([W, np.zeros((1, W.shape[1]))])   # row for unknown residues
    return np.array([best_window_score(s, W_padded, W.shape[1]) for s in seqs])


# ---------------------------------------------------------------------------
# 3. Metrics and threshold
# ---------------------------------------------------------------------------

def safe_div(num, den):
    return num / den if den else 0.0


def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    return TP, FP, FN, TN


def compute_metrics(TP, FP, FN, TN):
    recall = safe_div(TP, TP + FN)
    precision = safe_div(TP, TP + FP)
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": safe_div(TP + TN, TP + TN + FP + FN),
        "recall": recall,
        "specificity": safe_div(TN, TN + FP),
        "precision": precision,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "mcc": safe_div(TP * TN - FP * FN, denom),
        "fpr": safe_div(FP, FP + TN),
    }


def select_threshold(y_true, scores):
    """Threshold with the best F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-15)
    best_index = int(np.argmax(fscore[:-1]))   # last PR point has no threshold
    return thresholds[best_index], precision, recall, fscore, best_index


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def save_figure(fig, path, show=False):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pswm_heatmap(W, title, path, cbar_label="Score (log2 ratio)", show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(W, cmap="viridis", annot=True, fmt=".2f", annot_kws={"size": 7},
                yticklabels=list(ALPHABET), xticklabels=range(1, W.shape[1] + 1),
                cbar_kws={"label": cbar_label}, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"Position ({W.shape[1]}-mer window)", fontsize=12)
    ax.set_ylabel("Amino Acid", fontsize=12)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, path, show)


def plot_confusion_matrix(TP, FP, FN, TN, title, path, show=False):
    """Rows = predicted class, columns = actual class."""
    plot_cm = np.array([[TP, FP], [FN, TN]])
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=CM_COLORS[r, c], linewidth=0))
            text_color = "black" if (r, c) == (0, 0) else "white"
            ax.text(c + 0.5, 1.5 - r, plot_cm[r, c], ha="center", va="center",
                    fontsize=12, color=text_color)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Actual Positive", "Actual Negative"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Predicted Negative", "Predicted Positive"], rotation=90, va="center")
    ax.set_title(title, fontsize=14, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, path, show)


def plot_pr_round(result, path):
    """Validation-fold precision-recall curve of one round, optimal point marked."""
    idx = result["val_best_index"]
    prec, rec = result["val_precision"], result["val_recall"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color=cm.viridis(0.5), lw=2, label="Precision-Recall Curve")
    ax.scatter(rec[idx], prec[idx], color="yellow", s=100, zorder=5,
               label=f"Optimal F1={result['val_f1']:.3f}\nThreshold={result['threshold']:.2f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve (Round {result['run']}, validation fold)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def plot_round_figures(results, rounds_dir):
    """Per round: PSWM heatmap, validation PR curve, test-fold confusion matrix."""
    for r in results:
        n = r["run"]
        plot_pswm_heatmap(r["W"], f"Position-Specific Weight Matrix (PSWM {n})",
                          rounds_dir / f"pswm_round{n}.png")
        plot_pr_round(r, rounds_dir / f"pr_round{n}.png")
        plot_confusion_matrix(
            r["TP"], r["FP"], r["FN"], r["TN"],
            f"Confusion Matrix (Round {n})\nAcc={r['accuracy']:.3f}, "
            f"F1={r['f1']:.3f}, MCC={r['mcc']:.3f}",
            rounds_dir / f"cm_round{n}.png")


def plot_combined_metrics(results, out_dir, show=False):
    """Pooled ROC | pooled PR | average PSWM of all rounds."""
    all_y = np.concatenate([r["y_test"] for r in results])
    all_scores = np.concatenate([r["test_scores"] for r in results])

    fpr_curve, tpr_curve, _ = roc_curve(all_y, all_scores)
    roc_auc = auc(fpr_curve, tpr_curve)
    precision_curve, recall_curve, _ = precision_recall_curve(all_y, all_scores)
    avg_prec = average_precision_score(all_y, all_scores)
    W_avg = np.mean([r["W"] for r in results], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(recall_curve, precision_curve, color="blue", lw=2, label=f"AP = {avg_prec:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    sns.heatmap(W_avg, cmap="viridis", annot=True, fmt=".2f", annot_kws={"size": 6},
                yticklabels=list(ALPHABET), xticklabels=range(1, W_avg.shape[1] + 1),
                cbar_kws={"label": "Average score (log2 ratio)"}, ax=axes[2])
    axes[2].set_xlabel("Motif Position")
    axes[2].set_ylabel("Amino Acid")
    axes[2].tick_params(axis="y", rotation=0)
    axes[2].set_title(f"Average PSWM ({len(results)} rounds)")

    fig.tight_layout()
    save_figure(fig, out_dir / "combined_model_metrics.png", show)


def plot_pr_all_rounds(results, out_dir, show=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap("viridis")
    for i, r in enumerate(results):
        idx = r["val_best_index"]
        color = cmap(i / len(results))
        ax.plot(r["val_recall"], r["val_precision"], color=color, lw=2,
                label=f"Round {r['run']} (F1={r['val_f1']:.3f})")
        ax.scatter(r["val_recall"][idx], r["val_precision"][idx], color=color,
                   s=80, edgecolor="black", zorder=5)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (validation folds)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "pr_curves_all_rounds.png", show)


def plot_combined_confusion(results, summary_df, out_dir, show=False):
    TP, FP, FN, TN = (sum(r[k] for r in results) for k in ("TP", "FP", "FN", "TN"))
    print(f"Combined confusion matrix: TP={TP} FP={FP} FN={FN} TN={TN}")
    mean = summary_df.set_index("Metric")["Mean"]
    plot_confusion_matrix(
        TP, FP, FN, TN,
        f"Combined Confusion Matrix\nAvg Acc={mean['Accuracy']:.3f}, "
        f"F1={mean['F1-score']:.3f}, MCC={mean['MCC']:.3f}",
        out_dir / "confusion_matrix_combined.png", show)


# ---------------------------------------------------------------------------
# 5. Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(train_df):
    """Rotating folds: test = fold i, validation = next fold, training = the rest.
    The decision threshold is chosen on the validation fold (best F1)."""
    folds = sorted(int(f) for f in train_df["Label"].unique())
    print("Total folds:", len(folds))

    results = []
    for i, test_fold in enumerate(folds):
        val_fold = folds[(i + 1) % len(folds)]
        train_folds = [f for f in folds if f not in (test_fold, val_fold)]

        print("/" * 50)
        print(f"Run {i + 1}: Test fold = {test_fold}, Validation fold = {val_fold}, "
              f"Train folds = {train_folds}")

        motifs = get_motifs(train_df[train_df["Label"].isin(train_folds)])
        print("Training motif count:", len(motifs))
        if len(motifs) == 0:
            continue
        W = build_pswm(motifs)

        val_df = train_df[train_df["Label"] == val_fold]
        y_val = val_df["Class"].to_numpy()
        val_scores = score_sequences(val_df["Frag_90"], W)
        threshold, val_prec, val_rec, val_fscore, best_index = select_threshold(y_val, val_scores)
        print("Threshold (from validation):", threshold)

        test_df = train_df[train_df["Label"] == test_fold]
        y_test = test_df["Class"].to_numpy()
        test_scores = score_sequences(test_df["Frag_90"], W)
        y_pred = (test_scores >= threshold).astype(int)

        TP, FP, FN, TN = confusion_counts(y_test, y_pred)
        m = compute_metrics(TP, FP, FN, TN)
        print("Confusion matrix (test fold):")
        print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
        for name in ("accuracy", "recall", "specificity", "precision", "f1", "mcc"):
            print(f"{name}: {m[name]:.4f}")

        results.append({
            "run": i + 1, "test_fold": test_fold, "val_fold": val_fold,
            "train_folds": train_folds, "W": W, "threshold": threshold,
            "val_precision": val_prec, "val_recall": val_rec,
            "val_best_index": best_index, "val_f1": val_fscore[best_index],
            "y_test": y_test, "test_scores": test_scores, **m,
        })

    return results


def summarize_cv(results, out_dir):
    """Per-run table and mean / std / standard error of every metric."""
    metric_keys = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
                   "F1-score": "f1", "MCC": "mcc", "Threshold": "threshold"}

    runs_df = pd.DataFrame(
        [{"Run": r["run"], "Test fold": r["test_fold"], "Val fold": r["val_fold"],
          **{label: r[key] for label, key in metric_keys.items()}} for r in results]
    )

    rows = []
    for label, key in metric_keys.items():
        values = np.array([r[key] for r in results], dtype=float)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append({"Metric": label, "Mean": values.mean(), "Std": std,
                     "SE": std / np.sqrt(len(values))})
    summary_df = pd.DataFrame(rows)

    print("/" * 50)
    print("\nCross-validation runs:")
    print(runs_df.to_string(index=False))
    print("\nCross-validation average performance (mean, std, SE):")
    print(summary_df.to_string(index=False))

    runs_df.to_csv(out_dir / "cv_runs.tsv", sep="\t", index=False)
    summary_df.to_csv(out_dir / "cv_summary.tsv", sep="\t", index=False)
    return runs_df, summary_df


# ---------------------------------------------------------------------------
# 6. Benchmark evaluation
# ---------------------------------------------------------------------------

def export_benchmark_hits(bench_df, scores, y_true, y_pred, out_dir):
    """Save false negatives / true positives (+ their cleavage-site motifs)."""
    masks = {
        "false_negatives": (y_true == 1) & (y_pred == 0),
        "true_positives": (y_true == 1) & (y_pred == 1),
    }
    for name, mask in masks.items():
        rows = bench_df[mask].copy()
        rows["score"] = scores[mask]
        rows["y_true"] = y_true[mask]
        rows["y_pred"] = y_pred[mask]
        rows.to_csv(out_dir / f"benchmark_{name}.tsv", sep="\t", index=False)

        motifs = get_motifs(rows)   # SP_15 == seq[c-13:c+2], same window as the motif logos
        pd.DataFrame({"Motif": motifs}).to_csv(
            out_dir / f"motifs_{name}.tsv", sep="\t", index=False)
        print(f"Saved {len(rows)} {name.replace('_', ' ')} "
              f"(+ {len(motifs)} motifs) to {out_dir}")


def transmembrane_fpr(bench_df, y_true, y_pred):
    tm_col = next((c for c in TM_COLUMNS if c in bench_df.columns), None)
    if tm_col is None:
        print(f"\nNo transmembrane column found (looked for {TM_COLUMNS}) - "
              "skipping transmembrane-specific FPRs.")
        return
    is_tm = bench_df[tm_col].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()
    neg = y_true == 0
    fp = neg & (y_pred == 1)

    def fpr(mask):
        n = neg[mask].sum()
        return fp[mask].sum() / n if n > 0 else float("nan")

    print("\nTransmembrane-specific FPRs:")
    print(f"  FPR (Transmembrane)     = {fpr(is_tm) * 100:.2f}%")
    print(f"  FPR (Non-Transmembrane) = {fpr(~is_tm) * 100:.2f}%")


def refit_and_evaluate_benchmark(train_df, bench_df, best, out_dir, show=False):
    """Refit the PSWM of the best CV run on its train + validation folds (test
    fold left out) and evaluate on the benchmark with that run's threshold."""
    retrain_folds = list(best["train_folds"]) + [best["val_fold"]]
    motifs = get_motifs(train_df[train_df["Label"].isin(retrain_folds)])

    print("\n" + "/" * 50)
    print("Refitting best model on folds:", retrain_folds,
          "(excluding test fold", best["test_fold"], ")")
    print("Total motifs used for refit:", len(motifs))
    if not motifs:
        print("No motifs found for refit. Skipping benchmark evaluation.")
        return

    W_refit = build_pswm(motifs)
    pd.DataFrame(W_refit, index=list(ALPHABET), columns=range(1, WINDOW_LENGTH + 1)
                 ).to_csv(out_dir / "pswm_final.tsv", sep="\t")
    plot_pswm_heatmap(W_refit, "Final PSWM (refit on training + validation folds)",
                      out_dir / "pswm_final.png", show=show)

    y_true = bench_df["Class"].to_numpy()
    scores = score_sequences(bench_df["Frag_90"], W_refit)
    thr = best["threshold"]
    y_pred = (scores >= thr).astype(int)

    TP, FP, FN, TN = confusion_counts(y_true, y_pred)
    m = compute_metrics(TP, FP, FN, TN)

    print("\nBenchmark evaluation:")
    print("Threshold used:", thr)
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    for label, key in [("Accuracy", "accuracy"), ("Recall (Sensitivity)", "recall"),
                       ("Specificity", "specificity"), ("Precision", "precision"),
                       ("F1 Score", "f1"), ("MCC", "mcc"), ("FPR", "fpr")]:
        print(f"{label}: {m[key]:.4f}")

    transmembrane_fpr(bench_df, y_true, y_pred)
    export_benchmark_hits(bench_df, scores, y_true, y_pred, out_dir)
    plot_confusion_matrix(
        TP, FP, FN, TN,
        f"Benchmark Confusion Matrix\nAcc={m['accuracy']:.3f}, "
        f"F1={m['f1']:.3f}, MCC={m['mcc']:.3f}",
        out_dir / "confusion_matrix_benchmark.png", show)
    print("/" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
    
def parse_args():
    ap = argparse.ArgumentParser(description="Cross-validate the von Heijne model and evaluate it on the benchmark set.")
    ap.add_argument("--cv-dir", type=Path, default=CV_DIR,
                    help=f"folder with pos/neg_train/bench.tsv (default: {CV_DIR})")
    ap.add_argument("--pos-fasta", type=Path, default=POS_FASTA,
                    help=f"positive FASTA (default: {POS_FASTA})")
    ap.add_argument("--neg-fasta", type=Path, default=NEG_FASTA,
                    help=f"negative FASTA (default: {NEG_FASTA})")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR,
                    help=f"output folder (default: {OUTPUT_DIR})")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.output
    rounds_dir = out_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    train_df, bench_df = load_data(args.cv_dir, args.pos_fasta, args.neg_fasta)
    print(f"Training rows: {len(train_df)}  |  benchmark rows: {len(bench_df)}")

    # 2. Cross-validation
    results = run_cross_validation(train_df)
    if not results:
        raise SystemExit("No cross-validation run could be trained.")
    best = max(results, key=lambda r: r["mcc"])
    print("/" * 50)
    print("Best fold for test:", best["test_fold"], "Run number:", best["run"],
          "with MCC =", round(best["mcc"], 4))

    _, summary_df = summarize_cv(results, out_dir)

    # 3. Plots
    plot_round_figures(results, rounds_dir)
    plot_combined_metrics(results, out_dir, show=SHOW_PLOTS)
    plot_pr_all_rounds(results, out_dir, show=SHOW_PLOTS)
    plot_combined_confusion(results, summary_df, out_dir, show=SHOW_PLOTS)

    # 4. Benchmark
    if len(bench_df):
        refit_and_evaluate_benchmark(train_df, bench_df, best, out_dir, show=SHOW_PLOTS)
    else:
        print("\nNo benchmark data — "
              "skipping benchmark evaluation.")

    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
