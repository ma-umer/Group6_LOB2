"""
Extra report figures (recommended in the course-slide checklist, not produced by
the main scripts). Nothing is retrained: the script reads the saved outputs of
04_von_heijne and 05_svm, and reuses the data loader, scoring and feature code
of the two model scripts so every number matches them.

Figures written (each next to the results of the step it belongs to):
  02_data_preparation/data_split_diagram.png
      training folds -> validation -> hold-out benchmark, and how each final
      model uses them
  04_von_heijne/model_results/score_distribution.png
      PSWM score of positives vs negatives (pooled CV test folds and benchmark),
      with the threshold
  05_svm/model_results/inner_mcc_heatmap.png
      inner-CV mean MCC over C x gamma at the chosen k (5 outer runs + final)
  06_comparison/comparison_results/cv_metrics_comparison.png
      cross-validation metrics, von Heijne vs SVM, mean +- SE over the 5 test folds
  06_comparison/comparison_results/fp_hydrophobicity_examples.png
      hydrophobicity profiles of example false positives vs the typical SP profile

    python 06_comparison/extra_figures.py        # about 30 s
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VH_DIR = PROJECT_DIR / "04_von_heijne"
SVM_DIR = PROJECT_DIR / "05_svm"
PREP_DIR = PROJECT_DIR / "02_data_preparation"
CMP_DIR = BASE_DIR / "comparison_results"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vh = load_module("vh_model", VH_DIR / "von_heijne_method.py")
svm = load_module("svm_model", SVM_DIR / "svm_model.py")

SERIES, INK, INK_2, MUTED = vh.SERIES, vh.INK, vh.INK_2, vh.MUTED
VH_COL, SVM_COL = SERIES[0], SERIES[1]
POS_COL, NEG_COL, TM_COL, FP_COL = SERIES[0], vh.MUTED, INK, SERIES[7]


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(PROJECT_DIR))


def tm_flag(df):
    return df["N-term TM"].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()


def final_pswm():
    W = pd.read_csv(VH_DIR / "model_results" / "pswm_final.tsv", sep="\t", index_col=0)
    return W.loc[list(vh.ALPHABET)].to_numpy()


def final_threshold():
    t = pd.read_csv(VH_DIR / "model_results" / "final_model.tsv", sep="\t",
                    index_col="Parameter")["Value"]
    return float(t["threshold"])


# ---------------------------------------------------------------------------
# 1. von Heijne score distributions
# ---------------------------------------------------------------------------

def plot_score_distribution(train, bench):
    with contextlib.redirect_stdout(io.StringIO()):
        results = vh.run_cross_validation(train)
    cv_y = np.concatenate([r["y_test"] for r in results])
    cv_s = np.concatenate([r["test_scores"] for r in results])
    thr_cv = np.array([r["threshold"] for r in results])

    W, thr = final_pswm(), final_threshold()
    b_s = vh.score_sequences(bench["Frag_90"], W)
    b_y = bench["Class"].to_numpy()
    b_tm = tm_flag(bench)

    lo = np.floor(min(cv_s.min(), b_s.min()))
    hi = np.ceil(max(cv_s.max(), b_s.max()))
    bins = np.arange(lo, hi + 1, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    ax = axes[0]
    ax.hist(cv_s[cv_y == 0], bins=bins, density=True, color=NEG_COL, alpha=0.55,
            label=f"Negatives (n = {(cv_y == 0).sum():,})")
    ax.hist(cv_s[cv_y == 1], bins=bins, density=True, color=POS_COL, alpha=0.6,
            label=f"Positives (n = {(cv_y == 1).sum():,})")
    ax.axvspan(thr_cv.min(), thr_cv.max(), color=SERIES[3], alpha=0.25, lw=0,
               label=f"θ range of the 5 runs ({thr_cv.min():.2f}–{thr_cv.max():.2f})")
    ax.axvline(thr_cv.mean(), color=INK, ls="--", lw=1.4,
               label=f"mean θ = {thr_cv.mean():.2f}")
    ax.set_title("Cross-validation: 5 test folds pooled")

    ax = axes[1]
    ax.hist(b_s[(b_y == 0) & ~b_tm], bins=bins, density=True, color=NEG_COL, alpha=0.55,
            label=f"Negatives, no N-term TM (n = {((b_y == 0) & ~b_tm).sum():,})")
    ax.hist(b_s[(b_y == 0) & b_tm], bins=bins, density=True, histtype="step", lw=2,
            color=TM_COL, label=f"Negatives with N-term TM (n = {((b_y == 0) & b_tm).sum():,})")
    ax.hist(b_s[b_y == 1], bins=bins, density=True, color=POS_COL, alpha=0.6,
            label=f"Positives (n = {(b_y == 1).sum():,})")
    ax.axvline(thr, color=INK, ls="--", lw=1.4, label=f"final θ = {thr:.2f}")
    ax.set_title("Benchmark: final PSWM")

    for ax in axes:
        ax.set_xlabel("PSWM score (best 15-residue window in the first 90 residues)")
        ax.grid(axis="y")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("Density (each class sums to 1)")
    fig.suptitle("von Heijne score distribution: signal peptides vs non-signal peptides",
                 y=1.02)
    save(fig, VH_DIR / "model_results" / "score_distribution.png")

    pd.DataFrame({
        "Set": ["CV pooled"] * 2 + ["Benchmark"] * 3,
        "Group": ["Positives", "Negatives", "Positives", "Negatives, no N-term TM",
                  "Negatives with N-term TM"],
        "n": [(cv_y == 1).sum(), (cv_y == 0).sum(), (b_y == 1).sum(),
              ((b_y == 0) & ~b_tm).sum(), ((b_y == 0) & b_tm).sum()],
        "Median score": [np.median(cv_s[cv_y == 1]), np.median(cv_s[cv_y == 0]),
                         np.median(b_s[b_y == 1]), np.median(b_s[(b_y == 0) & ~b_tm]),
                         np.median(b_s[(b_y == 0) & b_tm])],
    }).to_csv(VH_DIR / "model_results" / "score_distribution.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# 2. SVM inner-CV MCC heatmaps (C x gamma at the chosen k)
# ---------------------------------------------------------------------------

def plot_inner_heatmaps():
    chosen = pd.read_csv(SVM_DIR / "model_results" / "selected_features.tsv", sep="\t")
    k_of = dict(zip(chosen["Model"], chosen["k (chosen)"]))
    panels = [(f"Outer run {r}", SVM_DIR / "model_results" / "rounds" /
               f"inner_scores_round{r}.tsv", k_of[f"Run {r}"]) for r in range(1, 6)]
    panels.append(("Final model (5-fold CV, whole training set)",
                   SVM_DIR / "model_results" / "inner_scores_final.tsv", k_of["Final"]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    rows = []
    for ax, (title, path, k) in zip(axes.flat, panels):
        s = pd.read_csv(path, sep="\t")
        s = s[s["k"] == k]
        grid = s.pivot_table(index="C", columns="gamma_factor", values="mean_MCC")
        grid = grid.sort_index(ascending=False).sort_index(axis=1)
        vals = grid.to_numpy()
        best = np.unravel_index(np.nanargmax(vals), vals.shape)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        ax.imshow(np.ma.masked_invalid(vals), cmap=vh.SEQ_BLUE, vmin=vmin, vmax=vmax,
                  aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color=MUTED)
                    continue
                f = (v - vmin) / (vmax - vmin) if vmax > vmin else 1
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if f > 0.55 else INK,
                        fontweight="bold" if (i, j) == best else "normal")
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                                   ec=SERIES[7], lw=2.5))
        ax.set_xticks(range(vals.shape[1]))
        ax.set_xticklabels([f"{g:g}/k" for g in grid.columns])
        ax.set_yticks(range(vals.shape[0]))
        ax.set_yticklabels([f"{c:g}" for c in grid.index])
        ax.set_xlabel("γ")
        ax.set_ylabel("C")
        ax.grid(False)
        bC, bg = grid.index[best[0]], grid.columns[best[1]]
        edge = []
        if best[0] in (0, vals.shape[0] - 1):
            edge.append("C")
        if best[1] in (0, vals.shape[1] - 1):
            edge.append("γ")
        ax.set_title(f"{title}\nk = {k}; best C = {bC:g}, γ = {bg:g}/k"
                     + (f"  (on {'/'.join(edge)} edge)" if edge else ""), fontsize=10.5)
        rows.append({"Panel": title, "k": k, "best C": bC, "best gamma x k": bg,
                     "best mean MCC": vals[best], "MCC range in grid": vmax - vmin,
                     "best on grid edge": ", ".join(edge) or "no"})
    fig.suptitle("Inner cross-validation: mean validation MCC over C × γ at the chosen k\n"
                 "(red box = selected combination; colour scale per panel)", y=1.0)
    fig.tight_layout()
    save(fig, SVM_DIR / "model_results" / "inner_mcc_heatmap.png")
    pd.DataFrame(rows).to_csv(SVM_DIR / "model_results" / "inner_mcc_heatmap.tsv",
                              sep="\t", index=False)


# ---------------------------------------------------------------------------
# 3. CV metrics, von Heijne vs SVM (mean +- SE)
# ---------------------------------------------------------------------------

def plot_cv_comparison():
    metrics = ["MCC", "Precision", "Recall", "F1-score", "Accuracy"]
    vs = pd.read_csv(VH_DIR / "model_results" / "cv_summary.tsv", sep="\t").set_index("Metric")
    vr = pd.read_csv(VH_DIR / "model_results" / "cv_runs.tsv", sep="\t")
    ss = pd.read_csv(SVM_DIR / "model_results" / "cv_summary.tsv", sep="\t")
    ss = ss[ss["Model"] == "Selected features"].set_index("Metric")
    sr = pd.read_csv(SVM_DIR / "model_results" / "cv_runs.tsv", sep="\t")
    sr = sr[sr["Model"] == "Selected features"]

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for off, name, summ, runs, col in [(-w / 2, "von Heijne", vs, vr, VH_COL),
                                        (w / 2, "SVM", ss, sr, SVM_COL)]:
        m = summ.loc[metrics, "Mean"].to_numpy()
        se = summ.loc[metrics, "SE"].to_numpy()
        ax.bar(x + off, m, w * 0.95, yerr=se, color=col, label=name, capsize=4,
               error_kw={"elinewidth": 1.4, "ecolor": INK})
        for i, met in enumerate(metrics):
            pts = runs[met].to_numpy()
            ax.scatter(np.full(len(pts), x[i] + off) + np.linspace(-0.06, 0.06, len(pts)),
                       pts, s=12, color="white", edgecolor=INK, lw=0.7, zorder=3)
            ax.text(x[i] + off, 0.03, f"{m[i]:.3f}\n±{se[i]:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")
            rows.append({"Model": name, "Metric": met, "Mean": m[i], "SE": se[i]})
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score on the test folds")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    ax.set_title("Cross-validation performance (bar = mean of 5 test folds, "
                 "error bar = SE = SD/√5, dots = folds)", fontsize=11)
    save(fig, CMP_DIR / "cv_metrics_comparison.png")
    pd.DataFrame(rows).to_csv(CMP_DIR / "cv_metrics_comparison.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# 4. Data-split diagram
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc, ec=None, fs=9.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4,
                                shrinkA=2, shrinkB=2))


def plot_data_split(train, bench):
    fold_counts = (train.groupby(["Label", "Class"]).size().unstack(fill_value=0)
                   .rename(columns={0: "neg", 1: "pos"}))
    folds = list(fold_counts.index)
    n_pos_b, n_neg_b = (bench["Class"] == 1).sum(), (bench["Class"] == 0).sum()
    n_pos_t, n_neg_t = (train["Class"] == 1).sum(), (train["Class"] == 0).sum()
    pos_all = sum(1 for _ in open(vh.POS_FASTA) if _.startswith(">"))
    neg_all = sum(1 for _ in open(vh.NEG_FASTA) if _.startswith(">"))
    thr = final_threshold()
    bm = pd.read_csv(SVM_DIR / "model_results" / "benchmark_metrics.tsv", sep="\t").iloc[0]
    b = {"k": int(bm["k"]), "C": float(bm["C"]), "gamma": float(bm["gamma"]),
         "n": int(bm["n (composition)"])}

    LIGHT, BLUE, ORANGE, GREEN = "#eceae4", "#d7e6fa", "#fbe1d4", "#d3f0e4"
    fig, ax = plt.subplots(figsize=(14, 8.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.9)
    ax.axis("off")

    # data preparation row
    box(ax, 0.2, 6.75, 2.6, 1.2, f"UniProtKB/Swiss-Prot\nreviewed eukaryotes\n"
        f"{pos_all:,} SP+  ·  {neg_all:,} SP−", LIGHT)
    box(ax, 3.4, 6.75, 2.6, 1.2, "MMseqs2 clustering\n30 % identity, 40 % coverage\n"
        f"{n_pos_t + n_pos_b:,} SP+  ·  {n_neg_t + n_neg_b:,} SP−", LIGHT)
    arrow(ax, 2.8, 7.35, 3.4, 7.35)
    box(ax, 6.6, 7.2, 4.9, 0.75, f"Training set (80 %): {n_pos_t:,} SP+  ·  {n_neg_t:,} SP−",
        GREEN, weight="bold")
    box(ax, 11.9, 6.35, 1.9, 1.6, f"Benchmark\n(20 %, hold-out)\n{n_pos_b} SP+\n"
        f"{n_neg_b:,} SP−\nnever used for\nany choice", ORANGE, ec=SERIES[1], fs=9)
    arrow(ax, 6.0, 7.35, 6.6, 7.55)
    ax.annotate("", xy=(12.85, 7.97), xytext=(4.7, 7.97),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4, shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=-0.12"))
    fw = 4.9 / len(folds)
    for i, f in enumerate(folds):
        box(ax, 6.6 + i * fw + 0.04, 6.35, fw - 0.08, 0.7,
            f"fold {f}\n{fold_counts.loc[f, 'pos']} / {fold_counts.loc[f, 'neg']:,}",
            GREEN, fs=8.5)
    ax.text(9.05, 6.2, "SP+ / SP− per fold", ha="center", va="top", fontsize=8, color=MUTED)

    def lane(y, title, steps, col, arrows=True):
        ax.text(0.2, y + 0.95, title, fontsize=11.5, fontweight="bold", color=col)
        xs = [0.2, 3.65, 7.1, 10.55]
        for i, (txt, fc) in enumerate(steps):
            box(ax, xs[i], y, 3.1, 0.8, txt, fc, ec=SERIES[1] if fc == ORANGE else None, fs=9)
            if i and arrows:
                arrow(ax, xs[i - 1] + 3.1, y + 0.4, xs[i], y + 0.4)

    lane(4.3, "Cross-validation estimates (reported as mean ± SE over 5 test folds)", [
        ("von Heijne, run j: test fold j,\nvalidation fold j+1 (θ, best F1),\n"
         "PSWM on the other 3", GREEN),
        ("SVM, run j: test fold j; nested\n4-fold CV on the other 4 picks\nk, C, γ (best mean MCC)",
         GREEN),
        ("each fold is the test set once\n→ 5 test scores per metric", LIGHT),
        ("benchmark untouched", LIGHT)], INK, arrows=False)
    lane(2.45, "Final von Heijne model", [
        ("PSWM on folds 1–4\n(training subsets)", GREEN),
        (f"θ fixed on validation fold 5\nbest F1 → θ = {thr:.2f}", GREEN),
        ("PSWM rebuilt on all 5 folds\nθ kept", GREEN),
        ("one test on the benchmark", ORANGE)], VH_COL)
    lane(0.6, "Final SVM", [
        ("5-fold CV on all 5 folds\nover k, C, γ\n(best mean validation MCC)", GREEN),
        (f"best combination:\nk = {b['k']}, C = {b['C']:g}, γ = {b['gamma'] * b['k']:g}/k\n"
         f"n = {b['n']} (composition)", GREEN),
        ("retrained on all 5 folds", GREEN),
        ("one test on the benchmark", ORANGE)], SVM_COL)
    ax.text(14, 0.05, "green = training data  ·  orange = hold-out benchmark",
            ha="right", fontsize=8.5, color=MUTED)
    save(fig, PREP_DIR / "data_split_diagram.png")


# ---------------------------------------------------------------------------
# 5. Example false positives: hydrophobicity profiles
# ---------------------------------------------------------------------------

def best_window_start(seq, W):
    idx = np.array([vh.AA_INDEX.get(a, -1) for a in seq[:vh.SCAN_LIMIT]])
    Wp = np.vstack([W, np.zeros(W.shape[1])])    # unknown residue scores 0
    idx[idx < 0] = len(vh.ALPHABET)
    L = vh.WINDOW_LENGTH
    scores = [Wp[idx[i:i + L], np.arange(L)].sum() for i in range(len(idx) - L + 1)]
    return int(np.argmax(scores)), float(np.max(scores))


def plot_fp_examples(train, bench):
    pred = pd.read_csv(CMP_DIR / "benchmark_predictions.tsv", sep="\t")
    pred["N-term TM"] = tm_flag(pred)
    seq = dict(zip(bench["UniProt_ID"], bench["Frag_90"]))
    W = final_pswm()
    L = 90

    def kd(s):
        return svm.window_profile(s[:L], svm.KD_SCALE, 7)

    pos = train[train["Class"] == 1]["Frag_90"]
    prof = np.full((len(pos), L), np.nan)
    for i, s in enumerate(pos):
        p = kd(s)
        prof[i, :len(p)] = p
    sp_mean = np.nanmean(prof, axis=0)
    sp_q1, sp_q3 = np.nanpercentile(prof, 25, axis=0), np.nanpercentile(prof, 75, axis=0)

    svm_fp = pred[pred["svm_outcome"] == "FP"].sort_values("svm_score", ascending=False)
    ex_svm = pd.concat([svm_fp[svm_fp["N-term TM"]].head(2), svm_fp[~svm_fp["N-term TM"]].head(1)])
    vh_only = pred[(pred["vh_outcome"] == "FP") & (pred["svm_outcome"] == "TN")]
    vh_only = vh_only.sort_values("vh_score", ascending=False)
    ex_vh = pd.concat([vh_only[vh_only["N-term TM"]].head(1), vh_only[~vh_only["N-term TM"]].head(2)])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    x = np.arange(1, L + 1)
    rows = []
    for r, (examples, label) in enumerate([(ex_svm, "SVM false positive"),
                                           (ex_vh, "von Heijne false positive (SVM: TN)")]):
        for c, (_, e) in enumerate(examples.iterrows()):
            ax = axes[r, c]
            ax.fill_between(x, sp_q1, sp_q3, color=POS_COL, alpha=0.15, lw=0,
                            label="training SPs, IQR")
            ax.plot(x, sp_mean, color=POS_COL, lw=1.6, label="training SPs, mean")
            s = seq[e["UniProt_ID"]]
            p = kd(s)
            ax.plot(np.arange(1, len(p) + 1), p, color=FP_COL, lw=2, label="this protein")
            st, sc = best_window_start(s, W)
            ax.axvspan(st + 0.5, st + vh.WINDOW_LENGTH + 0.5, color=SERIES[3], alpha=0.25, lw=0,
                       label="best PSWM window")
            ax.axvline(40.5, color=MUTED, ls=":", lw=1,
                       label="end of the 40-residue window of the SVM hydrophobicity features")
            ax.axhline(0, color=INK_2, lw=0.6)
            ax.set_xlim(1, L)
            ax.grid(axis="y")
            ax.set_title(f"{e['UniProt_ID']} · {vh_short(e['Organism'])} · {e['Kingdom']}\n"
                         f"{label}; N-term TM: {'yes' if e['N-term TM'] else 'no'}\n"
                         f"PSWM {e['vh_score']:.1f} (θ {final_threshold():.2f}) · "
                         f"SVM {e['svm_score']:+.2f}", fontsize=9.5)
            rows.append({"UniProt_ID": e["UniProt_ID"], "Organism": e["Organism"],
                         "Example of": label, "N-term TM": e["N-term TM"],
                         "vh_score": e["vh_score"], "svm_score": e["svm_score"],
                         "best PSWM window (residues)": f"{st + 1}-{st + vh.WINDOW_LENGTH}",
                         "max KD (window 7, first 40)": kd(s)[:40].max(),
                         "position of max KD": int(np.argmax(kd(s)[:40])) + 1,
                         "N-terminal 40 residues": s[:40]})
    for ax in axes[1]:
        ax.set_xlabel("Residue position")
    for ax in axes[:, 0]:
        ax.set_ylabel("Kyte–Doolittle (window 7)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Example false positives: N-terminal hydrophobicity vs the typical signal peptide",
                 y=1.0)
    fig.tight_layout()
    save(fig, CMP_DIR / "fp_hydrophobicity_examples.png")
    pd.DataFrame(rows).to_csv(CMP_DIR / "fp_hydrophobicity_examples.tsv", sep="\t", index=False)


def vh_short(org):
    org = str(org).split(" (")[0].split()
    return f"{org[0][0]}. {' '.join(org[1:2])}" if len(org) > 1 else org[0]


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        train, bench = vh.load_data(vh.CV_DIR, vh.POS_FASTA, vh.NEG_FASTA)
    bench = bench.reset_index(drop=True)
    plot_data_split(train, bench)
    plot_score_distribution(train, bench)
    plot_inner_heatmaps()
    plot_cv_comparison()
    plot_fp_examples(train, bench)


if __name__ == "__main__":
    main()
