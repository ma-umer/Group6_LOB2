#!/usr/bin/env python3
"""
Benchmark comparison and error analysis of the von Heijne (PSWM) and SVM signal
peptide classifiers (course slides: "Training final models and analysis of
benchmarking results").

Both final models are applied to the same benchmark set (02_data_preparation/
cross_validation/pos_bench.tsv + neg_bench.tsv):
  - von Heijne: the final PSWM, built on all 5 training folds
    (04_von_heijne/model_results/pswm_final.tsv), with the threshold fixed on the
    validation fold (04_von_heijne/model_results/final_model.tsv)
  - SVM: the final models saved by 05_svm/svm_model.py, tuned by 5-fold CV on the
    whole training set and refitted on all of it; each saved bundle holds its own
    n and feature list. "SVM" is the full pipeline (RF top-k features -> SVM);
    "SVM (all features)" is the k = 39 baseline, shown only when it differs.

Nothing is retrained. The script reuses the data loading / feature extraction code
of the two model scripts (and the WebLogo code of 03_data_analysis), so the inputs
are exactly the same.

Error analysis, as asked by the slides:
  - false positives: share of FPs with a TM helix in the first 90 residues
  - false negatives, von Heijne: sequence logo of the cleavage-site context of the
    FNs, compared with the expected logo (training positives) and with the TPs
  - false negatives, SVM: training positives vs benchmark TP vs benchmark FN for
    the N-terminal composition (first n residues), the SP length and every
    feature the model uses
  - errors by kingdom and by species

Outputs (comparison_results/):
  benchmark_metrics.tsv          metrics + bootstrap 95% CI of MCC for each model
  mcnemar_test.tsv               paired test of the two classifiers' errors
  benchmark_predictions.tsv      one row per benchmark protein, both models' scores/calls
  error_overlap.tsv              FP / FN shared or unique to each model
  fpr_by_tm.tsv                  FPR with / without an N-terminal TM helix, share of FPs that are TM
  rates_by_kingdom.tsv, rates_by_species.tsv   FN / FP counts and rates per taxon
  sp_length_by_outcome.tsv       SP length of training positives, TP and FN
  vh_cleavage_site_by_outcome.tsv  cleavage-site residues and PSWM scores, training / TP / FN
  svm_fn_features.tsv            every SVM feature: training positives vs TP vs FN
  metrics_comparison.png         bar chart of the main metrics with MCC CIs
  error_overlap.png              shared vs unique FP and FN
  fpr_transmembrane.png          FPR on TM vs non-TM negatives
  rates_by_kingdom.png, rates_by_species.png   FPR and FNR per taxon
  score_scatter.png              von Heijne score vs SVM decision value
  logos_vh_false_negatives.png   cleavage-site logos: expected, vH TP, vH FN
  sp_length_tp_fn.png            SP length: training positives, TP, FN, for each model
  composition_tp_fn.png          composition of the first n residues: training positives, TP, FN
  svm_fn_features.png            standardised difference of every SVM feature (TP, FN, negatives)
  svm_fn_feature_distributions.png  the features that differ most, as box plots
  features_by_outcome.png        key SVM features across TP / FN / FP / TN

Run:  python 06_comparison/compare_models.py
"""

import argparse
import importlib.util
import math
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest, mannwhitneyu

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
VH_DIR = PROJECT_DIR / "04_von_heijne"
SVM_DIR = PROJECT_DIR / "05_svm"
DA_DIR = PROJECT_DIR / "03_data_analysis"
OUTPUT_DIR = BASE_DIR / "comparison_results"

N_BOOT = 2000
SEED = 42
MIN_SPECIES = 20        # species with fewer benchmark proteins are pooled as "Other species"

OUTCOME_ORDER = ["TP", "FN", "FP", "TN"]
AA_ORDER = "LAVIFMWCGSTPYNQHDEKR"   # hydrophobic -> polar -> charged
GROUPS = ["Training positives", "Benchmark TP", "Benchmark FN"]


def load_module(name, path):
    """Import one of the project scripts as a module (its main() is not run)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vh = load_module("vh_model", VH_DIR / "von_heijne_method.py")
svm = load_module("svm_model", SVM_DIR / "svm_model.py")
da = load_module("data_analysis", DA_DIR / "data_analysis_combined.py")

SERIES, INK, MUTED = vh.SERIES, vh.INK, vh.MUTED
COLORS = {"von Heijne": SERIES[0], "SVM": SERIES[1], "SVM (all features)": SERIES[2]}
OUTCOME_COLORS = {"TP": SERIES[0], "FN": SERIES[1], "FP": SERIES[7], "TN": MUTED}
GROUP_COLORS = {"Training positives": MUTED, "Benchmark TP": SERIES[0], "Benchmark FN": SERIES[1]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(TP, TP + FP), div(TP, TP + FN)
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "MCC": div(TP * TN - FP * FN, den), "Precision": prec, "Recall": rec,
            "Specificity": div(TN, TN + FP), "Accuracy": div(TP + TN, len(y_true)),
            "F1-score": div(2 * prec * rec, prec + rec), "FPR": div(FP, FP + TN)}


def bootstrap_mcc(y_true, preds, n=N_BOOT, seed=SEED):
    """95% CI of MCC per model and of the MCC difference SVM - von Heijne.
    Resamples positives and negatives separately (keeps the class ratio)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    out = {k: [] for k in preds}
    diff = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals = {k: metrics(y_true[idx], p[idx])["MCC"] for k, p in preds.items()}
        for k, v in vals.items():
            out[k].append(v)
        diff.append(vals["SVM"] - vals["von Heijne"])
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}
    return ci, np.percentile(diff, [2.5, 97.5]), np.mean(np.array(diff) <= 0)


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on the proteins where exactly one model is wrong."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b_only = int((~a_ok & b_ok).sum())   # model A wrong, model B right
    a_only = int((a_ok & ~b_ok).sum())
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


# ---------------------------------------------------------------------------
# Helpers for the biological analyses
# ---------------------------------------------------------------------------

def outcome(y, p):
    return np.select([(y == 1) & (p == 1), (y == 1) & (p == 0), (y == 0) & (p == 1)],
                     ["TP", "FN", "FP"], "TN")


def short_species(name):
    return str(name).split(" (")[0]


def abbreviate(species):
    """'Homo sapiens' -> 'H. sapiens' (figure labels only)."""
    parts = str(species).split()
    return f"{parts[0][0]}. {parts[1]}" if len(parts) > 1 and species != "Other species" else species


def window_scores(seq, W):
    """PSWM score of every 15-residue window in the first SCAN_LIMIT residues
    (index = 0-based start of the window)."""
    W_padded = np.vstack([W, np.zeros((1, W.shape[1]))])
    idx = vh._LUT[np.frombuffer(seq[:vh.SCAN_LIMIT].encode("ascii", "ignore"), dtype=np.uint8)]
    if len(idx) < W.shape[1]:
        return np.array([0.0])
    return W_padded[sliding_window_view(idx, W.shape[1]), np.arange(W.shape[1])].sum(axis=1)


SMALL_M1 = set("AGSCT")     # residues accepted at -1
SMALL_M3 = set("AVSGCT")    # residues accepted at -3


def cleavage_stats(motifs, seqs=None, cleavage=None, W=None):
    """Cleavage-site context [-13, +2] of a group of signal peptides: residues at
    -1 and -3 and, when the sequences are given, the PSWM score of the true
    cleavage window and whether it is the best-scoring window."""
    m = [x for x in motifs if len(x) == vh.WINDOW_LENGTH]
    p1, p3 = [x[12] for x in m], [x[10] for x in m]          # -1 and -3
    row = {"n": len(m),
           "Ala at -1 (%)": 100 * np.mean([a == "A" for a in p1]),
           "small residue at -1 (A,G,S,C,T) (%)": 100 * np.mean([a in SMALL_M1 for a in p1]),
           "small residue at -3 (A,V,S,G,C,T) (%)": 100 * np.mean([a in SMALL_M3 for a in p3]),
           "A-x-A (Ala at -3 and -1) (%)": 100 * np.mean([a == "A" == b for a, b in zip(p3, p1)]),
           "Leu in -13..-6 (%)": 100 * np.mean([x[:8].count("L") / 8 for x in m])}
    if seqs is not None:
        true_s, best_here = [], []
        for s, c in zip(seqs, cleavage):
            sc = window_scores(s, W)
            start = int(c) - vh.SP_LEFT
            if 0 <= start < len(sc):
                true_s.append(sc[start])
                best_here.append(int(np.argmax(sc)) == start)
        row["median PSWM score of the true cleavage window"] = float(np.median(true_s))
        row["best-scoring window = true cleavage window (%)"] = 100 * float(np.mean(best_here))
    return row


def logo_panel(groups, path):
    """WebLogo 3 logos of several motif groups (same 0-2.5 bit axis, drawn by
    03_data_analysis), stacked in one figure."""
    from matplotlib.image import imread
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (title, motifs) in enumerate(groups):
            p = Path(tmp) / f"logo{i}.png"
            da.plot_logo([(f"s{j}", m) for j, m in enumerate(motifs)], title, p)
            if not p.exists():
                print(f"  logo not drawn ({path.name}); install weblogo + Ghostscript")
                return
            images.append(imread(p))
    fig, axes = plt.subplots(len(images), 1, figsize=(10, 3.45 * len(images)))
    for ax, im in zip(axes, images):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout(h_pad=0.4)
    save(fig, path)


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def rates_by(bench, y, preds, column, models=("von Heijne", "SVM")):
    rows = []
    for taxon, g in bench.groupby(column, sort=False):
        idx = g.index.to_numpy()
        for name in models:
            p, yy = preds[name][idx], y[idx]
            npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
            fn, fp = int(((yy == 1) & (p == 0)).sum()), int(((yy == 0) & (p == 1)).sum())
            rows.append({column: taxon, "Model": name, "Positives": npos, "Negatives": nneg,
                         "FN": fn, "FP": fp,
                         "FNR (%)": fn / npos * 100 if npos else np.nan,
                         "FPR (%)": fp / nneg * 100 if nneg else np.nan})
    return pd.DataFrame(rows)


def plot_rates(table, column, order, title, path, rotate=20, italic=False):
    """FPR and FNR per taxon for both models; each bar is labelled with its
    counts (errors / proteins of that class)."""
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.25 * len(order) + 3), 4.4))
    for ax, (rate, err, tot) in zip(axes, (("FPR (%)", "FP", "Negatives"),
                                           ("FNR (%)", "FN", "Positives"))):
        sns.barplot(data=table, x=column, y=rate, hue="Model", order=order, palette=COLORS,
                    ax=ax, saturation=1)
        for cont, (name, sub) in zip(ax.containers, table.groupby("Model", sort=False)):
            sub = sub.set_index(column).reindex(order)
            labels = [f"{int(e)}/{int(t)}" if t else "–" for e, t in zip(sub[err], sub[tot])]
            ax.bar_label(cont, labels=labels, fontsize=6.5, padding=1.5)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=rotate)
        if italic:
            for lab in ax.get_xticklabels():
                if lab.get_text() != "Other species":
                    lab.set_fontstyle("italic")
        ax.set_title(f"{'False-positive' if err == 'FP' else 'False-negative'} rate {title}")
    axes[1].get_legend().remove()
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.99, -0.01, "bar labels: errors / proteins of that class", ha="right",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compare von Heijne and SVM on the benchmark set.")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return ap.parse_args()


def main():
    out = parse_args().output
    out.mkdir(parents=True, exist_ok=True)

    # ---- data (same loader as the von Heijne script: adds SP_15 / Frag_90) ----
    train, bench = vh.load_data(vh.CV_DIR, vh.POS_FASTA, vh.NEG_FASTA)
    bench = bench.reset_index(drop=True)
    y = bench["Class"].to_numpy()
    bench["N-term TM"] = bench["N-term TM"].astype(str).str.lower().isin(["true", "1", "1.0"])
    bench["SP cleavage"] = pd.to_numeric(bench["SP cleavage"], errors="coerce")
    bench["Species"] = bench["Organism"].map(short_species)
    print(f"Benchmark: {(y == 1).sum()} positives, {(y == 0).sum()} negatives")

    # ---- von Heijne: final PSWM + threshold fixed on the validation fold ----
    W = pd.read_csv(VH_DIR / "model_results" / "pswm_final.tsv", sep="\t", index_col=0)
    W = W.loc[list(vh.ALPHABET)].to_numpy()
    final_vh = pd.read_csv(VH_DIR / "model_results" / "final_model.tsv", sep="\t",
                           index_col="Parameter")["Value"]
    vh_thr = float(final_vh["threshold"])
    vh_score = vh.score_sequences(bench["Frag_90"], W)
    vh_pred = (vh_score >= vh_thr).astype(int)
    print(f"von Heijne: final PSWM (folds {final_vh['training folds']}), threshold {vh_thr:.3f}")

    # ---- SVM: saved final pipelines ----
    # each saved bundle carries its own n (composition length) and feature list
    fixed_b = svm.fixed_features(bench["Frag_90"])
    b_all = svm.load_model(SVM_DIR / "model_results" / "svm_all_final.pkl.gz")
    b_sel = svm.load_model(SVM_DIR / "model_results" / "svm_selected_final.pkl.gz")
    X_all = svm.bundle_features(b_all, bench["Frag_90"], fixed_b)
    X_sel = svm.bundle_features(b_sel, bench["Frag_90"], fixed_b)
    svm_score = b_sel["model"].decision_function(X_sel)
    svm_pred = b_sel["model"].predict(X_sel).astype(int)
    all_pred = b_all["model"].predict(X_all).astype(int)
    X = svm.extract_features(bench["Frag_90"], b_sel["n"], fixed_b)   # all 39, for the analyses
    same_svm = (b_sel["k"] == b_all["k"] and b_sel["C"] == b_all["C"]
                and b_sel["gamma"] == b_all["gamma"])
    print(f"SVM: final model k={b_sel['k']}, C={b_sel['C']:g}, gamma={b_sel['gamma']:.4g}, "
          f"n={b_sel['n']}" + ("  (same as the all-features model)" if same_svm else
                               f"; all features: C={b_all['C']:g}"))

    preds = {"von Heijne": vh_pred, "SVM": svm_pred}
    if not same_svm:
        preds["SVM (all features)"] = all_pred

    # ---- 1. metrics + bootstrap CI + McNemar ----
    ci, diff_ci, p_boot = bootstrap_mcc(y, preds)
    rows = []
    for name, p in preds.items():
        m = metrics(y, p)
        rows.append({"Model": name, **m, "MCC 95% CI low": ci[name][0],
                     "MCC 95% CI high": ci[name][1]})
    met = pd.DataFrame(rows)
    met.to_csv(out / "benchmark_metrics.tsv", sep="\t", index=False)
    print("\n" + met.round(3).to_string(index=False))

    vh_only, svm_only, p_mc = mcnemar(y, vh_pred, svm_pred)
    tests = [{"Comparison": "von Heijne vs SVM",
              "Correct only by first": vh_only, "Correct only by second": svm_only,
              "McNemar exact p": p_mc,
              "MCC difference (SVM - vH) 95% CI low": diff_ci[0],
              "MCC difference (SVM - vH) 95% CI high": diff_ci[1],
              "Bootstrap fraction diff <= 0": p_boot}]
    print(f"\nMcNemar: correct only by vH = {vh_only}, only by SVM = {svm_only}, p = {p_mc:.2e}")
    print(f"MCC difference SVM - vH: 95% CI [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]")
    if not same_svm:
        all_only, sel_only, p_sel = mcnemar(y, all_pred, svm_pred)
        tests.append({"Comparison": "SVM (all features) vs SVM",
                      "Correct only by first": all_only, "Correct only by second": sel_only,
                      "McNemar exact p": p_sel})
        print(f"McNemar SVM (all features) vs SVM: correct only by all-features = {all_only}, "
              f"only by SVM = {sel_only}, p = {p_sel:.3g}")
    pd.DataFrame(tests).to_csv(out / "mcnemar_test.tsv", sep="\t", index=False)

    # ---- per-protein table ----
    pred_df = bench[["UniProt_ID", "Class", "Organism", "Kingdom", "Length",
                     "SP cleavage", "N-term TM"]].copy()
    pred_df["vh_score"], pred_df["vh_pred"] = vh_score, vh_pred
    pred_df["svm_score"], pred_df["svm_pred"] = svm_score, svm_pred
    pred_df["vh_outcome"], pred_df["svm_outcome"] = outcome(y, vh_pred), outcome(y, svm_pred)
    feats = pd.DataFrame(X, columns=svm.FEATURE_NAMES)
    pred_df = pd.concat([pred_df, feats[["max_hydrophobicity", "mean_hydrophobicity",
                                         "tm_max", "tm_position", "max_charge",
                                         "composition_(L)"]]], axis=1)
    pred_df.to_csv(out / "benchmark_predictions.tsv", sep="\t", index=False)

    # ---- 2. error overlap ----
    ov = []
    for err, cls, wrong in (("FP", 0, 1), ("FN", 1, 0)):
        a = (y == cls) & (vh_pred == wrong)
        b = (y == cls) & (svm_pred == wrong)
        ov.append({"Error": err, "Both": int((a & b).sum()),
                   "von Heijne only": int((a & ~b).sum()),
                   "SVM only": int((~a & b).sum())})
    ov = pd.DataFrame(ov)
    ov.to_csv(out / "error_overlap.tsv", sep="\t", index=False)
    print("\nError overlap:\n" + ov.to_string(index=False))

    # ---- 3. false positives: TM helix in the first 90 residues ----
    neg = y == 0
    is_tm = bench["N-term TM"].to_numpy()
    tm_rows = []
    for name in ("von Heijne", "SVM"):
        p = preds[name]
        for grp, mask in (("N-terminal TM", is_tm), ("No TM", ~is_tm)):
            n = int((neg & mask).sum())
            fp = int((neg & mask & (p == 1)).sum())
            tm_rows.append({"Model": name, "Group": grp, "Negatives": n, "FP": fp,
                            "FPR (%)": fp / n * 100 if n else np.nan})
        fp_all = int((neg & (p == 1)).sum())
        fp_tm = int((neg & (p == 1) & is_tm).sum())
        tm_rows.append({"Model": name, "Group": "Share of FPs that are TM", "Negatives": fp_all,
                        "FP": fp_tm, "FPR (%)": fp_tm / fp_all * 100 if fp_all else np.nan})
    tm = pd.DataFrame(tm_rows)
    tm.to_csv(out / "fpr_by_tm.tsv", sep="\t", index=False)
    print("\n" + tm.round(2).to_string(index=False))
    print(f"(N-terminal TM proteins are {is_tm[neg].mean() * 100:.1f} % of the benchmark negatives)")

    # ---- 4. errors by kingdom and by species ----
    kd = rates_by(bench, y, preds, "Kingdom")
    kd.to_csv(out / "rates_by_kingdom.tsv", sep="\t", index=False)
    print("\n" + kd.round(2).to_string(index=False))

    counts = bench["Species"].value_counts()
    keep = counts[counts >= MIN_SPECIES].index
    bench["Species group"] = np.where(bench["Species"].isin(keep), bench["Species"], "Other species")
    sp_order = list(keep) + ["Other species"]
    spd = rates_by(bench, y, preds, "Species group")
    spd = spd.rename(columns={"Species group": "Species"})
    spd["Species"] = pd.Categorical(spd["Species"], sp_order, ordered=True)
    spd["Model"] = pd.Categorical(spd["Model"], ["von Heijne", "SVM"], ordered=True)
    spd = spd.sort_values(["Species", "Model"]).reset_index(drop=True)
    spd["Model"] = spd["Model"].astype(str)
    spd.to_csv(out / "rates_by_species.tsv", sep="\t", index=False)
    print("\n" + spd.round(2).to_string(index=False))

    # ---- 5. false negatives ----
    pos = bench[y == 1].copy()
    tr_pos = train[train["Class"] == 1].reset_index(drop=True)
    tr_pos["SP cleavage"] = pd.to_numeric(tr_pos["SP cleavage"], errors="coerce")
    o_vh, o_svm = outcome(y, vh_pred)[y == 1], outcome(y, svm_pred)[y == 1]
    out_by_model = {"von Heijne": o_vh, "SVM": o_svm}

    # SP length: training positives vs TP vs FN
    len_rows = [{"Model": "-", "Group": "Training positives", "n": len(tr_pos),
                 "Mean": tr_pos["SP cleavage"].mean(), "Median": tr_pos["SP cleavage"].median(),
                 "SD": tr_pos["SP cleavage"].std()}]
    for name, o in out_by_model.items():
        for oc in ("TP", "FN"):
            L = pos.loc[o == oc, "SP cleavage"]
            len_rows.append({"Model": name, "Group": f"Benchmark {oc}", "n": len(L),
                             "Mean": L.mean(), "Median": L.median(), "SD": L.std(),
                             "Mann-Whitney p vs training positives":
                                 mannwhitneyu(L, tr_pos["SP cleavage"]).pvalue})
    ln = pd.DataFrame(len_rows)
    ln.to_csv(out / "sp_length_by_outcome.tsv", sep="\t", index=False)
    print("\n" + ln.round(3).to_string(index=False))

    # von Heijne: cleavage-site context of the FNs vs the expected one
    cs_rows = [{"Group": "Training positives (expected)",
                **cleavage_stats(tr_pos["SP_15"].tolist())}]
    for oc in ("TP", "FN"):
        g = pos[o_vh == oc]
        cs_rows.append({"Group": f"von Heijne benchmark {oc}",
                        **cleavage_stats(g["SP_15"].tolist(), g["Frag_90"], g["SP cleavage"], W)})
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(out / "vh_cleavage_site_by_outcome.tsv", sep="\t", index=False)
    print("\n" + cs.round(2).T.to_string())

    # SVM: every feature, training positives vs TP vs FN (n of the final model)
    fixed_tr = svm.fixed_features(tr_pos["Frag_90"])
    X_trpos = svm.extract_features(tr_pos["Frag_90"], b_sel["n"], fixed_tr)
    tr_neg = train[train["Class"] == 0]
    X_trneg = svm.extract_features(tr_neg["Frag_90"], b_sel["n"], svm.fixed_features(tr_neg["Frag_90"]))
    X_pos = X[y == 1]
    grp_X = {"Training positives": X_trpos, "Benchmark TP": X_pos[o_svm == "TP"],
             "Benchmark FN": X_pos[o_svm == "FN"]}
    names = list(svm.FEATURE_NAMES)
    used = set(b_sel["features"])
    f_rows = []
    for j, f in enumerate(names):
        ref = X_trpos[:, j]
        sd = ref.std(ddof=1) or 1.0
        f_rows.append({"feature": f, "used by the final SVM": f in used,
                       **{f"median {g}": float(np.median(v[:, j])) for g, v in grp_X.items()},
                       "median training negatives": float(np.median(X_trneg[:, j])),
                       "SMD Benchmark TP": (grp_X["Benchmark TP"][:, j].mean() - ref.mean()) / sd,
                       "SMD Benchmark FN": (grp_X["Benchmark FN"][:, j].mean() - ref.mean()) / sd,
                       "SMD training negatives": (X_trneg[:, j].mean() - ref.mean()) / sd,
                       "Mann-Whitney p, FN vs training positives":
                           mannwhitneyu(grp_X["Benchmark FN"][:, j], ref).pvalue,
                       "Mann-Whitney p, TP vs training positives":
                           mannwhitneyu(grp_X["Benchmark TP"][:, j], ref).pvalue})
    fdf = (pd.DataFrame(f_rows).sort_values("SMD Benchmark FN", key=np.abs, ascending=False)
           .reset_index(drop=True))
    fdf.to_csv(out / "svm_fn_features.tsv", sep="\t", index=False)
    print("\nSVM features (standardised mean difference vs training positives, by |SMD FN|):")
    print(fdf[["feature", "used by the final SVM", "SMD Benchmark TP", "SMD Benchmark FN",
               "SMD training negatives", "Mann-Whitney p, FN vs training positives"]]
          .head(15).round(3).to_string(index=False))

    # =======================================================================
    # Figures
    # =======================================================================
    vh.setup_style()
    mpl.rcParams.update({"font.size": 10, "axes.grid": True, "axes.grid.axis": "y",
                         "axes.axisbelow": True})

    # metrics bar chart
    show = ["MCC", "Precision", "Recall", "F1-score"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    nm = len(met)
    w = 0.8 / nm
    for i, (_, r) in enumerate(met.iterrows()):
        xs = np.arange(len(show)) + (i - (nm - 1) / 2) * w
        vals = [r[s] for s in show]
        ax.bar(xs, vals, w, label=r["Model"], color=COLORS[r["Model"]])
        ax.errorbar(xs[0], r["MCC"], yerr=[[r["MCC"] - r["MCC 95% CI low"]],
                                          [r["MCC 95% CI high"] - r["MCC"]]],
                    color=INK, capsize=3, lw=1)
        tops = [r["MCC 95% CI high"]] + vals[1:]   # MCC label goes above its error bar
        for x, v, top in zip(xs, vals, tops):
            ax.text(x, top + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(show)), show)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark performance (error bar: bootstrap 95% CI of MCC)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=nm, frameon=False)
    save(fig, out / "metrics_comparison.png")

    # error overlap
    fig, ax = plt.subplots(figsize=(6, 3.5))
    left = np.zeros(2)
    for col, color in (("von Heijne only", COLORS["von Heijne"]), ("Both", MUTED),
                       ("SVM only", COLORS["SVM"])):
        ax.barh(ov["Error"], ov[col], left=left, color=color, label=col)
        for yi, (l, v) in enumerate(zip(left, ov[col])):
            if v:
                ax.text(l + v / 2, yi, str(v), ha="center", va="center", color="white", fontsize=9)
        left += ov[col].to_numpy()
    ax.set_xlabel("Benchmark proteins")
    ax.set_title("Which model makes which errors")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    save(fig, out / "error_overlap.png")

    # FPR by TM
    t = tm[tm["Group"] != "Share of FPs that are TM"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=t, x="Group", y="FPR (%)", hue="Model", palette=COLORS, ax=ax, saturation=1)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    ax.set_xlabel("")
    ax.set_title("False-positive rate on benchmark negatives")
    save(fig, out / "fpr_transmembrane.png")

    # rates by kingdom and by species
    k_order = [k for k in ("Metazoa", "Fungi", "Viridiplantae", "Other") if k in set(kd["Kingdom"])]
    plot_rates(kd, "Kingdom", k_order, "by kingdom", out / "rates_by_kingdom.png")
    plot_rates(spd.assign(Species=spd["Species"].astype(str).map(abbreviate)), "Species",
               [abbreviate(s) for s in sp_order],
               f"by species (≥ {MIN_SPECIES} benchmark proteins)", out / "rates_by_species.png",
               rotate=35, italic=True)

    # score scatter
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls, lab, mk, col, a in ((0, "Negative", "o", MUTED, 0.3),
                                 (1, "Positive", "^", SERIES[2], 0.85)):
        m = y == cls
        ax.scatter(vh_score[m], svm_score[m], s=12, marker=mk, c=col, alpha=a, label=lab,
                   edgecolors="none")
    tm_neg = (y == 0) & is_tm
    ax.scatter(vh_score[tm_neg], svm_score[tm_neg], s=22, facecolors="none",
               edgecolors=INK, lw=0.5, label="Negative with N-terminal TM")
    ax.axvline(vh_thr, color=COLORS["von Heijne"], ls="--", lw=1.2,
               label=f"vH threshold ({vh_thr:.2f})")
    ax.axhline(0, color=COLORS["SVM"], ls="--", lw=1.2, label="SVM decision boundary (0)")
    ax.set_xlabel("von Heijne score (best window, log2 odds)")
    ax.set_ylabel("SVM decision value")
    ax.set_title("Benchmark proteins scored by both models")
    ax.grid(True, axis="both")
    ax.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    save(fig, out / "score_scatter.png")

    # von Heijne FN logos: expected (training positives) vs TP vs FN
    logo_panel([(f"Expected: training positives (n = {len(tr_pos)})", tr_pos["SP_15"].tolist()),
                (f"von Heijne benchmark TP (n = {int((o_vh == 'TP').sum())})",
                 pos.loc[o_vh == "TP", "SP_15"].tolist()),
                (f"von Heijne benchmark FN (n = {int((o_vh == 'FN').sum())})",
                 pos.loc[o_vh == "FN", "SP_15"].tolist())],
               out / "logos_vh_false_negatives.png")

    # SP length: training positives vs TP vs FN, both models
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, o) in zip(axes, out_by_model.items()):
        d = pd.concat([pd.DataFrame({"Group": "Training positives", "SP length": tr_pos["SP cleavage"]}),
                       pd.DataFrame({"Group": np.where(o == "TP", "Benchmark TP", "Benchmark FN"),
                                     "SP length": pos["SP cleavage"].to_numpy()})])
        sns.boxplot(data=d, x="Group", y="SP length", order=GROUPS, hue="Group",
                    palette=GROUP_COLORS, legend=False, ax=ax, width=0.5, saturation=1, fliersize=0)
        sns.stripplot(data=d, x="Group", y="SP length", order=GROUPS, color=INK,
                      size=2, alpha=0.35, ax=ax)
        n = d["Group"].value_counts()
        ax.set_xticks([0, 1, 2], [f"{g.replace('Benchmark ', '')}\n(n={n.get(g, 0)})" for g in GROUPS])
        ax.set_title(name)
        ax.set_xlabel("")
    axes[0].set_ylabel("Signal peptide length (aa)")
    fig.suptitle("Signal peptide length: training positives vs detected vs missed benchmark positives")
    fig.tight_layout()
    save(fig, out / "sp_length_tp_fn.png")

    # composition of the first n residues: training positives vs TP vs FN, both models
    comp_cols = [names.index(f"composition_({aa})") for aa in AA_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    for ax, (name, o) in zip(axes, out_by_model.items()):
        comp = pd.DataFrame({"Training positives": X_trpos[:, comp_cols].mean(axis=0) * 100,
                             "Benchmark TP": X_pos[o == "TP"][:, comp_cols].mean(axis=0) * 100,
                             "Benchmark FN": X_pos[o == "FN"][:, comp_cols].mean(axis=0) * 100},
                            index=list(AA_ORDER))
        comp.plot.bar(ax=ax, color=[GROUP_COLORS[g] for g in GROUPS], width=0.8, edgecolor="none")
        ax.set_ylabel("% of residues")
        ax.grid(False, axis="x")
        ax.set_title(f"{name}: composition of the first {b_sel['n']} residues")
        ax.legend(title="", fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, out / "composition_tp_fn.png")

    # SVM features: standardised mean difference vs training positives
    fd = fdf.sort_values("SMD Benchmark FN")
    labels = [f + ("" if u else "  (not used)") for f, u in zip(fd["feature"], fd["used by the final SVM"])]
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(fd) + 1.6))
    yy = np.arange(len(fd))
    ax.axvline(0, color=INK, lw=0.8)
    ax.scatter(fd["SMD training negatives"], yy, s=26, facecolors="none", edgecolors=MUTED,
               lw=1.1, label="Training negatives", zorder=3)
    ax.scatter(fd["SMD Benchmark TP"], yy, s=26, color=GROUP_COLORS["Benchmark TP"],
               label=f"Benchmark TP (n = {len(grp_X['Benchmark TP'])})", zorder=4)
    ax.scatter(fd["SMD Benchmark FN"], yy, s=34, color=GROUP_COLORS["Benchmark FN"],
               label=f"Benchmark FN (n = {len(grp_X['Benchmark FN'])})", zorder=5)
    ax.set_yticks(yy, labels, fontsize=8)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Difference from the training positives (mean, in SD of the training positives)")
    ax.set_title("SVM features: benchmark TP and FN vs training positives", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor="none")
    save(fig, out / "svm_fn_features.png")

    # the features that differ most, as box plots
    top = fdf[fdf["used by the final SVM"]].head(6)["feature"].tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(2.6 * len(top), 3.8))
    for ax, f in zip(np.atleast_1d(axes), top):
        j = names.index(f)
        d = pd.concat([pd.DataFrame({"Group": g, "v": v[:, j]}) for g, v in grp_X.items()])
        sns.boxplot(data=d, x="Group", y="v", order=GROUPS, hue="Group", palette=GROUP_COLORS,
                    legend=False, ax=ax, width=0.6, saturation=1, fliersize=1.2)
        ax.set_xticks([0, 1, 2], ["Train\npos.", "TP", "FN"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f, fontsize=9)
    fig.suptitle("SVM false negatives: the features that differ most from the training positives")
    fig.tight_layout()
    save(fig, out / "svm_fn_feature_distributions.png")

    # key features by outcome (SVM outcomes)
    keys = [("max_hydrophobicity", "Max hydrophobicity (KD, w=7)"),
            ("tm_max", "Max TM propensity"),
            ("tm_position", "Position of max TM propensity"),
            ("max_charge", "Max positive charge (res. 1-20)")]
    d = pred_df.assign(Outcome=pred_df["svm_outcome"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (col, lab) in zip(axes, keys):
        sns.boxplot(data=d, x="Outcome", y=col, order=OUTCOME_ORDER, hue="Outcome",
                    palette=OUTCOME_COLORS, legend=False, ax=ax, fliersize=1.5, width=0.6, saturation=1)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("SVM benchmark outcomes: key features")
    fig.tight_layout()
    save(fig, out / "features_by_outcome.png")

    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
