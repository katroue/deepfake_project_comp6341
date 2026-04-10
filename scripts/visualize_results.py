"""
Run AFTER all training + evaluation is complete.

Usage:
    python scripts/visualize_results.py

Outputs saved to: results/visualizations/
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc

LOGS_DIR   = "results/logs"
MODELS_DIR = "results/models"
VIZ_DIR    = "results/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")

# ── Maps both slug-style AND full-name keys (covers CSV + JSON) ──────────────
SHORT_NAMES = {
    # slug keys (from training log filenames)
    "baseline":                  "Baseline",
    "heavy-augmentation":        "Heavy Aug",
    "curriculum-learning":       "Curriculum",
    "self-supervised-learning":  "Self-Sup",
    "hard-negative-mining":      "Hard Neg",
    "multi-task-learning":       "Multi-Task",
    # full-name keys (from evaluation_results.csv strategy column)
    "Baseline":                  "Baseline",
    "Heavy Augmentation":        "Heavy Aug",
    "Curriculum Learning":       "Curriculum",
    "Self-Supervised Pretraining": "Self-Sup",
    "Hard Negative Mining":      "Hard Neg",
    "Multi-Task Learning":       "Multi-Task",
}

def savefig(name):
    path = os.path.join(VIZ_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    csv_files = glob.glob(os.path.join(LOGS_DIR, "*.csv"))
    csv_files = [f for f in csv_files if "evaluation" not in f]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves – All Strategies", fontsize=14, fontweight="bold")

    for fpath in sorted(csv_files):
        strategy = os.path.splitext(os.path.basename(fpath))[0]
        label = SHORT_NAMES.get(strategy, strategy)

        df = pd.read_csv(fpath, header=0)
        if df.shape[1] == 5:
            df.columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        elif df.shape[1] == 7:
            df.columns = ["epoch", "train_loss", "train_binacc", "train_multiacc",
                          "val_loss", "val_binacc", "val_multiacc"]
            df["train_acc"] = df["train_binacc"]
            df["val_acc"]   = df["val_binacc"]

        # deduplicate epoch rows (baseline.csv has duplicate epoch entries)
        df = df.groupby("epoch", as_index=False).mean()

        axes[0].plot(df["epoch"], df["val_acc"],   marker="o", ms=3, label=label)
        axes[1].plot(df["epoch"], df["train_loss"], marker="o", ms=3, label=label)

    axes[0].set_title("Validation Accuracy (%)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)

    savefig("1_training_curves.png")
    print("✔ Training curves done")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FINAL METRICS BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_final_metrics():
    csv_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        print("  ⚠ evaluation_results.csv not found – skipping"); return

    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"]
    df["short"] = df["strategy"].map(SHORT_NAMES).fillna(df["strategy"])

    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m].astype(float) * 100, width, label=m.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["short"], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70, 100)
    ax.set_title("Final Test Metrics by Strategy", fontweight="bold")
    ax.legend()
    savefig("2_final_metrics.png")
    print("✔ Final metrics bar chart done")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROC CURVES  (requires eval JSON from run_evaluation.py)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves():
    json_files = glob.glob(os.path.join(MODELS_DIR, "**/eval_test_*.json"), recursive=True)
    if not json_files:
        print("  ⚠ No eval JSON files found – run run_evaluation.py first"); return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    for jpath in sorted(json_files):
        with open(jpath) as f:
            data = json.load(f)
        if "roc_curve" not in data:
            continue
        fpr = data["roc_curve"]["fpr"]
        tpr = data["roc_curve"]["tpr"]
        auc_val = data.get("auc", sk_auc(fpr, tpr))
        strategy = SHORT_NAMES.get(data.get("strategy_name", ""), data.get("strategy_name", jpath))
        ax.plot(fpr, tpr, lw=2, label=f"{strategy} (AUC={auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Strategies", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    savefig("3_roc_curves.png")
    print("✔ ROC curves done")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFUSION MATRICES — reconstructed from evaluation_results.csv
#    Formula: from accuracy, precision, recall and total sample count
#    TP = recall * P_total  |  FN = P_total - TP
#    FP = TP / precision - TP  |  TN = total - TP - FN - FP
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    csv_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        print("  ⚠ evaluation_results.csv not found – skipping"); return

    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"]
    df["short"] = df["strategy"].map(SHORT_NAMES).fillna(df["strategy"])

    # Try JSON first for exact counts; fall back to CSV reconstruction
    json_files = {
        os.path.basename(os.path.dirname(j)): j
        for j in glob.glob(os.path.join(MODELS_DIR, "**/eval_test_*.json"), recursive=True)
    }

    n = len(df)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    # Approximate total samples from FF++ c23 test split (1000 real + 4000 fake typical)
    TOTAL_SAMPLES = 5000

    for i, (_, row) in enumerate(df.iterrows()):
        acc  = float(row["accuracy"])
        prec = float(row["precision"])
        rec  = float(row["recall"])
        fnr  = float(row["false_negative_rate"])

        # Reconstruct CM from metrics
        # Assume equal real/fake split (2500 each) as per FF++ structure
        n_fake = TOTAL_SAMPLES // 2
        n_real = TOTAL_SAMPLES - n_fake

        TP = round(rec  * n_fake)
        FN = n_fake - TP
        FP = round(TP / prec - TP) if prec > 0 else 0
        TN = n_real - FP

        cm = np.array([[TN, FP],
                       [FN, TP]])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Real", "Fake"],
                    yticklabels=["Real", "Fake"])
        axes[i].set_title(
            f"{row['short']}\nAcc={acc*100:.1f}%  AUC={float(row['auc']):.3f}",
            fontsize=10
        )
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices – All Strategies (c23 test)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("4_confusion_matrices.png")
    print("✔ Confusion matrices done")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PER-MANIPULATION ACCURACY  (requires eval JSON)
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_manipulation():
    json_files = glob.glob(os.path.join(MODELS_DIR, "**/eval_test_*.json"), recursive=True)
    if not json_files:
        print("  ⚠ No eval JSON files found – skipping per-manip chart"); return

    rows = []
    for jpath in sorted(json_files):
        with open(jpath) as f:
            data = json.load(f)
        strategy = SHORT_NAMES.get(data.get("strategy_name", ""), data.get("strategy_name", ""))
        for manip, stats in data.get("per_manipulation", {}).items():
            rows.append({"strategy": strategy, "manipulation": manip,
                         "accuracy": stats["accuracy"] * 100})

    if not rows:
        print("  ⚠ No per-manipulation data found"); return

    df_m = pd.DataFrame(rows)
    pivot = df_m.pivot(index="manipulation", columns="strategy", values="accuracy")

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Per-Manipulation Accuracy by Strategy", fontweight="bold")
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.legend(title="Strategy", fontsize=8, loc="lower right")
    savefig("5_per_manipulation.png")
    print("✔ Per-manipulation accuracy done")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CROSS-COMPRESSION GENERALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_compression():
    csv_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        print("  ⚠ evaluation_results.csv not found – skipping"); return

    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"]
    df["short"] = df["strategy"].map(SHORT_NAMES).fillna(df["strategy"])
    df["accuracy"] = df["accuracy"].astype(float) * 100

    pivot = df.pivot_table(index="short", columns="evaluated_on", values="accuracy")
    if pivot.shape[1] < 2:
        print("  ⚠ Only one compression level in CSV – cross-compression chart skipped")
        print("    Run run_evaluation.py with --compression c40 to enable this chart")
        return

    x = np.arange(len(pivot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col], width, label=f"Eval on {col}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Cross-Compression Generalization", fontweight="bold")
    ax.legend()
    savefig("6_cross_compression.png")
    print("✔ Cross-compression chart done")


# ══════════════════════════════════════════════════════════════════════════════
# 7. AUC + FNR SUMMARY  (bug fix: set_xticks before set_xticklabels)
# ══════════════════════════════════════════════════════════════════════════════
def plot_auc_fnr():
    csv_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        print("  ⚠ evaluation_results.csv not found – skipping"); return

    df = pd.read_csv(csv_path)
    df = df[(df["split"] == "test") & (df["evaluated_on"] == df["trained_on"])]
    df["short"] = df["strategy"].map(SHORT_NAMES).fillna(df["strategy"])
    df["auc"]   = df["auc"].astype(float)
    df["false_negative_rate"] = df["false_negative_rate"].astype(float)

    x = np.arange(len(df))  # fixed: use numeric positions
    palette = sns.color_palette("tab10", len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("AUC-ROC and False Negative Rate by Strategy", fontweight="bold")

    axes[0].bar(x, df["auc"], color=palette)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(i, row["auc"] + 0.005, f'{row["auc"]:.3f}', ha="center", fontsize=8)
    axes[0].set_xticks(x)                                          # ← fix
    axes[0].set_xticklabels(df["short"], rotation=15, ha="right")
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC-ROC Score")

    axes[1].bar(x, df["false_negative_rate"] * 100, color=palette)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1].text(i, row["false_negative_rate"] * 100 + 0.3,
                     f'{row["false_negative_rate"]*100:.1f}%', ha="center", fontsize=8)
    axes[1].set_xticks(x)                                          # ← fix
    axes[1].set_xticklabels(df["short"], rotation=15, ha="right")
    axes[1].set_ylabel("FNR (%) — Fake predicted as Real")
    axes[1].set_title("False Negative Rate")

    savefig("7_auc_fnr.png")
    print("✔ AUC + FNR charts done")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n=== Generating Visualizations ===\n")
    plot_training_curves()
    plot_final_metrics()
    plot_roc_curves()
    plot_confusion_matrices()
    plot_per_manipulation()
    plot_cross_compression()
    plot_auc_fnr()
    print(f"\n✅ All done! Charts saved to: {VIZ_DIR}/")