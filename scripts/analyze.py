"""
Step 3: Compute the stability measures from results.csv.

Usage:
    python analyze.py --input results.csv

Measures (per model):
  - Accuracy per wording: share of runs whose label equals the gold label.
  - Repeat consistency: share of (item, wording) pairs where all repeats gave
    the same label -> variation caused by the model's own non-determinism.
  - Wording consistency: share of items where all wordings gave the same
    (majority-over-repeats) label -> variation caused by rewording.
  - Accuracy spread: best minus worst wording accuracy.
  - Invalid rate: answers that were not one of the three labels.
Writes summary tables to summary_accuracy.csv and summary_stability.csv.
"""
import argparse

import pandas as pd


def majority(labels: pd.Series) -> str:
    return labels.value_counts().idxmax()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results.csv")
    a = p.parse_args()

    df = pd.read_csv(a.input)
    df["correct"] = df["label"] == df["gold"]

    # Accuracy per model x wording
    acc = (df.groupby(["model", "variant"])["correct"].mean()
             .unstack("variant").round(3))
    print("\nAccuracy per wording (share correct):")
    print(acc.to_string())

    rows = []
    for model, d in df.groupby("model"):
        # repeat consistency: all repeats identical for a given item+wording
        rep = d.groupby(["item_id", "variant"])["label"].nunique().eq(1).mean()
        # wording consistency: majority labels identical across wordings
        maj = d.groupby(["item_id", "variant"])["label"].apply(majority)
        word = maj.groupby("item_id").nunique().eq(1).mean()
        accs = d.groupby("variant")["correct"].mean()
        rows.append({
            "model": model,
            "repeat_consistency": round(rep, 3),
            "wording_consistency": round(word, 3),
            "accuracy_min": round(accs.min(), 3),
            "accuracy_max": round(accs.max(), 3),
            "accuracy_spread": round(accs.max() - accs.min(), 3),
            "invalid_rate": round((d["label"] == "invalid").mean(), 3),
            "mean_seconds": round(d["seconds"].mean(), 2),
        })
    stab = pd.DataFrame(rows)
    print("\nStability summary:")
    print(stab.to_string(index=False))

    # Items whose answer changed with wording (good examples for the paper)
    flips = []
    for (model, item), d in df.groupby(["model", "item_id"]):
        maj = d.groupby("variant")["label"].apply(majority)
        if maj.nunique() > 1:
            flips.append({"model": model, "item_id": item,
                          "gold": d["gold"].iloc[0], **maj.to_dict()})
    if flips:
        print(f"\nItems whose label changed across wordings: {len(flips)}")
        print(pd.DataFrame(flips).to_string(index=False))

    acc.to_csv("summary_accuracy.csv")
    stab.to_csv("summary_stability.csv", index=False)
    print("\nSaved summary_accuracy.csv and summary_stability.csv")


if __name__ == "__main__":
    main()
