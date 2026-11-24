"""
Step 1: Draw a reproducible, balanced random sample from the Senti4SD gold standard.

Usage with the Senti4SD gold standard (Excel file from the authors' GitHub repo):
    python sample_data.py --input Senti4SD_GoldStandard_EmotionPolarity.xlsx \
        --sheet "Annotations and Gold Labels" --text-col Text \
        --label-col "Final Label (majority voting)" --per-label 100 --seed 42

For a CSV file, use --sep to give the separator instead of --sheet.
Output: sample.csv with columns id, text, gold
"""
import argparse
import pandas as pd

LABEL_MAP = {  # normalise whatever the dataset uses to three labels
    "positive": "positive", "pos": "positive", "1": "positive",
    "negative": "negative", "neg": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--sep", default=",")
    p.add_argument("--sheet", default=0,
                   help="Excel sheet name (only for .xlsx input)")
    p.add_argument("--per-label", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-chars", type=int, default=1000,
                   help="skip very long posts to keep prompts short")
    p.add_argument("--output", default="sample.csv")
    a = p.parse_args()

    if a.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(a.input, sheet_name=a.sheet)
    else:
        df = pd.read_csv(a.input, sep=a.sep)
    df["gold"] = df[a.label_col].astype(str).str.strip().str.lower().map(LABEL_MAP)
    unknown = df["gold"].isna().sum()
    if unknown:
        print(f"Warning: {unknown} rows had unrecognised labels and were dropped. "
              f"Labels seen: {sorted(df[a.label_col].astype(str).unique())[:10]}")
    df = df.dropna(subset=["gold"])
    df = df[df[a.text_col].astype(str).str.len() <= a.max_chars]

    parts = [g.sample(n=a.per_label, random_state=a.seed)
             for _, g in df.groupby("gold")]
    out = pd.concat(parts).sample(frac=1, random_state=a.seed)  # shuffle order
    out = out.reset_index(drop=True)
    out.insert(0, "id", range(1, len(out) + 1))
    out = out.rename(columns={a.text_col: "text"})[["id", "text", "gold"]]
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out)} items to {a.output} "
          f"(seed={a.seed}, {a.per_label} per label)")
    print(out["gold"].value_counts().to_string())

if __name__ == "__main__":
    main()
