"""
Step 2: Run every (item x wording x repeat x model) combination against a local
Ollama server and save raw results.

Usage (on UCloud, inside the Chat UI app terminal where Ollama is running):
    python run_experiment.py --models llama3.1:8b qwen2.5:7b --repeats 3

Pilot first:
    python run_experiment.py --models llama3.1:8b --limit 2 --output pilot.csv

Output: results.csv (one row per run) and run_metadata.json (model digests,
settings, dates - copy these into your Methods section).
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import time

import pandas as pd
import requests

from prompts import TEMPLATES

OLLAMA = "http://localhost:11434"
LABELS = ("positive", "negative", "neutral")


def parse_label(answer: str) -> str:
    """Map a free-text answer to one of the three labels, or 'invalid'."""
    words = re.findall(r"[a-z]+", answer.lower())
    found = [w for w in words if w in LABELS]
    if not found:
        return "invalid"
    # if the model mentions several labels, take the first one it states
    return found[0]


def model_digest(model: str) -> str:
    """Exact model version (digest) from Ollama's list of local models."""
    r = requests.get(f"{OLLAMA}/api/tags", timeout=60)
    r.raise_for_status()
    for m in r.json().get("models", []):
        if model in (m.get("name"), m.get("model")):
            return m.get("digest", "unknown")
    raise SystemExit(f"Model '{model}' not found. Run: ollama pull {model}")


def ask(model: str, prompt: str, temperature: float, num_ctx: int,
        think: str = "default") -> tuple[str, float]:
    t0 = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 20,
                    "num_ctx": num_ctx},
    }
    if think != "default":          # only for "thinking" models such as Qwen 3
        payload["think"] = (think == "on")
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    if think != "on" and data.get("thinking"):
        raise SystemExit(f"{model} returned reasoning text although --think {think} "
                         "was set; stop and check the settings.")
    return data["response"].strip(), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--sample", default="sample.csv")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=2048,
                   help="context window; small = less GPU memory (prompts are short)")
    p.add_argument("--think", choices=["default", "off", "on"], default="default",
                   help="'off' switches reasoning off for thinking models (e.g. Qwen 3); "
                        "leave 'default' for models without a thinking mode")
    p.add_argument("--limit", type=int, default=None, help="only first N items (pilot)")
    p.add_argument("--output", default="results.csv")
    p.add_argument("--metadata", default=None,
                   help="metadata file (default: <output name>_metadata.json)")
    p.add_argument("--resume", action="store_true",
                   help="keep existing rows in --output and only run what is missing")
    a = p.parse_args()
    if a.metadata is None:
        a.metadata = os.path.splitext(a.output)[0] + "_metadata.json"

    items = pd.read_csv(a.sample)
    if a.limit:
        items = items.head(a.limit)

    meta = {
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "temperature": a.temperature,
        "num_ctx": a.num_ctx,
        "num_predict": 20,
        "think": a.think,
        "repeats": a.repeats,
        "n_items": len(items),
        "templates": TEMPLATES,
        "models": {m: model_digest(m) for m in a.models},
    }
    total = len(a.models) * len(items) * len(TEMPLATES) * a.repeats
    print(f"Running {total} prompts ...")

    done_keys = set()
    if a.resume and os.path.exists(a.output):
        prev = pd.read_csv(a.output)
        done_keys = set(zip(prev["model"], prev["item_id"].astype(int),
                            prev["variant"], prev["run"].astype(int)))
        print(f"Resuming: {len(done_keys)} answers already in {a.output}")
    elif os.path.exists(a.output):
        raise SystemExit(f"{a.output} already exists. Use --resume to continue it, "
                         "or choose another --output name.")

    done = len(done_keys)
    new_file = not done_keys
    with open(a.output, "a" if done_keys else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "item_id", "variant", "run", "gold",
                        "raw_answer", "label", "seconds"])
        for model in a.models:
            for _, row in items.iterrows():
                for variant, tpl in TEMPLATES.items():
                    prompt = tpl.format(text=row["text"])
                    for run in range(1, a.repeats + 1):
                        if (model, int(row["id"]), variant, run) in done_keys:
                            continue
                        answer, secs = ask(model, prompt, a.temperature, a.num_ctx, a.think)
                        w.writerow([model, row["id"], variant, run, row["gold"],
                                    answer, parse_label(answer), round(secs, 2)])
                        f.flush()
                        done += 1
                        if done % 25 == 0 or done == total:
                            print(f"  {done}/{total}", flush=True)

    meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    if done_keys and os.path.exists(a.metadata):
        with open(a.metadata, encoding="utf-8") as f:
            meta["earlier_sessions"] = json.load(f)
    with open(a.metadata, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Results in {a.output}, settings in {a.metadata}")


if __name__ == "__main__":
    main()
