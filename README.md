# Prompt Wording Sensitivity of LLMs in SE Sentiment Analysis

Code, data sample and raw results for the paper:

> D. Gansauskas, "An Experiment on the Prompt Wording Sensitivity of Large Language Models in Software Engineering Sentiment Analysis," Scientific Methods (E26), University of Southern Denmark, 2026.

The experiment asks how much of the variation in an LLM's sentiment labels for Stack Overflow posts is caused by **rewording the prompt** compared with **repeating the same prompt**. Each model classified the same 300 posts with four prompt wordings, three times each, at temperature 0.

## Repository structure

```
scripts/      Python scripts (sampling, running, analysis)
notebooks/    Kaggle notebooks for the follow-up runs
data/         The 300-post sample and the Senti4SD licence
results/      Raw answers, model metadata and summary tables, one folder per run
```

| Folder | Contents |
|---|---|
| `results/main_laptop/` | Main experiment: Llama 3.1 8B and Qwen 2.5 7B on a laptop (20 Nov 2026) |
| `results/followup_size/` | Follow-up: Qwen 2.5 7B, 14B and 32B on Kaggle |
| `results/followup_generation/` | Follow-up: Qwen 3 8B, 14B and 32B and Qwen 3.8 27B on Kaggle (thinking off) |
| `results/pilots/` | Pilot runs (2 posts per model). **Not part of the results.** |

Each run has metadata files (`run_metadata.json` for the main run, `results_kaggle_*_metadata.json` and `results_run2_*_metadata.json` for the follow-ups) with the exact model IDs (digests), settings, prompt wordings and timestamps.

## Data

The posts come from the **Senti4SD gold standard**:

> F. Calefato, F. Lanubile, F. Maiorano and N. Novielli, "Sentiment Polarity Detection for Software Development," *Empirical Software Engineering*, vol. 23, no. 3, pp. 1352–1382, 2018, doi: 10.1007/s10664-017-9546-9.

Original dataset: https://github.com/collab-uniba/Senti4SD (file `Senti4SD_GoldStandard_and_DSM/Senti4SD_GoldStandard_EmotionPolarity.xlsx`, sheet "Annotations and Gold Labels"). It is released under the MIT licence; a copy is in `data/SENTI4SD_LICENSE.txt`. The post texts in `data/sample.csv` and in the results files remain under that licence.

`data/sample.csv` contains 300 posts (100 positive, 100 negative, 100 neutral), drawn with random seed 42 from posts of at most 1,000 characters. Columns: `id`, `text`, `gold`.

## Settings

| Setting | Value |
|---|---|
| Wordings | 4 (`v1_plain`, `v2_formal`, `v3_casual`, `v4_reordered`; see `scripts/prompts.py`) |
| Repetitions | 3 per prompt |
| Temperature | 0 |
| Context length (`num_ctx`) | 2,048 tokens |
| Max output tokens (`num_predict`) | 20 |
| Runtime | Ollama (https://ollama.com), 4-bit (Q4_K_M) models |

Answers are mapped to labels by taking the first of the words *positive*, *negative* or *neutral* in the answer; answers with none of them are marked `invalid`.

## How to rerun

Requirements: Python 3.10+, [Ollama](https://ollama.com), and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**1. Download the models** (check with `ollama list` that the IDs match the metadata files):

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

**2. (Optional) Draw the sample again.** This reproduces `data/sample.csv` from the original xlsx file:

```bash
python scripts/sample_data.py --input Senti4SD_GoldStandard_EmotionPolarity.xlsx --sheet "Annotations and Gold Labels" --text-col Text --label-col "Final Label (majority voting)" --per-label 100 --seed 42 --output data/sample.csv
```

**3. Pilot run** (2 posts, to check the setup):

```bash
python scripts/run_experiment.py --models llama3.1:8b --sample data/sample.csv --limit 2 --output pilot.csv
```

**4. Full run:**

```bash
python scripts/run_experiment.py --models llama3.1:8b qwen2.5:7b --sample data/sample.csv --repeats 3 --output results.csv
```

300 posts × 4 wordings × 3 repetitions = 3,600 requests per model. The script writes one row per answer and a `results_metadata.json` file with model IDs and settings. If it stops, rerun the same command with `--resume`.

For Qwen 3 and Qwen 3.8, add `--think off` to disable the thinking mode.

**5. Analyse:**

```bash
python scripts/analyze.py --input results.csv
```

This prints accuracy per wording, repeat consistency and wording consistency, and writes `summary_accuracy.csv` and `summary_stability.csv`.

The follow-up runs were done on Kaggle (2× NVIDIA Tesla T4) with the notebooks in `notebooks/`: `followup_size_qwen25.ipynb` (Qwen 2.5 7B/14B/32B) and `followup_generation_qwen3.ipynb` (Qwen 3 and Qwen 3.8, thinking off). The notebooks include their original outputs; only the model download progress bars were removed to keep the files small.

**Script versions:** the main laptop run used an earlier version of `run_experiment.py` that did not yet have the `--think`, `--resume` and `--metadata` options and always wrote its metadata to `run_metadata.json`. The generation settings (temperature, `num_ctx`, `num_predict`, repetitions, wordings, parsing rule) are identical in both versions. This repository contains the later version, which was used for the Kaggle runs.

## Note on reproducibility

Even at temperature 0, outputs may differ slightly between runs and hardware. In this study, 99.7% of Qwen 2.5 7B's labels were identical between the laptop and Kaggle.

## Use of generative AI

The scripts and notebooks in this repository were written with the help of an AI assistant (Claude, Anthropic), and the four prompt wordings were drafted with its help. All runs and results were produced and checked by the author.

## Licence

Code: MIT (see `LICENSE`). Senti4SD post texts: MIT, © 2017 Collaborative Development Group (the Senti4SD authors; see `data/SENTI4SD_LICENSE.txt`).
