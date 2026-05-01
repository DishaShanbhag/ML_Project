# CS-GY 6923 Project: SVG Scaling Laws

**Disha D Shanbhag** | NetID: ds8223 / N11461466 | NYU Tandon School of Engineering

Decoder-only transformer language models trained on SVG code to study neural scaling laws
and the effect of Maximal Update Parameterization (muP) on learning rate transfer.

---

## Large Files (Not in Git)

Binary data files, tokenized corpora, and model checkpoints exceed GitHub's file-size
limits and are **not tracked in this repository**. They are available on Google Drive:

> **[project/data/ and project/outputs/ on Google Drive](https://drive.google.com/drive/folders/1sZYy2dAzxUwfLs1l5quaqM_8XeGLrRle?usp=sharing)**

Files excluded from git:

| Path | Description |
|------|-------------|
| `project/data/train.bin` | Binary token file for training (~221 MB) |
| `project/data/val.bin` | Binary token file for validation |
| `project/data/test.bin` | Binary token file for testing |
| `project/data/train.jsonl` | JSONL with SVG text and token IDs (train) |
| `project/data/val.jsonl` | JSONL with SVG text and token IDs (val) |
| `project/data/test.jsonl` | JSONL with SVG text and token IDs (test) |
| `project/data/svg_corpus.txt` | Raw concatenated SVG corpus |
| `project/outputs/best_model_epoch*.pt` | Per-epoch XL muP checkpoints (~340 MB each) |
| `project/outputs/best_model_final.pt` | Final checkpoint after 5 epochs |

Everything else (plots, JSON summaries, generated SVGs, notebooks, configs) is tracked
in git and available directly in this repo.

---

## Repository Structure

```
ML_Project/
  README.md
  requirements.txt
  report.pdf                        # Final project report

  notebooks/
    Part1_Data_Preprocessing.ipynb  # Data download, cleaning, tokenization
    Part2_Scaling_Study.ipynb       # Transformer training, LR sweep, scaling plot
    Part3_MuP_Scaling.ipynb         # muP training, comparison, extrapolation
    Part4_Generation.ipynb          # Best model training, generation, evaluation
    Part5_Analysis.ipynb            # Additional analysis

  configs/
    model_configs.json              # All model architecture and training hyperparameters

  project/
    data/                           # ** NOT IN GIT -- see Google Drive link above **
      svg_tokenizer.json            # Trained BPE tokenizer (vocab size 4096)
      train.bin / val.bin / test.bin
      train.jsonl / val.jsonl / test.jsonl
      svg_corpus.txt

    outputs/
      dataset_statistics.json       # Dataset stats (token counts, seq lengths)
      lr_sweep_results.json         # SP learning rate sweep results
      mup_lr_sweep_results.json     # muP learning rate sweep results
      part2_summary.json            # Full Part 2 results including power law fit
      part3_summary.json            # Full Part 3 results including muP comparison
      part4_summary.json            # Full Part 4 results including generation metrics
      scaling_results.json          # Per-step training logs (SP, all models)
      mup_scaling_results.json      # Per-step training logs (muP, all models)
      example_svgs/                 # 5 example SVGs at p5/p25/p50/p75/p95 complexity
      generated/
        unconditional/              # 18 unconditional SVG samples (6 per temperature)
        prefix/                     # 10 prefix files (5 prefixes + 5 completions)
      best_model_epoch*.pt          # ** NOT IN GIT -- see Google Drive link above **
      best_model_final.pt           # ** NOT IN GIT -- see Google Drive link above **

    plots/
      seq_length_histogram.png
      lr_sweep.png
      lr_sweep_sp_vs_mup.png
      scaling_plot.png
      scaling_sp_vs_mup.png
      scaling_extrapolation.png
      training_curves.png
      mup_training_curves.png
      best_model_training_curve.png
      grid_unconditional_t0.5.png
      grid_unconditional_t0.8.png
      grid_unconditional_t1.0.png
      grid_temperature_comparison.png
      grid_prefix_completions.png
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets tokenizers transformers
pip install scipy matplotlib numpy tqdm
pip install lxml cairosvg Pillow
pip install mup
```

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8, on NVIDIA A100-SXM4-40GB.

### Downloading Large Files

Download the `data/` and `outputs/` folders from Google Drive and place them at
`project/data/` and `project/outputs/` respectively before running any notebook:

```
ML_Project/
  project/
    data/       <-- place downloaded data/ contents here
    outputs/    <-- place downloaded outputs/ contents here
```

---

## Running the Notebooks

Notebooks are in `notebooks/` and must be run in order. Each notebook reads from and
writes to `project/` using the `PROJECT_DIR` variable at the top of each file.

**Part 1** - Data preprocessing (required before any other part):
```bash
jupyter nbconvert --to notebook --execute notebooks/Part1_Data_Preprocessing.ipynb
```

**Part 2** - Standard parameterization scaling study:
```bash
jupyter nbconvert --to notebook --execute notebooks/Part2_Scaling_Study.ipynb
```

**Part 3** - muP scaling and extrapolation (requires Part 2 outputs):
```bash
jupyter nbconvert --to notebook --execute notebooks/Part3_MuP_Scaling.ipynb
```

**Part 4** - Best model training and generation (requires Parts 2 and 3 outputs):
```bash
jupyter nbconvert --to notebook --execute notebooks/Part4_Generation.ipynb
```

**Part 5** - Additional analysis:
```bash
jupyter nbconvert --to notebook --execute notebooks/Part5_Analysis.ipynb
```

All notebooks can also be run interactively cell by cell in JupyterLab.

### Path Configuration

`PROJECT_DIR` at the top of each notebook controls where data and outputs are read/written:

```python
PROJECT_DIR = Path("project")  # relative to the repo root
```

---

## Compute Requirements

| Part | Description | Approx. Time | Peak GPU Memory |
|------|-------------|-------------|-----------------|
| 1 | Data preprocessing | ~30 min | CPU only |
| 2 | SP scaling study (5 models, 1 epoch each) | ~6 hr | 17.7 GB |
| 3 | muP scaling study (5 models, 1 epoch each) | ~6 hr | 17.7 GB |
| 4 | XL muP best model (5 epochs) | ~6.25 hr | 17.7 GB |

Hardware: single NVIDIA A100-SXM4-40GB (42.4 GB VRAM).

---

## Key Results

| Model | Params | SP Val Loss | muP Val Loss |
|-------|--------|-------------|--------------|
| Tiny  | 1.35M  | 1.229       | 1.221        |
| Small | 3.51M  | 1.051       | 1.074        |
| Medium| 12.32M | 0.947       | 1.025        |
| Large | 33.75M | 1.067       | 0.941        |
| XL    | 88.40M | 1.367       | 0.975        |

- SP scaling is non-monotonic under fixed LR (degenerate power law fit, R^2 = 0.041)
- muP restores smooth scaling: alpha = 0.866, R^2 = 0.960
- Best model (XL muP, 5 epochs): test perplexity = 2.51, SVG validity = 82.6% (19/23 samples)

---

## Reference Code

The transformer model is adapted from the nanoGPT architecture by Andrej Karpathy:
https://github.com/karpathy/nanoGPT

muP is implemented using the Microsoft mup package:
https://github.com/microsoft/mup
