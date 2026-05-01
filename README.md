# SVG Scaling Laws

**CS-GY 6923 | Disha D Shanbhag (ds8223 / N11461466) | NYU Tandon School of Engineering**

This project investigates whether neural scaling laws hold for transformer language models trained on structured, non-linguistic data -- specifically Scalable Vector Graphics (SVG) code. Five decoder-only transformer models ranging from 1.35M to 88.4M parameters are trained on a corpus of ~284K SVG files (~116M tokens) drawn from three HuggingFace datasets.

Under standard parameterization (SP), validation loss is non-monotonic with scale: larger models perform worse when trained at the same fixed learning rate. Maximal Update Parameterization (muP) corrects this by enabling zero-shot learning rate transfer across model widths, restoring smooth power-law scaling (α = 0.866, R² = 0.960). The best model (XL, 91.5M params, 5 epochs) achieves a test perplexity of 2.51 and generates structurally valid SVGs in 82.6% of samples.

---

## Large Files

Data and model checkpoint files are too large for GitHub and are hosted on Google Drive:

**[Download project/data/ and project/outputs/ here](https://drive.google.com/drive/folders/1sZYy2dAzxUwfLs1l5quaqM_8XeGLrRle?usp=sharing)**

Place the downloaded folders at `project/data/` and `project/outputs/` before running any notebook.

---

## Repository Structure

```
ML_Project/
  notebooks/
    Part1_Data_Preprocessing.ipynb
    Part2_Scaling_Study.ipynb
    Part3_MuP_Scaling.ipynb
    Part4_Generation.ipynb
    Part5_Analysis.ipynb
  configs/
    model_configs.json
  project/
    data/               # download from Drive
    outputs/
      part1_summary.json      # dataset statistics (vocab size, split sizes, seq lengths)
      part2_summary.json      # SP scaling results and power law fit
      part3_summary.json      # muP scaling results and extrapolation
      part4_summary.json      # generation metrics and evaluation
      ...                     # model checkpoints, plots, generated SVGs (download from Drive)
    plots/
  requirements.txt
  report.pdf
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8, on NVIDIA A100-SXM4-40GB.

---

## Running the Notebooks

Open any notebook in JupyterLab or VS Code and run cells interactively, or execute them from the command line in order:

```bash
jupyter nbconvert --to notebook --execute notebooks/Part1_Data_Preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/Part2_Scaling_Study.ipynb
jupyter nbconvert --to notebook --execute notebooks/Part3_MuP_Scaling.ipynb
jupyter nbconvert --to notebook --execute notebooks/Part4_Generation.ipynb
```

Each notebook reads and writes to `project/`. The notebooks were originally developed with a different directory layout, so **update `PROJECT_DIR` at the top of each notebook** before running:

```python
# Original (do not use)
PROJECT_DIR = Path("ml_final_project")

# Change to this (matches the repo structure)
PROJECT_DIR = Path("project")
```

Run from the repo root (`ML_Project/`) so the relative path resolves correctly.

---

## Results

| Model  | Params  | SP Val Loss | muP Val Loss |
|--------|---------|-------------|--------------|
| Tiny   | 1.35M   | 1.229       | 1.221        |
| Small  | 3.51M   | 1.051       | 1.074        |
| Medium | 12.32M  | 0.947       | 1.025        |
| Large  | 33.75M  | 1.067       | 0.941        |
| XL     | 88.40M  | 1.367       | 0.975        |

- SP scaling is non-monotonic under a fixed learning rate (R² = 0.041)
- muP restores smooth power-law scaling: α = 0.866, R² = 0.960
- Best model (XL muP, 5 epochs): test perplexity = 2.51, SVG validity = 82.6%

---

## References

- Transformer architecture adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- muP implemented using the [Microsoft mup package](https://github.com/microsoft/mup)
