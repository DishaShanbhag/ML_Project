# SVG Scaling Laws

**CS-GY 6923 | Disha D Shanbhag (ds8223 / N11461466) | NYU Tandon School of Engineering**

Decoder-only transformer language models trained on SVG code to study neural scaling laws and the effect of Maximal Update Parameterization (muP) on learning rate transfer.

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
    outputs/            # download from Drive
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

Each notebook reads and writes to `project/`. Update `PROJECT_DIR` at the top of each file if your working directory differs.

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
