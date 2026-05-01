# CS-GY 6923 Optional Project: SVG Scaling Laws

Decoder-only transformer language models trained on SVG code to study neural scaling laws
and the effect of Maximal Update Parameterization (muP) on learning rate transfer.

## Repository Structure

```
ml_final_project/
    data/
        svg_tokenizer.json      # Trained BPE tokenizer (vocab size 4096)
        train.bin               # Binary token file for training (uint16)
        val.bin                 # Binary token file for validation
        test.bin                # Binary token file for testing
        train.jsonl             # JSONL with SVG text and token IDs (train)
        val.jsonl               # JSONL with SVG text and token IDs (val)
        test.jsonl              # JSONL with SVG text and token IDs (test)
    outputs/
        dataset_statistics.json    # Dataset stats (token counts, seq lengths)
        lr_sweep_results.json      # SP learning rate sweep results
        mup_lr_sweep_results.json  # muP learning rate sweep results
        part2_summary.json         # Full Part 2 results including power law fit
        part3_summary.json         # Full Part 3 results including muP comparison
        part4_summary.json         # Full Part 4 results including generation metrics
        scaling_results.json       # Detailed per-step training logs (SP, all models)
        mup_scaling_results.json   # Detailed per-step training logs (muP, all models)
        example_svgs/              # Example SVGs at different complexity levels
        generated/
            unconditional/         # 18 unconditional generated SVG samples
            prefix/                # 10 prefix SVG files (5 prefixes + 5 completions)
        best_model_epoch*.pt       # Per-epoch checkpoints of the XL muP model
        best_model_final.pt        # Final checkpoint after 5 epochs
    plots/
        seq_length_histogram.png       # Sequence length distribution
        lr_sweep.png                   # LR sweep on Tiny SP model
        lr_sweep_sp_vs_mup.png         # SP vs muP LR sweep comparison
        scaling_plot.png               # SP scaling law plot
        scaling_sp_vs_mup.png          # SP vs muP scaling comparison
        scaling_extrapolation.png      # Extrapolation to 10x XL
        training_curves.png            # Training loss curves (SP)
        mup_training_curves.png        # Training loss curves (muP)
        best_model_training_curve.png  # XL muP 5-epoch training curve
        grid_unconditional_t0.5.png    # Generated samples at T=0.5
        grid_unconditional_t0.8.png    # Generated samples at T=0.8
        grid_unconditional_t1.0.png    # Generated samples at T=1.0
        grid_temperature_comparison.png
        grid_prefix_completions.png

Part1_Data_Preprocessing.ipynb   # Data download, cleaning, tokenization
Part2_Scaling_Study.ipynb         # Transformer training, LR sweep, scaling plot
Part3_MuP_Scaling.ipynb           # muP training, comparison, extrapolation
Part4_Generation (1).ipynb        # Best model training, generation, evaluation
report.tex                        # LaTeX report source
requirements.txt                  # Python dependencies
```

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

### Running Order

Run the notebooks in order. Each notebook saves its outputs to `ml_final_project/` and the
next notebook loads from there.

**Part 1** - Data preprocessing (required before any other part):
```
jupyter nbconvert --to notebook --execute Part1_Data_Preprocessing.ipynb
```

**Part 2** - Standard parameterization scaling study:
```
jupyter nbconvert --to notebook --execute Part2_Scaling_Study.ipynb
```

**Part 3** - muP scaling and extrapolation (requires Part 2 outputs):
```
jupyter nbconvert --to notebook --execute Part3_MuP_Scaling.ipynb
```

**Part 4** - Best model training and generation (requires Parts 2 and 3 outputs):
```
jupyter nbconvert --to notebook --execute "Part4_Generation (1).ipynb"
```

All notebooks can also be run interactively cell by cell in JupyterLab.

### Compute Requirements

- Parts 1-3 run on a single NVIDIA A100 (40 GB) in approximately 12 hours total.
- Part 4 (5 epochs of XL) requires approximately 6.25 additional hours on the same GPU.
- Estimated peak GPU memory: 17.7 GB for the XL model.

### Path Configuration

Update `PROJECT_DIR` at the top of each notebook if you use a different working directory:
```python
PROJECT_DIR = Path("ml_final_project")  # change this if needed
```

## Key Results

| Model | Params | SP Val Loss | muP Val Loss |
|-------|--------|-------------|--------------|
| Tiny  | 1.35M  | 1.229       | 1.221        |
| Small | 3.51M  | 1.051       | 1.074        |
| Medium| 12.32M | 0.947       | 1.025        |
| Large | 33.75M | 1.067       | 0.941        |
| XL    | 88.40M | 1.367       | 0.975        |

- muP scaling exponent alpha = 0.866, R^2 = 0.960
- SP scaling is non-monotonic (degenerate fit, R^2 = 0.041)
- XL muP model (5 epochs): test perplexity = 2.51, XML validity = 82.6%

## Reference Code

The transformer model is based on the nanoGPT architecture by Andrej Karpathy:
https://github.com/karpathy/nanoGPT

muP is implemented using the Microsoft mup package:
https://github.com/microsoft/mup
