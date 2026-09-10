# Source provenance

The public entry points were adapted from the accepted DEG2MOL implementation.
Absolute workstation paths were replaced with repository-relative paths; the
accepted architecture and training algorithm were retained.

| Public entry point | Private working filename | SHA-256 |
|---|---|---|
| `train.py` | `02_2_flow_DEG_condition_ema.py` | `28083f3920cf478df9083d37aaf2279a7730201c27990c3bf766774b91973fa3` |
| `test.py` | `03_2_evaluate.py` | `2b4ca6daa8c57400cf8d3d3c3e30efb05a47933b212e35d55f207ff073d6e2df` |
| `inference.py` | `04_inference.py` | `7238b6afbb528f6033733a6f897ed2a8b7e6ddd07d9be5529ffe3d93dfdc9c1b` |

## Accepted checkpoints

| Split | Original run directory | Zero-based epoch | Validation loss | SHA-256 |
|---|---|---:|---:|---|
| Scaffold | `flow_final_20260324-1314_20260324_131413` | 61 | 1.64073692519822 | `6714ad14bfa57156a3786217358de5292d319be1a41d9ed233b576ba4847b546` |
| Random | `flow_final_20260324-1625_20260324_162544` | 112 | 1.019520669057566 | `fdcb6ee89ceae42e7373bf8262d02fd1ad918f7db5dda92d36dcb3617ff6d421` |

The two run configurations use the same architecture: 10,280 input genes,
DEGMON AE dimensions `[10280, 2011, 1614, 1075]`, 64-dimensional condition and
molecule latents, six 512-dimensional Gated MLP blocks, sum conditioning,
classifier-free condition dropout 0.3, and EMA decay 0.999. The public relative
configurations are in `configs/accepted_scaffold.json` and
`configs/accepted_random.json`.

ScafVAE is pinned as the `third_party/ScafVAE` git submodule at commit
`21574304f8ee9948f70820e55e423c9da3e6694d`.

## Reproducibility fixes in the public entry points

- The random `null_condition` replacement in the working evaluation scripts is
  removed; the zero-valued buffer saved in each accepted checkpoint is retained.
- Duplicate molecule identifiers stay aligned by dataframe row during training.
- ScafVAE weights are loaded from the explicit relative checkpoint instead of a
  package-global parameter path.
- Test-set scaffold availability filtering is retained for accepted-cohort
  parity; inference no longer loads unused molecular feature files.
- Python, NumPy, and PyTorch are seeded, and output paths are separated from all
  input data paths.
