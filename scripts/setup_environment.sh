#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "${repo_dir}" submodule update --init --recursive
conda env create --file "${repo_dir}/environment.yml"

eval "$(conda shell.bash hook)"
conda activate deg2mol

python -m pip install \
  torch-scatter==2.0.9 \
  torch-sparse==0.6.13 \
  torch-cluster==1.6.0 \
  torch-spline-conv==1.2.1 \
  --find-links https://data.pyg.org/whl/torch-1.10.2+cu113.html
python -m pip install torch-geometric==2.0.4
python -m pip install \
  "openfold @ git+https://github.com/aqlaboratory/openfold@e938c184a291bf053af3b14c1e3e8bb29aee57e2"
python -m pip install --no-deps --editable "${repo_dir}/third_party/ScafVAE"

python -c "import torch, ScafVAE; print('torch', torch.__version__); print('ScafVAE import: OK')"
