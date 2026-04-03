#!/bin/bash
# Build gaussian-semantic-rasterization CUDA extension
# Bypasses CUDA version mismatch (system nvcc 12.0 vs PyTorch cu130)
set -e

VENV="/mnt/forge-data/modules/03_wave7/project_slam_gs3lam/.venv"
RAST_DIR="/mnt/forge-data/modules/03_wave7/project_slam_gs3lam/repositories/GS3LAM/submodules/gaussian-semantic-rasterization"
SHARED_DIR="/mnt/forge-data/modules/03_wave7/shared_slam_cuda"

source "$VENV/bin/activate"

# Monkey-patch the CUDA version check and build
cd "$RAST_DIR"

export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=2

python -c "
import os, sys
os.chdir('$RAST_DIR')

# Bypass CUDA version check
import torch.utils.cpp_extension as ext
ext._check_cuda_version = lambda *a, **kw: None

# Set __file__ for setup.py
__file__ = '$RAST_DIR/setup.py'
sys.argv = ['setup.py', 'install']
exec(open('$RAST_DIR/setup.py').read())
"

echo ""
echo "[BUILD] Verifying import..."
python -c "from gaussian_semantic_rasterization import GaussianRasterizer; print('[OK] GaussianRasterizer available')"

echo "[BUILD] Copying to shared location: $SHARED_DIR"
mkdir -p "$SHARED_DIR"
python -c "
import shutil, gaussian_semantic_rasterization as gsr
from pathlib import Path
src = Path(gsr.__file__).parent
dst = Path('$SHARED_DIR/gaussian_semantic_rasterization')
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'[OK] Shared at {dst}')
"

echo "[DONE] gaussian-semantic-rasterization built and shared"
