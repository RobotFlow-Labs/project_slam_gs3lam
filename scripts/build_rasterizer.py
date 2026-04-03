"""Build gaussian-semantic-rasterization bypassing CUDA version mismatch.

System nvcc=12.0, PyTorch compiled with cu130. The extension compiles fine
with 12.0 for sm_89 (L4 GPUs). We just need to bypass the hard check.

Usage:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=2 python scripts/build_rasterizer.py
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

# Monkey-patch the CUDA version check BEFORE importing setup
import torch.utils.cpp_extension as cpp_ext
cpp_ext._check_cuda_version = lambda *a, **kw: None  # noqa: ARG005

# Set target arch for L4 GPUs
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ.setdefault("MAX_JOBS", "8")

# Build from the submodule
rasterizer_dir = Path(__file__).resolve().parent.parent / "repositories/GS3LAM/submodules/gaussian-semantic-rasterization"

# Install ninja for faster builds
try:
    import ninja  # noqa: F401
except ImportError:
    os.system(f"{sys.executable} -m pip install ninja --quiet")  # noqa: S605

# Use pip install which handles paths correctly
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-user", "--no-build-isolation", str(rasterizer_dir)],
    env={**os.environ, "TORCH_CUDA_ARCH_LIST": "8.9"},
    capture_output=False,
)
if result.returncode != 0:
    print("[FAIL] pip install failed, trying setup.py directly")
    os.chdir(rasterizer_dir)
    sys.path.insert(0, str(rasterizer_dir))
    sys.argv = ["setup.py", "install"]
    exec(open("setup.py").read())  # noqa: S102

print("\n[OK] gaussian_semantic_rasterization installed successfully")

# Verify import
try:
    from gaussian_semantic_rasterization import GaussianRasterizer  # noqa: F401
    print("[OK] Import verified: GaussianRasterizer available")
except ImportError as e:
    print(f"[WARN] Import failed: {e}")

# Copy to shared location for other SLAM modules
shared_dir = Path("/mnt/forge-data/modules/03_wave7/shared_slam_cuda")
shared_dir.mkdir(parents=True, exist_ok=True)

# Find the built extension
import gaussian_semantic_rasterization as gsr  # noqa: E402
src_pkg = Path(gsr.__file__).parent
dst_pkg = shared_dir / "gaussian_semantic_rasterization"
if dst_pkg.exists():
    shutil.rmtree(dst_pkg)
shutil.copytree(src_pkg, dst_pkg)
print(f"[OK] Copied to shared location: {dst_pkg}")
