# SLAM-GS3LAM — Asset Manifest

## Paper
- Title: GS3LAM: Gaussian Semantic Splatting SLAM
- ACM MM 2024 DOI: https://doi.org/10.1145/3664647.3680739
- arXiv: https://arxiv.org/abs/2603.27781
- Authors: Linfei Li, Lin Zhang, Zhong Wang, Ying Shen
- Correct local PDF: `papers/2603.27781_GS3LAM.pdf`
- Incorrect local PDF to ignore: `papers/2503.15909_GS3LAM.pdf`
  This file contains an unrelated quantum teleportation paper and must not be used for implementation planning.

## Status: ALMOST

## Reference Implementation
| Asset | Source | Path | Status |
|---|---|---|---|
| GS3LAM official repo | https://github.com/lif314/GS3LAM | `repositories/GS3LAM/` | DONE |
| CUDA rasterizer submodule | official repo submodule | `repositories/GS3LAM/submodules/gaussian-semantic-rasterization/` | DONE |

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---|---|---|---|
| Semantic decoder | 1x1 conv, 16->256 logits | trained in repo, no published checkpoint required | `/mnt/forge-data/models/slam/gs3lam/decoder.pt` | MISSING |
| SG-Field checkpoints | scene-specific | produced by training/incremental optimization | `/mnt/forge-data/models/slam/gs3lam/checkpoints/` | MISSING |
| External semantic label generator for TUM | DEVA pseudo labels | https://huggingface.co/datasets/3David14/TUM-DEVA | `/mnt/forge-data/datasets/slam/gs3lam/TUM-DEVA/` | MISSING |

## Datasets
| Dataset | Scenes / Split | Source | Path | Status |
|---|---|---|---|---|
| Replica + semantic labels | `room0`, `room1`, `room2`, `office0`, `office1`, `office2`, `office3`, `office4` | https://huggingface.co/datasets/3David14/GS3LAM-Replica | `/mnt/forge-data/datasets/slam/gs3lam/Replica/` | MISSING |
| ScanNet RGB-D + labels | `scene0000_00`, `scene0059_00`, `scene0106_00`, `scene0169_00`, `scene0181_00`, `scene0207_00` | http://www.scan-net.org/ plus frame extraction from `.sens` | `/mnt/forge-data/datasets/slam/gs3lam/scannet/` | MISSING |
| TUM-RGBD + pseudo semantics | `freiburg1_desk` in the paper/repo | https://huggingface.co/datasets/3David14/TUM-DEVA | `/mnt/forge-data/datasets/slam/gs3lam/TUM-DEVA/` | MISSING |

## Build Dependencies
| Asset | Source | Path on Server | Status |
|---|---|---|---|
| CUDA 11.7 toolchain | NVIDIA | `/usr/local/cuda-11.7/` | UNKNOWN |
| PyTorch 1.13.1 + cu117 parity env | official repo README | `/mnt/forge-data/envs/gs3lam-cu117/` | MISSING |
| PyTorch >= 2.0 ANIMA port env | local `uv` environment | `.venv/` | PENDING |

## Hyperparameters (from paper + repo)
| Param | Value | Paper Section |
|---|---|---|
| semantic feature dim (`N_sem`) | 16 | §3.2, Eq. (1), Fig. 3 |
| semantic class dim (`K_sem`) | 256 in repo configs | §3.2.2, App C.1 |
| tracking LR, pose rotation | 0.0004 | App C.1 |
| tracking LR, pose translation | 0.002 | App C.1 |
| mapping LR, position | 0.0001 | App C.1 |
| mapping LR, color | 0.0025 | App C.1 |
| mapping LR, rotation | 0.001 | App C.1 |
| mapping LR, opacity | 0.05 | App C.1 |
| mapping LR, scale | 0.001 | App C.1 |
| mapping LR, semantic feature | 0.0025 | App C.1 |
| tracking loss weights | color `0.5`, depth `1.0`, semantic `0.001` | §3.4, Eq. (21), App C.1 |
| mapping loss weights | color `0.5`, depth `1.0`, semantic `0.01`, big-scale `0.01`, small-scale `0.001` | §3.3.4, Eq. (18), App C.1 |
| Replica iterations | tracking `40`, mapping `60` | App C.1 |
| ScanNet iterations | tracking `100`, mapping `30` | App C.1 |

## Repo-Observed Hyperparameters To Reconcile
| Param | Repo Value | Paper Value | Note |
|---|---|---|---|
| ScanNet tracking iterations | `200` | `100` | `repositories/GS3LAM/configs/Scannet/scannet.py` diverges from App C.1 |
| ScanNet mapping iterations | `60` | `30` | same divergence as above |
| TUM tracking iterations | `360` | not reported in main paper | treat as extra repo-only experiment |
| TUM mapping iterations | `150` | not reported in main paper | treat as extra repo-only experiment |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---|---|
| Replica avg | PSNR | 36.26 dB | >= 35.0 dB |
| Replica avg | SSIM | 0.989 | >= 0.985 |
| Replica avg | LPIPS | 0.052 | <= 0.065 |
| Replica avg | ATE RMSE | 0.37 cm | <= 0.50 cm |
| Replica avg | mIoU | 96.63 % | >= 95.0 % |
| ScanNet avg | PSNR | 22.86 dB | >= 21.5 dB |
| ScanNet avg | SSIM | 0.868 | >= 0.84 |
| ScanNet avg | LPIPS | 0.222 | <= 0.26 |
| ScanNet avg | ATE RMSE | 11.24 cm | <= 12.5 cm |
| Replica render speed | FPS | 109.12 | >= 90 FPS |
| Replica office0 runtime | Mapping iteration | 55 ms | <= 70 ms |
| Replica office0 runtime | Tracking iteration | 89 ms | <= 110 ms |

## Hardware Requirements
| Requirement | Paper / Repo Evidence | Target |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3090 | CUDA workstation or cloud GPU |
| CPU | AMD EPYC 7302 16-Core | any modern x86_64 server CPU |
| Rasterizer backend | CUDA custom extension | must remain CUDA-first for faithful reproduction |
| RGB-D sensor input | Replica / ScanNet / TUM RGB-D | RGB-D stream mandatory |
| Semantics input | dataset labels or DEVA pseudo labels | dense per-pixel labels mandatory |

## Known Risks
- The project scaffold still contains stale `TSUKUYOMI` naming in source/package files and configs.
- The official repo is CUDA-first and not immediately compatible with the repo convention of Python `>=3.10` plus `uv`.
- The paper text and repo differ on ScanNet iteration counts; PRDs target paper-faithful defaults and treat repo drift as a validation item.
