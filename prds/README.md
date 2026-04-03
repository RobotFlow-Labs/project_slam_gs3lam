# SLAM-GS3LAM PRD Suite

This directory contains the execution-oriented PRDs for rebuilding GS3LAM inside this ANIMA module while staying paper-faithful to `arXiv:2603.27781` and grounded in `repositories/GS3LAM/`.

## PRD Index
- [PRD-01 Foundation](PRD-01-foundation.md)
- [PRD-02 Core Model](PRD-02-core-model.md)
- [PRD-03 Inference](PRD-03-inference.md)
- [PRD-04 Evaluation](PRD-04-evaluation.md)
- [PRD-05 API & Docker](PRD-05-api-docker.md)
- [PRD-06 ROS2 Integration](PRD-06-ros2-integration.md)
- [PRD-07 Production](PRD-07-production.md)

## Build Order
1. PRD-01
2. PRD-02
3. PRD-03
4. PRD-04
5. PRD-05
6. PRD-06
7. PRD-07

## Paper Fidelity Rules
- Use the correct paper: `papers/2603.27781_GS3LAM.pdf`
- Treat `papers/2503.15909_GS3LAM.pdf` as invalid for this module
- Prefer the paper when the paper and repo disagree on hyperparameters
- Document every deliberate ANIMA adaptation explicitly
