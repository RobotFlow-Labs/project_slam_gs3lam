"""Paper-vs-reproduction report builder."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_paper_targets(path: str | Path) -> dict[str, float]:
    text = Path(path).read_text()
    patterns = {
        "replica_psnr": r"Replica avg \| PSNR \| ([0-9.]+)",
        "replica_ssim": r"Replica avg \| SSIM \| ([0-9.]+)",
        "replica_lpips": r"Replica avg \| LPIPS \| ([0-9.]+)",
        "replica_ate_cm": r"Replica avg \| ATE RMSE \| ([0-9.]+) cm",
        "replica_miou": r"Replica avg \| mIoU \| ([0-9.]+)",
        "scannet_psnr": r"ScanNet avg \| PSNR \| ([0-9.]+)",
        "scannet_ssim": r"ScanNet avg \| SSIM \| ([0-9.]+)",
        "scannet_lpips": r"ScanNet avg \| LPIPS \| ([0-9.]+)",
        "scannet_ate_cm": r"ScanNet avg \| ATE RMSE \| ([0-9.]+) cm",
        "replica_fps": r"Replica render speed \| FPS \| ([0-9.]+)",
    }
    targets: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            targets[key] = float(match.group(1))
    return targets


def build_gap_report(results: dict[str, float], paper_targets: dict[str, float]) -> str:
    lines = [
        "# GS3LAM Reproduction Gap Report",
        "",
        "| Metric | Paper | Reproduced | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, target in sorted(paper_targets.items()):
        reproduced = results.get(metric)
        if reproduced is None:
            lines.append(f"| {metric} | {target:.4f} | n/a | n/a |")
            continue
        delta = reproduced - target
        lines.append(f"| {metric} | {target:.4f} | {reproduced:.4f} | {delta:+.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-targets", required=True)
    args = parser.parse_args()

    targets = parse_paper_targets(args.paper_targets)
    print(build_gap_report({}, targets))


if __name__ == "__main__":
    main()
