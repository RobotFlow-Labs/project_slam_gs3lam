"""Release regression gates for GS3LAM.

Loads an evaluation report JSON and checks every metric against the paper
thresholds defined in ``PaperThresholds``.  Returns a structured pass/fail
summary suitable for CI gating and rich console output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from anima_slam_gs3lam.version import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper thresholds — all values taken from PRD Section 10 / ASSETS.md
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    """Whether a metric should be above or below the threshold."""

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


@dataclass(frozen=True)
class MetricThreshold:
    """A single paper-target threshold."""

    name: str
    threshold: float
    direction: Direction
    unit: str = ""

    def passes(self, value: float) -> bool:
        if self.direction is Direction.HIGHER_BETTER:
            return value >= self.threshold
        return value <= self.threshold


@dataclass(frozen=True)
class PaperThresholds:
    """All paper-target thresholds for GS3LAM release gating."""

    thresholds: tuple[MetricThreshold, ...] = (
        # Replica rendering quality
        MetricThreshold("replica_psnr", 35.0, Direction.HIGHER_BETTER, "dB"),
        MetricThreshold("replica_ssim", 0.985, Direction.HIGHER_BETTER, ""),
        MetricThreshold("replica_lpips", 0.065, Direction.LOWER_BETTER, ""),
        # Replica semantics
        MetricThreshold("replica_miou", 95.0, Direction.HIGHER_BETTER, "%"),
        # Replica tracking
        MetricThreshold("replica_ate_cm", 0.50, Direction.LOWER_BETTER, "cm"),
        # ScanNet rendering quality
        MetricThreshold("scannet_psnr", 21.5, Direction.HIGHER_BETTER, "dB"),
        # ScanNet tracking
        MetricThreshold("scannet_ate_cm", 12.5, Direction.LOWER_BETTER, "cm"),
        # Replica runtime
        MetricThreshold("replica_fps", 90.0, Direction.HIGHER_BETTER, "FPS"),
    )

    def lookup(self, name: str) -> MetricThreshold | None:
        for t in self.thresholds:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """Result of checking a single metric against its threshold."""

    name: str
    value: float
    threshold: float
    direction: Direction
    unit: str
    passed: bool
    delta: float  # positive = better than threshold


@dataclass
class GateSummary:
    """Aggregate gate pass/fail summary."""

    version: str = __version__
    total: int = 0
    passed: int = 0
    failed: int = 0
    missing: int = 0
    results: list[MetricResult] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.missing == 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total * 100.0


# ---------------------------------------------------------------------------
# Core gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gates(
    report: dict[str, Any],
    thresholds: PaperThresholds | None = None,
) -> GateSummary:
    """Check ``report`` values against ``thresholds``.

    Parameters
    ----------
    report:
        Flat dict mapping metric name to float value, e.g.
        ``{"replica_psnr": 35.8, "replica_ssim": 0.987, ...}``.
    thresholds:
        Paper thresholds. Defaults to ``PaperThresholds()``.

    Returns
    -------
    GateSummary with per-metric pass/fail and aggregate status.
    """
    if thresholds is None:
        thresholds = PaperThresholds()

    summary = GateSummary()

    for metric_thresh in thresholds.thresholds:
        summary.total += 1
        value = report.get(metric_thresh.name)

        if value is None:
            summary.missing += 1
            summary.missing_metrics.append(metric_thresh.name)
            logger.warning("Metric %s not found in report", metric_thresh.name)
            continue

        value = float(value)
        passed = metric_thresh.passes(value)

        if metric_thresh.direction is Direction.HIGHER_BETTER:
            delta = value - metric_thresh.threshold
        else:
            delta = metric_thresh.threshold - value

        result = MetricResult(
            name=metric_thresh.name,
            value=value,
            threshold=metric_thresh.threshold,
            direction=metric_thresh.direction,
            unit=metric_thresh.unit,
            passed=passed,
            delta=delta,
        )
        summary.results.append(result)

        if passed:
            summary.passed += 1
        else:
            summary.failed += 1

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_summary_table(summary: GateSummary) -> str:
    """Render a human-readable table of gate results."""
    lines: list[str] = [
        "",
        "=" * 76,
        f"  GS3LAM Release Gate Report  (v{summary.version})",
        "=" * 76,
        "",
        f"  {'Metric':<22} {'Value':>10} {'Thresh':>10} {'Delta':>10} {'Status':>8}",
        f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}",
    ]

    for r in summary.results:
        sign = "+" if r.delta >= 0 else ""
        lines.append(
            f"  {r.name:<22} {r.value:>9.4f}{'':<1} "
            f"{r.threshold:>9.4f}{'':<1} "
            f"{sign}{r.delta:>8.4f}{'':<1} "
            f"{'[OK]' if r.passed else '[!!]':>6}"
        )

    if summary.missing_metrics:
        lines.append("")
        lines.append(f"  Missing metrics ({summary.missing}):")
        for name in summary.missing_metrics:
            lines.append(f"    - {name}")

    lines.append("")
    lines.append(f"  {'-' * 66}")
    verdict = "PASS" if summary.all_passed else "FAIL"
    lines.append(
        f"  Total: {summary.total}  |  "
        f"Passed: {summary.passed}  |  "
        f"Failed: {summary.failed}  |  "
        f"Missing: {summary.missing}  |  "
        f"Verdict: {verdict}"
    )
    lines.append(f"  Pass rate: {summary.pass_rate:.1f}%")
    lines.append("=" * 76)
    lines.append("")

    return "\n".join(lines)


def summary_to_dict(summary: GateSummary) -> dict[str, Any]:
    """Serialize a ``GateSummary`` to a JSON-compatible dict."""
    return {
        "version": summary.version,
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "missing": summary.missing,
        "all_passed": summary.all_passed,
        "pass_rate": round(summary.pass_rate, 2),
        "missing_metrics": summary.missing_metrics,
        "results": [
            {
                "name": r.name,
                "value": r.value,
                "threshold": r.threshold,
                "direction": r.direction.value,
                "unit": r.unit,
                "passed": r.passed,
                "delta": round(r.delta, 6),
            }
            for r in summary.results
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GS3LAM release regression gate check",
    )
    parser.add_argument(
        "report",
        type=str,
        help="Path to evaluation report JSON with metric values",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write gate summary JSON to this path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing metrics as failures (exit 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report_path = Path(args.report)
    if not report_path.exists():
        logger.error("Report file not found: %s", report_path)
        return 1

    report = json.loads(report_path.read_text())
    summary = evaluate_gates(report)

    print(format_summary_table(summary))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary_to_dict(summary), indent=2) + "\n")
        logger.info("Gate summary written to %s", out)

    if summary.failed > 0:
        return 1
    if args.strict and summary.missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
