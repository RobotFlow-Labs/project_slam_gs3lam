"""Tests for release regression gates."""

from __future__ import annotations

import json

import pytest

from anima_slam_gs3lam.release_checks import (
    Direction,
    GateSummary,
    MetricThreshold,
    PaperThresholds,
    evaluate_gates,
    format_summary_table,
    summary_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _passing_report() -> dict[str, float]:
    """A report that meets or exceeds all paper thresholds."""
    return {
        "replica_psnr": 36.2,
        "replica_ssim": 0.990,
        "replica_lpips": 0.050,
        "replica_miou": 96.5,
        "replica_ate_cm": 0.40,
        "scannet_psnr": 22.0,
        "scannet_ate_cm": 11.0,
        "replica_fps": 95.0,
    }


def _failing_report() -> dict[str, float]:
    """A report that fails every single paper threshold."""
    return {
        "replica_psnr": 30.0,
        "replica_ssim": 0.950,
        "replica_lpips": 0.100,
        "replica_miou": 80.0,
        "replica_ate_cm": 1.00,
        "scannet_psnr": 18.0,
        "scannet_ate_cm": 15.0,
        "replica_fps": 60.0,
    }


# ---------------------------------------------------------------------------
# MetricThreshold unit tests
# ---------------------------------------------------------------------------


class TestMetricThreshold:
    def test_higher_better_pass(self):
        t = MetricThreshold("psnr", 35.0, Direction.HIGHER_BETTER, "dB")
        assert t.passes(35.0) is True
        assert t.passes(36.0) is True

    def test_higher_better_fail(self):
        t = MetricThreshold("psnr", 35.0, Direction.HIGHER_BETTER, "dB")
        assert t.passes(34.9) is False

    def test_lower_better_pass(self):
        t = MetricThreshold("lpips", 0.065, Direction.LOWER_BETTER)
        assert t.passes(0.065) is True
        assert t.passes(0.060) is True

    def test_lower_better_fail(self):
        t = MetricThreshold("lpips", 0.065, Direction.LOWER_BETTER)
        assert t.passes(0.066) is False

    def test_exact_threshold_boundary(self):
        """Exact threshold value should always pass."""
        higher = MetricThreshold("m", 10.0, Direction.HIGHER_BETTER)
        lower = MetricThreshold("m", 10.0, Direction.LOWER_BETTER)
        assert higher.passes(10.0) is True
        assert lower.passes(10.0) is True


# ---------------------------------------------------------------------------
# PaperThresholds
# ---------------------------------------------------------------------------


class TestPaperThresholds:
    def test_has_all_expected_metrics(self):
        pt = PaperThresholds()
        names = {t.name for t in pt.thresholds}
        expected = {
            "replica_psnr",
            "replica_ssim",
            "replica_lpips",
            "replica_miou",
            "replica_ate_cm",
            "scannet_psnr",
            "scannet_ate_cm",
            "replica_fps",
        }
        assert names == expected

    def test_lookup_existing(self):
        pt = PaperThresholds()
        t = pt.lookup("replica_psnr")
        assert t is not None
        assert t.threshold == 35.0

    def test_lookup_missing(self):
        pt = PaperThresholds()
        assert pt.lookup("nonexistent") is None


# ---------------------------------------------------------------------------
# evaluate_gates
# ---------------------------------------------------------------------------


class TestEvaluateGates:
    def test_all_passing(self):
        summary = evaluate_gates(_passing_report())
        assert summary.all_passed is True
        assert summary.failed == 0
        assert summary.missing == 0
        assert summary.passed == summary.total

    def test_all_failing(self):
        summary = evaluate_gates(_failing_report())
        assert summary.all_passed is False
        assert summary.failed == summary.total
        assert summary.passed == 0

    def test_missing_metrics(self):
        report = {"replica_psnr": 36.0}  # only 1 of 8 metrics
        summary = evaluate_gates(report)
        assert summary.missing == 7
        assert summary.passed == 1
        assert summary.all_passed is False  # missing counts as not-passed

    def test_empty_report(self):
        summary = evaluate_gates({})
        assert summary.total == 8
        assert summary.missing == 8
        assert summary.passed == 0
        assert summary.all_passed is False

    def test_partial_pass(self):
        report = _passing_report()
        report["replica_psnr"] = 30.0  # fail this one
        summary = evaluate_gates(report)
        assert summary.passed == 7
        assert summary.failed == 1
        assert summary.all_passed is False

    def test_delta_higher_better(self):
        report = {"replica_psnr": 36.0}
        summary = evaluate_gates(report)
        psnr_result = next(r for r in summary.results if r.name == "replica_psnr")
        assert psnr_result.delta == pytest.approx(1.0, abs=1e-6)

    def test_delta_lower_better(self):
        report = {"replica_ate_cm": 0.40}
        summary = evaluate_gates(report)
        ate_result = next(r for r in summary.results if r.name == "replica_ate_cm")
        assert ate_result.delta == pytest.approx(0.10, abs=1e-6)

    def test_custom_thresholds(self):
        custom = PaperThresholds(
            thresholds=(
                MetricThreshold("my_metric", 50.0, Direction.HIGHER_BETTER, "%"),
            )
        )
        summary = evaluate_gates({"my_metric": 55.0}, thresholds=custom)
        assert summary.total == 1
        assert summary.passed == 1
        assert summary.all_passed is True

    def test_pass_rate(self):
        report = _passing_report()
        report["replica_psnr"] = 30.0  # fail 1 of 8
        summary = evaluate_gates(report)
        assert summary.pass_rate == pytest.approx(87.5, abs=0.1)


# ---------------------------------------------------------------------------
# GateSummary properties
# ---------------------------------------------------------------------------


class TestGateSummary:
    def test_all_passed_true_when_no_failures_or_missing(self):
        s = GateSummary(total=3, passed=3, failed=0, missing=0)
        assert s.all_passed is True

    def test_all_passed_false_when_failures(self):
        s = GateSummary(total=3, passed=2, failed=1, missing=0)
        assert s.all_passed is False

    def test_all_passed_false_when_missing(self):
        s = GateSummary(total=3, passed=2, failed=0, missing=1)
        assert s.all_passed is False

    def test_pass_rate_zero_when_empty(self):
        s = GateSummary(total=0, passed=0, failed=0, missing=0)
        assert s.pass_rate == 0.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_summary_table_contains_verdict(self):
        summary = evaluate_gates(_passing_report())
        table = format_summary_table(summary)
        assert "PASS" in table
        assert "Verdict: PASS" in table

    def test_format_summary_table_contains_fail(self):
        summary = evaluate_gates(_failing_report())
        table = format_summary_table(summary)
        assert "Verdict: FAIL" in table

    def test_format_missing_section(self):
        summary = evaluate_gates({"replica_psnr": 36.0})
        table = format_summary_table(summary)
        assert "Missing metrics" in table

    def test_summary_to_dict_roundtrips(self):
        summary = evaluate_gates(_passing_report())
        d = summary_to_dict(summary)
        # Verify JSON serializable
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["all_passed"] is True
        assert parsed["total"] == 8
        assert len(parsed["results"]) == 8

    def test_summary_to_dict_failing(self):
        summary = evaluate_gates(_failing_report())
        d = summary_to_dict(summary)
        assert d["all_passed"] is False
        assert d["failed"] == 8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_float_precision_at_boundary(self):
        """Values extremely close to threshold should behave correctly."""
        report = {"replica_psnr": 34.9999999}
        summary = evaluate_gates(report)
        psnr = next(r for r in summary.results if r.name == "replica_psnr")
        # 34.9999999 < 35.0, should fail
        assert psnr.passed is False

    def test_value_exactly_at_threshold(self):
        """Exact threshold value should pass for both directions."""
        report = {
            "replica_psnr": 35.0,
            "replica_lpips": 0.065,
        }
        summary = evaluate_gates(report)
        psnr = next(r for r in summary.results if r.name == "replica_psnr")
        lpips = next(r for r in summary.results if r.name == "replica_lpips")
        assert psnr.passed is True
        assert lpips.passed is True

    def test_negative_delta(self):
        """Failing metric should have negative delta."""
        report = {"replica_psnr": 30.0}
        summary = evaluate_gates(report)
        psnr = next(r for r in summary.results if r.name == "replica_psnr")
        assert psnr.delta < 0

    def test_report_with_extra_keys(self):
        """Extra keys in report should be ignored, not cause errors."""
        report = _passing_report()
        report["unknown_metric"] = 999.0
        summary = evaluate_gates(report)
        assert summary.all_passed is True

    def test_string_values_coerced(self):
        """String-typed values in JSON should be coerced to float."""
        report = {"replica_psnr": "36.0"}
        summary = evaluate_gates(report)
        psnr = next(r for r in summary.results if r.name == "replica_psnr")
        assert psnr.passed is True
        assert psnr.value == 36.0
