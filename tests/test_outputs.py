import json
import math
import os
import subprocess
import sys

import pytest

TRUTH_DIR = "/tests/ref"
ARTIFACT_DIR = "/app/logs"
CHECK_STEPS = ["50", "100", "200", "400"]
SPAN_COUNTS = [1, 4, 8]
MAX_REL_ERR = 0.02


def _read_artifact(path):
    assert os.path.isfile(path), f"required artifact not found: {path}"
    assert not os.path.islink(path), f"artifact must not be a symlink: {path}"
    resolved = os.path.realpath(path)
    assert resolved.startswith("/app/"), f"artifact must resolve inside /app: {path}"
    with open(resolved) as fh:
        return json.load(fh)


def _compute_truth(span_count, tmp_path):
    dest = tmp_path / f"truth_span_{span_count}.json"
    subprocess.run(
        [
            sys.executable,
            os.path.join(TRUTH_DIR, "train_reference.py"),
            "--span_count",
            str(span_count),
            "--out",
            str(dest),
        ],
        cwd=TRUTH_DIR,
        check=True,
        timeout=300,
    )
    with open(dest) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def truth(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("truth")
    return {n: _compute_truth(n, workdir) for n in SPAN_COUNTS}


@pytest.fixture(scope="session")
def artifacts():
    return {n: _read_artifact(os.path.join(ARTIFACT_DIR, f"span_{n}.json")) for n in SPAN_COUNTS}


class TestLossCurve:
    @pytest.mark.parametrize("span_count", SPAN_COUNTS)
    def test_within_tolerance(self, span_count, artifacts, truth):
        """loss_curve at each checkpoint must track the reference trajectory within 2% relative error, for every span_count."""
        observed = artifacts[span_count]["loss_curve"]
        expected = truth[span_count]["loss_curve"]
        for step in CHECK_STEPS:
            assert step in observed, f"span_count={span_count}: checkpoint {step} missing from loss_curve"
            obs, exp = observed[step], expected[step]
            rel_err = abs(obs - exp) / max(abs(exp), 1e-8)
            assert rel_err < MAX_REL_ERR, (
                f"span_count={span_count} step={step}: {obs} vs reference {exp} "
                f"(relative error {rel_err:.3f} >= {MAX_REL_ERR})"
            )

    def test_single_span_is_unaffected(self, artifacts, truth):
        """span_count=1 must reproduce the reference baseline, confirming a correct fix leaves the already-correct trivial case untouched."""
        observed = artifacts[1]["loss_curve"]
        expected = truth[1]["loss_curve"]
        for step in CHECK_STEPS:
            obs, exp = observed[step], expected[step]
            rel_err = abs(obs - exp) / max(abs(exp), 1e-8)
            assert rel_err < MAX_REL_ERR, (
                f"span_count=1 step={step}: {obs} vs reference {exp} "
                f"(relative error {rel_err:.3f} >= {MAX_REL_ERR})"
            )


class TestCommitTrace:
    @pytest.mark.parametrize("span_count", SPAN_COUNTS)
    def test_tally_is_multiple_of_span_count(self, span_count, artifacts, truth):
        """Every commit_trace entry's running tally must be an exact multiple of span_count and step index must match the reference step exactly."""
        trace = artifacts[span_count]["commit_trace"]
        truth_trace = truth[span_count]["commit_trace"]
        assert len(trace) == 400, f"span_count={span_count}: expected 400 commit entries, found {len(trace)}"
        for i, (step, tally) in enumerate(trace):
            assert tally % span_count == 0, (
                f"span_count={span_count} step={step}: running tally {tally} is not a "
                f"multiple of span_count ({span_count}); the optimizer committed mid-span"
            )
            expected_step = truth_trace[i][0]
            assert step == expected_step, (
                f"span_count={span_count} tally={tally}: expected commit at step {expected_step}, "
                f"but found commit at step {step}"
            )

    @pytest.mark.parametrize("span_count", SPAN_COUNTS)
    def test_tally_progresses_monotonically(self, span_count, artifacts):
        """Successive commit_trace tallies must strictly increase by exactly span_count each time."""
        trace = artifacts[span_count]["commit_trace"]
        tallies = [t for _, t in trace]
        diffs = {tallies[i] - tallies[i - 1] for i in range(1, len(tallies))}
        assert diffs == {span_count}, (
            f"span_count={span_count}: expected every consecutive tally to differ by "
            f"exactly {span_count}, found differences {diffs}"
        )


class TestDiagnosticSchema:
    @pytest.mark.parametrize("span_count", SPAN_COUNTS)
    def test_grad_and_scale_curves_populated(self, span_count, artifacts, truth):
        """grad_curve and scale_curve must carry numeric values that track the reference trajectory within 2% relative error and be finite."""
        record = artifacts[span_count]
        expected_record = truth[span_count]
        for field in ("grad_curve", "scale_curve"):
            assert field in record, f"span_count={span_count}: '{field}' missing from output"
            for step in CHECK_STEPS:
                assert step in record[field], (
                    f"span_count={span_count}: checkpoint {step} missing from '{field}'"
                )
                obs = record[field][step]
                exp = expected_record[field][step]
                assert isinstance(obs, (int, float)), (
                    f"span_count={span_count}: '{field}[{step}]' must be numeric, "
                    f"got {obs!r}"
                )
                assert not math.isnan(obs) and not math.isinf(obs), (
                    f"span_count={span_count}: '{field}[{step}]' cannot be NaN or Inf"
                )
                rel_err = abs(obs - exp) / max(abs(exp), 1e-8)
                assert rel_err < MAX_REL_ERR, (
                    f"span_count={span_count} step={step} {field}: {obs} vs reference {exp} "
                    f"(relative error {rel_err:.3f} >= {MAX_REL_ERR})"
                )