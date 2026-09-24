"""
tests/test_aggregation.py
==========================
Minimal smoke tests for the aggregation strategies -- run these with
`pytest` (or `python3 -m pytest`) before you trust any experiment
results. No ML involved: these operate on tiny hand-picked arrays where
the correct answer is obvious, so a bug in the aggregation math itself
(as opposed to the learning dynamics) is easy to catch fast.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aggregation import fedavg, coordinate_median, trimmed_mean


def _params(vals):
    """Wrap a list of scalars into the [param_array] format the
    aggregation functions expect (one client per entry)."""
    return [[np.array([v])] for v in vals]


def test_fedavg_matches_weighted_mean():
    client_params = _params([1.0, 2.0, 3.0])
    weights = [1, 1, 1]
    out = fedavg(client_params, weights)
    assert np.isclose(out[0][0], 2.0)


def test_fedavg_is_not_robust_to_one_outlier():
    # A single extreme report should be able to drag a plain mean
    # arbitrarily far -- this is the vulnerability the whole repo is
    # about, so we assert it explicitly rather than just assuming it.
    client_params = _params([1.0, 1.0, 1.0, 1.0, 1000.0])
    out = fedavg(client_params, [1, 1, 1, 1, 1])
    assert out[0][0] > 100, "fedavg should be dominated by the outlier"


def test_coordinate_median_resists_one_outlier():
    client_params = _params([1.0, 1.0, 1.0, 1.0, 1000.0])
    out = coordinate_median(client_params)
    assert np.isclose(out[0][0], 1.0), "median should ignore a single extreme report"


def test_coordinate_median_resists_up_to_but_not_including_half():
    # 4 honest (value 1.0) vs 3 malicious (value 1000.0) out of 7 -> still
    # a minority, median must still return the honest value.
    client_params = _params([1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0])
    out = coordinate_median(client_params)
    assert np.isclose(out[0][0], 1.0)


def test_trimmed_mean_resists_outliers_within_trim_budget():
    # 8 honest + 2 malicious extreme values, trim_frac=0.2 drops the top
    # and bottom 20% (2 values each) before averaging -> the 2 malicious
    # extreme-high values get trimmed away entirely.
    client_params = _params([1.0] * 8 + [1000.0, 1000.0])
    out = trimmed_mean(client_params, trim_frac=0.2)
    assert np.isclose(out[0][0], 1.0), "trimmed mean should drop the malicious outliers"


def test_all_strategies_agree_with_no_attack():
    # With identical honest reports, every strategy should return exactly
    # that value -- a basic identity sanity check.
    client_params = _params([2.5, 2.5, 2.5, 2.5])
    for fn in (fedavg, coordinate_median, trimmed_mean):
        out = fn(client_params, [1, 1, 1, 1])
        assert np.isclose(out[0][0], 2.5), f"{fn.__name__} failed on identical inputs"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
