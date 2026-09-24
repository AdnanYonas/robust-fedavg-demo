"""
aggregation.py
==============
Server-side aggregation rules. Every function has the same signature:
    aggregate(list_of_client_param_lists, client_weights) -> new_param_list
so the training loop in server.py can swap strategies with one line.

fedavg              -- the standard, NON-robust baseline: a
                        sample-size-weighted mean, coordinate by
                        coordinate. A single malicious client can shift
                        this arbitrarily far (its report has unbounded
                        influence) which is exactly the vulnerability
                        this repo demonstrates.

coordinate_median    -- replace the mean with the coordinate-wise
                        MEDIAN across clients. The median of n values is
                        insensitive to any single value being pushed to
                        +/-infinity as long as fewer than half the
                        clients are malicious -- a textbook Byzantine-
                        robust aggregator (Yin et al., 2018,
                        "Byzantine-Robust Distributed Learning").

trimmed_mean         -- sort each coordinate across clients, drop the
                        top and bottom `trim_frac` fraction of values,
                        average what's left. Degrades to the ordinary
                        mean when trim_frac=0, and to something close to
                        the median as trim_frac -> 0.5.
"""
import numpy as np


def fedavg(client_params_list, client_weights):
    weights = np.array(client_weights, dtype=float)
    weights = weights / weights.sum()
    n_arrays = len(client_params_list[0])
    aggregated = []
    for a in range(n_arrays):
        stacked = np.stack([cp[a] for cp in client_params_list], axis=0)
        w = weights.reshape((-1,) + (1,) * (stacked.ndim - 1))
        aggregated.append((stacked * w).sum(axis=0))
    return aggregated


def coordinate_median(client_params_list, client_weights=None):
    n_arrays = len(client_params_list[0])
    aggregated = []
    for a in range(n_arrays):
        stacked = np.stack([cp[a] for cp in client_params_list], axis=0)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated


def trimmed_mean(client_params_list, client_weights=None, trim_frac=0.2):
    n_clients = len(client_params_list)
    k = int(np.floor(trim_frac * n_clients))
    n_arrays = len(client_params_list[0])
    aggregated = []
    for a in range(n_arrays):
        stacked = np.stack([cp[a] for cp in client_params_list], axis=0)  # (n_clients, ...)
        sorted_vals = np.sort(stacked, axis=0)
        if k > 0:
            trimmed = sorted_vals[k: n_clients - k]
        else:
            trimmed = sorted_vals
        aggregated.append(trimmed.mean(axis=0))
    return aggregated


STRATEGIES = {
    "fedavg": fedavg,
    "coordinate_median": coordinate_median,
    "trimmed_mean": trimmed_mean,
}
