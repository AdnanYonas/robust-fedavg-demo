"""
model.py
========
Small MLP classifier wrapper built on scikit-learn's MLPClassifier.

We use MLPClassifier (with partial_fit + warm_start) instead of hand-rolled
NumPy so the whole demo stays under ~300 lines total and is easy to read
top-to-bottom in one sitting -- the point of this repo is the FEDERATED
AGGREGATION logic and the ATTACK/DEFENSE story, not a custom autodiff
engine. sklearn exposes .coefs_ / .intercepts_ directly, which is exactly
what a federated-averaging implementation needs to read and overwrite.
"""
from sklearn.neural_network import MLPClassifier
import numpy as np

HIDDEN_LAYER_SIZES = (64,)
N_CLASSES = 10  # sklearn.datasets.load_digits() has 10 classes (0-9)


def new_model(random_state=0):
    """A fresh, unfit model. classes_ is set manually via partial_fit's
    `classes` argument the first time each model is used, so every client
    and the server share an identical output layer shape from round 0."""
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        solver="sgd",
        learning_rate_init=0.05,
        max_iter=1,          # we drive training ourselves, one partial_fit per local step
        warm_start=True,
        random_state=random_state,
    )


def get_params(model):
    """Extract weights as a flat list of NumPy arrays (coefs then intercepts),
    e.g. [W1, W2, b1, b2] for a single hidden layer network."""
    return [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]


def set_params(model, params):
    n_layers = len(model.coefs_)
    model.coefs_ = [p.copy() for p in params[:n_layers]]
    model.intercepts_ = [p.copy() for p in params[n_layers:]]
    return model


def clone_with_params(template_model, params, random_state=0):
    """Build a new, structurally-identical model initialized to `params`
    (used to give every client a private local copy of the current global
    model each round, without clients accidentally sharing sklearn's
    internal optimizer state)."""
    m = new_model(random_state=random_state)
    # Prime internal structures (n_layers_, out_activation_, etc.) with a
    # single dummy partial_fit call, then immediately overwrite weights.
    n_features = template_model.coefs_[0].shape[0]
    dummy_X = np.zeros((N_CLASSES, n_features))
    dummy_y = np.arange(N_CLASSES)  # must include every class up front:
    # sklearn's warm_start partial_fit rejects a `classes=` argument that
    # doesn't match the unique labels actually present in `y`, even on
    # the very first call, so the priming batch must cover all 10 digits.
    m.partial_fit(dummy_X, dummy_y, classes=np.arange(N_CLASSES))
    set_params(m, params)
    return m
