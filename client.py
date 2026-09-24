"""
client.py
=========
One simulated federation participant. Every client runs the exact same
`local_update()` entry point each round; the only difference for a
malicious client is which function is plugged in as its "reporting"
step -- this mirrors how a real Byzantine client behaves: it still talks
to the server in the same protocol, it just lies about (or corrupts)
its update.

Two attack styles are implemented, selectable independently:

  "label_flip"   -- DATA poisoning. The client trains normally, but on
                    its own copy of the data with labels flipped
                    (y -> 9 - y for the digits dataset). This produces
                    an update that *looks* like an ordinary gradient
                    step (same magnitude, same shape) but points the
                    model toward wrong predictions. This is the harder
                    attack for robust aggregation to catch, because the
                    malicious update isn't a statistical outlier in
                    magnitude -- only in direction.

  "sign_flip"     -- MODEL/update poisoning ("boosted negation"). The
                    client trains normally, computes its honest update
                    (new params - old global params), then SENDS THE
                    OPPOSITE, scaled up by `boost`:
                        reported = global - boost * (honest - global)
                    This is a classic strong Byzantine attack: it tries
                    to drag the global model backwards, hard, every
                    round. It is very easy for a robust aggregator to
                    spot (its magnitude and direction are both
                    anomalous) which is exactly why it's the default
                    attack used to demonstrate FedAvg failing and
                    coordinate-wise median/trimmed-mean succeeding.
"""
import numpy as np
from model import clone_with_params, get_params, N_CLASSES

LOCAL_EPOCHS = 1  # local partial_fit passes per federated round


def _train_locally(global_params, X, y, seed):
    local_model = clone_with_params(_template[0], global_params, random_state=seed)
    for _ in range(LOCAL_EPOCHS):
        local_model.partial_fit(X, y, classes=np.arange(N_CLASSES))
    return get_params(local_model)


# a template model is stashed here the first time set_template() is called,
# purely so clone_with_params() knows the network's input dimensionality
# without every call needing to pass it explicitly.
_template = [None]


def set_template(model):
    _template[0] = model


def honest_update(global_params, client_data, seed):
    return _train_locally(global_params, client_data["X"], client_data["y"], seed)


def label_flip_update(global_params, client_data, seed):
    y_poisoned = (N_CLASSES - 1) - client_data["y"]  # 0<->9, 1<->8, ...
    return _train_locally(global_params, client_data["X"], y_poisoned, seed)


def sign_flip_update(global_params, client_data, seed, boost=8.0):
    honest = _train_locally(global_params, client_data["X"], client_data["y"], seed)
    reported = []
    for g, h in zip(global_params, honest):
        delta = h - g
        reported.append(g - boost * delta)
    return reported


ATTACKS = {
    "label_flip": label_flip_update,
    "sign_flip": sign_flip_update,
}


def client_round(global_params, client_data, seed, is_malicious, attack):
    if is_malicious:
        return ATTACKS[attack](global_params, client_data, seed)
    return honest_update(global_params, client_data, seed)
