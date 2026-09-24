"""
server.py
=========
Orchestrates the federated training loop: broadcast the current global
model, collect every client's (possibly malicious) reported update,
aggregate with the chosen strategy, evaluate, repeat.
"""
import numpy as np
from sklearn.metrics import accuracy_score

from model import new_model, get_params, set_params, clone_with_params
import client as client_mod
from aggregation import STRATEGIES


def run_federated_training(
    clients, test_data, n_rounds=30, aggregation="fedavg",
    malicious_ids=(), attack="sign_flip", seed=0, verbose=True,
):
    X_test, y_test = test_data
    n_clients = len(clients)

    global_model = new_model(random_state=seed)
    # prime it so .coefs_/.intercepts_ exist before round 0 -- must include
    # every class in this first batch (see model.py's clone_with_params).
    n_features = X_test.shape[1]
    global_model.partial_fit(
        np.zeros((10, n_features)), np.arange(10), classes=np.arange(10)
    )
    client_mod.set_template(global_model)

    global_params = get_params(global_model)
    agg_fn = STRATEGIES[aggregation]

    history = []
    for r in range(n_rounds):
        reported_params = []
        for cid, c in enumerate(clients):
            is_malicious = cid in malicious_ids
            params = client_mod.client_round(
                global_params, c, seed=seed * 1000 + r * 10 + cid,
                is_malicious=is_malicious, attack=attack,
            )
            reported_params.append(params)

        client_sizes = [len(c["X"]) for c in clients]
        global_params = agg_fn(reported_params, client_sizes)

        eval_model = clone_with_params(global_model, global_params, random_state=seed)
        preds = eval_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        history.append(dict(round=r, accuracy=float(acc)))

        if verbose:
            tag = f" ({len(malicious_ids)}/{n_clients} malicious, attack={attack})" if malicious_ids else " (no attack)"
            print(f"  [{aggregation:18s}] round {r:2d}  test_acc={acc:.3f}{tag if r == 0 else ''}")

    return history, global_params
