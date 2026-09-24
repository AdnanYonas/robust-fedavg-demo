"""
data.py
=======
Loads a small, real, zero-download dataset (digits: 8x8 handwritten
digit images, 10 classes, 1797 samples -- ships inside scikit-learn) and
splits it IID across N simulated clients.

We deliberately use an IID split here (unlike the non-IID Dirichlet
splits you'll see in other federated-learning demos) so the only
variable in this experiment is the poisoning attack and its defense --
mixing in non-IID data would make it harder to tell whether an accuracy
drop is caused by the attack or by ordinary statistical heterogeneity.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_federated_digits(n_clients=10, test_size=0.2, seed=0):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_train))
    client_splits = np.array_split(order, n_clients)

    clients = []
    for idx in client_splits:
        clients.append(dict(X=X_train[idx], y=y_train[idx]))

    return clients, (X_test, y_test)
