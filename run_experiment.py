"""
run_experiment.py
==================
Top-level entry point. Runs the full comparison and saves a figure +
metrics.json. This is the single script a reader would run to reproduce
every number and plot in the README.

Usage:
    python3 run_experiment.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
import server

N_CLIENTS = 10
N_MALICIOUS = 2          # 20% of clients are Byzantine
N_ROUNDS = 30
SEEDS = [0, 1, 2, 3, 4]
MALICIOUS_IDS = tuple(range(N_MALICIOUS))  # clients 0..N_MALICIOUS-1 are attackers

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SCENARIOS = [
    dict(key="fedavg_no_attack", aggregation="fedavg", malicious_ids=(), attack=None,
         label="FedAvg, no attack (reference)"),
    dict(key="fedavg_sign_flip", aggregation="fedavg", malicious_ids=MALICIOUS_IDS, attack="sign_flip",
         label="FedAvg, under attack"),
    dict(key="coordinate_median_sign_flip", aggregation="coordinate_median", malicious_ids=MALICIOUS_IDS, attack="sign_flip",
         label="Coordinate-median, under attack"),
    dict(key="trimmed_mean_sign_flip", aggregation="trimmed_mean", malicious_ids=MALICIOUS_IDS, attack="sign_flip",
         label="Trimmed-mean, under attack"),
]

LABEL_FLIP_SCENARIOS = [
    dict(key="fedavg_label_flip", aggregation="fedavg", malicious_ids=MALICIOUS_IDS, attack="label_flip",
         label="FedAvg, label-flip attack"),
    dict(key="coordinate_median_label_flip", aggregation="coordinate_median", malicious_ids=MALICIOUS_IDS, attack="label_flip",
         label="Coordinate-median, label-flip attack"),
]


def run_all(scenarios):
    results = {}
    for sc in scenarios:
        print(f"\n=== {sc['label']} ===")
        seed_histories = []
        for seed in SEEDS:
            clients, test = data.load_federated_digits(n_clients=N_CLIENTS, seed=seed)
            t0 = time.time()
            history, _ = server.run_federated_training(
                clients, test, n_rounds=N_ROUNDS, aggregation=sc["aggregation"],
                malicious_ids=sc["malicious_ids"], attack=sc["attack"] or "sign_flip",
                seed=seed, verbose=False,
            )
            dt = time.time() - t0
            print(f"  seed={seed}  final_acc={history[-1]['accuracy']:.3f}  ({dt:.1f}s)")
            seed_histories.append(history)
        results[sc["key"]] = dict(scenario=sc, seed_histories=seed_histories)
    return results


def summarize(results):
    summary = {}
    for key, r in results.items():
        finals = [h[-1]["accuracy"] for h in r["seed_histories"]]
        summary[key] = dict(
            label=r["scenario"]["label"],
            final_acc_mean=float(np.mean(finals)),
            final_acc_std=float(np.std(finals)),
        )
    return summary


def plot_main_comparison(results):
    colors = {
        "fedavg_no_attack": "#666666",
        "fedavg_sign_flip": "#E41A1C",
        "coordinate_median_sign_flip": "#2166AC",
        "trimmed_mean_sign_flip": "#4DAF4A",
    }
    plt.figure(figsize=(7.5, 5.5))
    for key, r in results.items():
        acc_matrix = np.array([[row["accuracy"] for row in h] for h in r["seed_histories"]])
        mean_acc = acc_matrix.mean(axis=0)
        std_acc = acc_matrix.std(axis=0)
        rounds = np.arange(len(mean_acc))
        plt.plot(rounds, mean_acc, label=r["scenario"]["label"], color=colors[key], linewidth=2.2)
        plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color=colors[key], alpha=0.15)
    plt.xlabel("Federated round")
    plt.ylabel(f"Held-out test accuracy ({N_MALICIOUS}/{N_CLIENTS} clients malicious)")
    plt.title("FedAvg vs. robust aggregation under a model-poisoning attack")
    plt.ylim(-0.02, 1.0)
    plt.legend(fontsize=9, loc="center right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_vs_round.png"), dpi=160)
    plt.close()


def plot_label_flip_comparison(results):
    plt.figure(figsize=(6.5, 5))
    colors = {"fedavg_label_flip": "#E41A1C", "coordinate_median_label_flip": "#2166AC"}
    for key, r in results.items():
        acc_matrix = np.array([[row["accuracy"] for row in h] for h in r["seed_histories"]])
        mean_acc = acc_matrix.mean(axis=0)
        rounds = np.arange(len(mean_acc))
        plt.plot(rounds, mean_acc, label=r["scenario"]["label"], color=colors[key], linewidth=2.2)
    plt.xlabel("Federated round")
    plt.ylabel("Held-out test accuracy")
    plt.title("Subtler attack: label-flipping (data poisoning)")
    plt.ylim(-0.02, 1.0)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "label_flip_comparison.png"), dpi=160)
    plt.close()


def main():
    print(f"Setup: {N_CLIENTS} clients, {N_MALICIOUS} malicious (ids {MALICIOUS_IDS}), "
          f"{N_ROUNDS} rounds, {len(SEEDS)} seeds.")

    main_results = run_all(SCENARIOS)
    summary = summarize(main_results)
    print("\n=== SUMMARY (final-round accuracy, mean +/- std over 5 seeds) ===")
    for key, s in summary.items():
        print(f"  {s['label']:38s} {s['final_acc_mean']:.3f} +/- {s['final_acc_std']:.3f}")
    plot_main_comparison(main_results)

    print("\nRunning secondary label-flip (data-poisoning) comparison...")
    lf_results = run_all(LABEL_FLIP_SCENARIOS)
    lf_summary = summarize(lf_results)
    for key, s in lf_summary.items():
        print(f"  {s['label']:38s} {s['final_acc_mean']:.3f} +/- {s['final_acc_std']:.3f}")
    plot_label_flip_comparison(lf_results)

    out = dict(
        setup=dict(n_clients=N_CLIENTS, n_malicious=N_MALICIOUS, malicious_ids=list(MALICIOUS_IDS),
                   n_rounds=N_ROUNDS, seeds=SEEDS),
        main_summary=summary,
        label_flip_summary=lf_summary,
        main_per_seed_final={k: [h[-1] for h in r["seed_histories"]] for k, r in main_results.items()},
    )
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. Results -> {RESULTS_DIR}/metrics.json ; figures -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
