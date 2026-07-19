"""Does training *with* meaning-preserving perturbations actually help a model
learn — or does a plain RSI loop just learn to reward-hack?

This is a small but real learning experiment (logistic regression trained with
gradient descent), built around the same channels the Perturbation Agent uses on
real tasks: the true bug signal plus three incidental channels
(problem-statement phrasing, README, test order).

Setup
-----
Ground truth: a fix passes iff it addresses the *bug signal*. The three
incidental channels carry no real information about correctness.

  * Plain RSI baseline: trained on one canonical configuration per task. There,
    the incidental channels happen to line up with the label, so an ordinary
    learner is free to lean on them — a shortcut / reward hack. We do NOT defend
    it. (Per the brief: the baseline is not made hack-proof.)

  * Skeptical / perturbation-augmented: trained on the same tasks re-presented
    under many perturbations, where the incidental channels are decorrelated from
    the label. The only way to keep reward is to rely on the bug signal.

Then both are tested on held-out tasks under fresh perturbations. We measure
held-out accuracy and — the reward-hacking signature — the per-task score
variance across perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Feature layout: [bug_signal, ps, readme, test_order]
D = 4
BUG = 0
SPUR = [1, 2, 3]
CHANNEL_NAMES = ["bug_signal", "ps_phrasing", "readme", "test_order"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_canonical_training(n: int, corr: float, rng: np.random.Generator):
    """One configuration per task. Incidental channels are correlated with the
    label here (only one config is ever seen), which is exactly what tempts a
    plain learner into a shortcut."""
    x_bug = rng.normal(0, 1, size=n)
    y = (x_bug > 0).astype(float)
    sign = 2 * y - 1  # +/-1
    X = np.zeros((n, D))
    X[:, BUG] = x_bug
    for k in SPUR:
        X[:, k] = corr * sign + np.sqrt(1 - corr**2) * rng.normal(0, 1, size=n)
    return X, y


def make_perturbed_training(n: int, m: int, corr: float, rng: np.random.Generator):
    """Same tasks, re-presented under ``m`` perturbations each. The bug signal is
    held fixed per task; incidental channels are resampled independently, so they
    no longer predict the label."""
    x_bug = rng.normal(0, 1, size=n)
    y = (x_bug > 0).astype(float)
    Xs, ys = [], []
    for _ in range(m):
        X = np.zeros((n, D))
        X[:, BUG] = x_bug
        for k in SPUR:
            X[:, k] = rng.normal(0, 1, size=n)  # decorrelated from y
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def train_logreg(X, y, steps=800, lr=0.3, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


@dataclass
class HeldOutResult:
    name: str
    accuracy: float
    mean_task_variance: float  # reward-hacking signature (score swing under perturbation)
    weight_on_bug: float  # fraction of |weight| mass on the true bug signal
    weight_on_spurious: float


def evaluate_heldout(w, b, name, n_tasks=400, k_perturb=12, seed=99) -> HeldOutResult:
    """Fresh tasks, each re-checked under k fresh perturbations."""
    rng = np.random.default_rng(seed)
    x_bug = rng.normal(0, 1, size=n_tasks)
    y = (x_bug > 0).astype(float)

    per_task_scores = np.zeros((n_tasks, k_perturb))
    for j in range(k_perturb):
        X = np.zeros((n_tasks, D))
        X[:, BUG] = x_bug
        for kk in SPUR:
            X[:, kk] = rng.normal(0, 1, size=n_tasks)  # perturbed each re-check
        pred = (_sigmoid(X @ w + b) > 0.5).astype(float)
        per_task_scores[:, j] = (pred == y).astype(float)

    accuracy = float(per_task_scores.mean())
    mean_task_variance = float(per_task_scores.std(axis=1).mean())

    absw = np.abs(w)
    total = absw.sum() + 1e-9
    return HeldOutResult(
        name=name,
        accuracy=accuracy,
        mean_task_variance=mean_task_variance,
        weight_on_bug=float(absw[BUG] / total),
        weight_on_spurious=float(absw[SPUR].sum() / total),
    )


def run_experiment(n_tasks=600, m_perturb=10, corr=0.9, seed=0):
    rng = np.random.default_rng(seed)

    Xc, yc = make_canonical_training(n_tasks, corr, rng)
    Xp, yp = make_perturbed_training(n_tasks, m_perturb, corr, rng)

    w_base, b_base = train_logreg(Xc, yc, seed=seed)
    w_skep, b_skep = train_logreg(Xp, yp, seed=seed)

    base = evaluate_heldout(w_base, b_base, "plain RSI (canonical only)")
    skep = evaluate_heldout(w_skep, b_skep, "skeptical (perturbation-augmented)")
    return base, skep, (w_base, w_skep)


if __name__ == "__main__":
    base, skep, (wb, ws) = run_experiment()
    for r in (base, skep):
        print(r)
