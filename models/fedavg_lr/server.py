"""Server-side evaluation for the FedAvg-LR pipeline.

Rebuilds a :class:`LogisticRegression` from the aggregated parameters and
computes AUPRC / F1 / Precision / Recall on the held-out validation set
each round, plus the test set on the final round. Tracks early-stop
state so the orchestrator can summarise after the simulation ends.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import best_f1_threshold, metrics_at_threshold


N_FEATURES: int = 13


def _build_eval_lr(cfg: dict) -> LogisticRegression:
    params = dict(cfg.get("lr_params", {}))
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 1000)),
        warm_start=False,
        random_state=int(cfg.get("random_seed", 42)),
    )


def _set_params(
    model: LogisticRegression, parameters: List[np.ndarray], n_features: int
) -> None:
    coef, intercept = parameters
    model.coef_ = coef.astype(np.float64, copy=False)
    model.intercept_ = intercept.astype(np.float64, copy=False)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = n_features


def _scores(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def make_server_eval_fn(
    cfg: dict,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    wandb_run=None,
) -> Tuple[Callable, Dict[str, Any]]:
    """Return ``(eval_fn, state)``.

    ``eval_fn`` matches Flower's ``evaluate_fn`` contract:
    ``(server_round, parameters, eval_config) -> (val_loss, metrics)``.

    ``state`` is a shared dict the caller can inspect after the simulation
    to report best round, best val_auprc, and the final test metrics.
    """
    patience: int = int(cfg.get("early_stop_patience", 10))
    num_rounds: int = int(cfg["num_rounds"])

    state: Dict[str, Any] = {
        "best_val_auprc": -1.0,
        "best_round": -1,
        "patience_counter": 0,
        "early_stop_triggered": False,
        "history": [],
        "final_test": None,
    }

    def eval_fn(server_round: int, parameters: List[np.ndarray], eval_config: Dict[str, Any]):
        model = _build_eval_lr(cfg)
        _set_params(model, parameters, N_FEATURES)

        val_scores = _scores(model, x_val)
        # Tune the decision threshold on validation (max-F1); AUPRC is
        # threshold-free. See evaluation.metrics for the rationale.
        threshold = best_f1_threshold(y_val, val_scores)
        v = metrics_at_threshold(y_val, val_scores, threshold)
        val_loss = 1.0 - v["auprc"]

        print(
            f"[server] round {server_round} | thr={threshold:.4f} | "
            f"val_auprc={v['auprc']:.4f} | val_f1={v['f1']:.4f} | "
            f"val_precision={v['precision']:.4f} | val_recall={v['recall']:.4f}"
        )

        metrics: Dict[str, float] = {
            "val_auprc": v["auprc"],
            "val_f1": v["f1"],
            "val_precision": v["precision"],
            "val_recall": v["recall"],
        }
        if wandb_run is not None:
            wandb_run.log({"round": server_round, "val_loss": val_loss, **metrics})

        if server_round > 0:
            if v["auprc"] > state["best_val_auprc"]:
                state["best_val_auprc"] = v["auprc"]
                state["best_round"] = server_round
                state["patience_counter"] = 0
            else:
                state["patience_counter"] += 1
            if (
                state["patience_counter"] >= patience
                and not state["early_stop_triggered"]
            ):
                print(
                    f"[server] early-stop signal at round {server_round}: "
                    f"val_auprc unchanged for {patience} rounds "
                    f"(best={state['best_val_auprc']:.4f} at round "
                    f"{state['best_round']})"
                )
                state["early_stop_triggered"] = True

        state["history"].append({"round": server_round, **metrics})

        if server_round == num_rounds:
            test_scores = _scores(model, x_test)
            # Reuse the threshold tuned on this round's validation scores.
            t = metrics_at_threshold(y_test, test_scores, threshold)
            test_loss = 1.0 - t["auprc"]
            print(
                f"[server] FINAL round {server_round} | "
                f"test_auprc={t['auprc']:.4f} | test_f1={t['f1']:.4f} | "
                f"test_precision={t['precision']:.4f} | "
                f"test_recall={t['recall']:.4f}"
            )
            state["final_test"] = {
                "test_auprc": t["auprc"],
                "test_f1": t["f1"],
                "test_precision": t["precision"],
                "test_recall": t["recall"],
            }
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "round": server_round,
                        "test_loss": test_loss,
                        **state["final_test"],
                    }
                )

        return val_loss, metrics

    return eval_fn, state
