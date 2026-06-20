"""Server-side evaluation for FFD.

Each round, rebuilds an :class:`FFDModel`, loads the aggregated parameters,
and scores AUPRC/F1/Precision/Recall on the held-out validation set. On the
final round, the test set is also scored.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
from evaluation.metrics import best_f1_threshold, metrics_at_threshold

from .model import FFDModel


def make_server_eval_fn(
    cfg: dict,
    input_dim: int,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    wandb_run=None,
) -> Tuple[Callable, Dict[str, Any]]:
    """Return ``(eval_fn, state)``.

    ``eval_fn`` matches Flower's ``evaluate_fn`` contract:
    ``(server_round, parameters, eval_config) -> (val_loss, metrics)``.

    ``state`` is a shared dict the caller inspects after the simulation to
    report best round, best val_auprc, and final test metrics.
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

    def eval_fn(
        server_round: int,
        parameters: List[np.ndarray],
        eval_config: Dict[str, Any],
    ):
        model = FFDModel(input_dim=input_dim)
        model.set_weights(parameters)

        val_scores = model.predict_proba(x_val)[:, 1]
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
            test_scores = model.predict_proba(x_test)[:, 1]
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
