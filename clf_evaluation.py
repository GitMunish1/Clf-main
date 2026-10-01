"""Held-out real-session evaluation for CLF V1 and its motion ablation."""
from __future__ import annotations

import time
import numpy as np


def localization_metrics(
    true_floor,
    true_cell,
    true_xy_m,
    predicted_floor,
    predicted_cell,
    predicted_xy_m,
    accepted,
    latency_ms,
):
    accepted = np.asarray(accepted, dtype=bool)
    true_xy = np.asarray(true_xy_m, dtype=float)
    predicted_xy = np.asarray(predicted_xy_m, dtype=float)
    errors = np.linalg.norm(predicted_xy[accepted] - true_xy[accepted], axis=1)
    finite = errors[np.isfinite(errors)]
    return {
        "floor_accuracy": float(np.mean(np.asarray(predicted_floor) == np.asarray(true_floor))),
        "grid_cell_accuracy": float(np.mean(np.asarray(predicted_cell) == np.asarray(true_cell))),
        "mean_position_error_m": float(np.mean(finite)) if len(finite) else None,
        "median_position_error_m": float(np.median(finite)) if len(finite) else None,
        "p95_position_error_m": float(np.percentile(finite, 95)) if len(finite) else None,
        "confidence_acceptance_rate": float(np.mean(accepted)),
        "low_confidence_rejection_rate": float(np.mean(~accepted)),
        "mean_inference_latency_ms": float(np.mean(latency_ms)),
    }


def timed_predict(model, inputs):
    start = time.perf_counter()
    outputs = model.predict(inputs, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    batch = len(next(iter(inputs.values())))
    return outputs, np.full(batch, elapsed_ms / max(batch, 1), dtype=float)


def ablation_inputs(inputs):
    """Return comparable Wi-Fi-only and Wi-Fi+motion model input mappings."""
    return {
        "wifi_only": {
            "wifi_input": inputs["wifi_input"],
            "motion_input": np.zeros_like(inputs["motion_input"]),
        },
        "wifi_plus_motion": inputs,
    }
