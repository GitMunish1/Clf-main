"""Canonical CLF V1 feature construction shared by training and live inference.

This module deliberately has no TensorFlow dependency.  The JSON contract is the
source of truth; BSSID names and signal strength never affect feature ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


FEATURE_VERSION = "clf-features-v2"
WIFI_INPUT_NAME = "wifi_input"
MOTION_INPUT_NAME = "motion_input"
WIFI_DIM = 13
MOTION_DIM = 19
MOTION_NAMES = (
    "accel_x", "accel_y", "accel_z", "accel_magnitude",
    "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude",
    "linear_accel_x", "linear_accel_y", "linear_accel_z",
    "linear_accel_magnitude", "magnet_x", "magnet_y", "magnet_z",
    "magnet_magnitude", "step_delta", "heading_sin", "heading_cos",
)


def default_contract_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "contracts" / "feature_contract.json"


def load_feature_contract(path: str | Path | None = None) -> dict:
    contract_path = Path(path) if path else default_contract_path()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_feature_contract(contract)
    return contract


def validate_feature_contract(contract: Mapping) -> None:
    if contract.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"feature version mismatch: expected {FEATURE_VERSION}, "
            f"got {contract.get('feature_version')!r}"
        )
    wifi = contract.get("wifi", {})
    ordered = wifi.get("access_points", [])
    if wifi.get("dimensions") != WIFI_DIM or len(ordered) != WIFI_DIM:
        raise ValueError("clf-features-v2 requires exactly 13 Wi-Fi features")
    indexes = [item.get("index") for item in ordered]
    bssids = [normalize_bssid(item.get("bssid", "")) for item in ordered]
    if indexes != list(range(WIFI_DIM)) or "" in bssids or len(set(bssids)) != WIFI_DIM:
        raise ValueError("Wi-Fi registry indexes/BSSIDs are invalid or non-unique")
    motion = contract.get("motion", {})
    if motion.get("dimensions") != MOTION_DIM or tuple(motion.get("ordering", [])) != MOTION_NAMES:
        raise ValueError("clf-features-v2 motion order must match the canonical 19-value order")
    if contract.get("ble", {}).get("enabled") is not False:
        raise ValueError("BLE must be disabled for CLF V1")
    if contract.get("model_inputs") != [WIFI_INPUT_NAME, MOTION_INPUT_NAME]:
        raise ValueError("CLF V1 model inputs must be wifi_input and motion_input")


def normalize_bssid(value: str) -> str:
    raw = re.sub(r"[^0-9a-fA-F]", "", str(value)).lower()
    if len(raw) != 12:
        return ""
    return ":".join(raw[i:i + 2] for i in range(0, 12, 2))


def normalize_rssi(value: float | np.ndarray) -> float | np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    result = (np.clip(values, -110.0, -20.0) + 110.0) / 90.0
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float32)


def ordered_bssids(contract: Mapping) -> tuple[str, ...]:
    return tuple(normalize_bssid(item["bssid"]) for item in contract["wifi"]["access_points"])


def build_wifi_vector(observations: Iterable[Mapping], contract: Mapping) -> np.ndarray:
    strongest: dict[str, float] = {}
    approved = set(ordered_bssids(contract))
    for observation in observations:
        key = normalize_bssid(observation.get("bssid", ""))
        if key not in approved:
            continue
        value = float(observation.get("rssi_dbm", -110.0))
        if not math.isfinite(value):
            raise ValueError("RSSI values must be finite")
        strongest[key] = max(strongest.get(key, -math.inf), value)
    raw = np.asarray([strongest.get(key, -110.0) for key in ordered_bssids(contract)], dtype=np.float32)
    return np.asarray(normalize_rssi(raw), dtype=np.float32)


def _xyz(sample: Mapping, name: str) -> tuple[float, float, float]:
    value = sample.get(name) or {}
    if isinstance(value, Mapping):
        result = (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))
    else:
        result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


def build_motion_vector(sample: Mapping) -> np.ndarray:
    accel = _xyz(sample, "accelerometer")
    gyro = _xyz(sample, "gyroscope")
    linear = _xyz(sample, "linear_acceleration")
    magnet = _xyz(sample, "magnetometer")
    heading = float(sample.get("heading_deg", 0.0))
    step_delta = float(sample.get("step_delta", 0.0))
    if not math.isfinite(heading) or not math.isfinite(step_delta):
        raise ValueError("heading and step_delta must be finite")

    def with_magnitude(values: Sequence[float]) -> tuple[float, float, float, float]:
        return (*values, math.sqrt(sum(item * item for item in values)))

    heading_rad = math.radians(heading % 360.0)
    result = np.asarray(
        [*with_magnitude(accel), *with_magnitude(gyro), *with_magnitude(linear),
         *with_magnitude(magnet), step_delta, math.sin(heading_rad), math.cos(heading_rad)],
        dtype=np.float32,
    )
    if result.shape != (MOTION_DIM,) or not np.isfinite(result).all():
        raise ValueError("motion preprocessing produced an invalid vector")
    return result


@dataclass
class MotionScaler:
    """Z-score scaler fitted only on training-session samples.

    Heading sine/cosine are already bounded and circular, so indexes 17 and 18
    pass through unchanged. Zero-variance training columns use a scale of 1.
    """

    mean: np.ndarray
    scale: np.ndarray
    fitted: bool = True

    @classmethod
    def fit(cls, training_motion: np.ndarray) -> "MotionScaler":
        values = np.asarray(training_motion, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != MOTION_DIM or len(values) == 0:
            raise ValueError("motion scaler needs non-empty [samples, 19] training data")
        if not np.isfinite(values).all():
            raise ValueError("motion training data contains NaN/Inf")
        mean = np.zeros(MOTION_DIM, dtype=np.float32)
        scale = np.ones(MOTION_DIM, dtype=np.float32)
        mean[:17] = np.mean(values[:, :17], axis=0)
        std = np.std(values[:, :17], axis=0)
        scale[:17] = np.where(std > 1e-8, std, 1.0)
        return cls(mean=mean, scale=scale, fitted=True)

    @classmethod
    def from_artifact(cls, artifact: Mapping, expected_feature_version: str = FEATURE_VERSION) -> "MotionScaler":
        if artifact.get("feature_version") != expected_feature_version:
            raise ValueError("preprocessing feature version mismatch")
        motion = artifact.get("motion", {})
        if motion.get("method") != "zscore" or motion.get("fit_on") != "training_split_only":
            raise ValueError("unsupported or unsafe motion preprocessing")
        if not motion.get("fitted", False):
            raise ValueError("motion preprocessing parameters are not fitted")
        mean = np.asarray(motion.get("mean", []), dtype=np.float32)
        scale = np.asarray(motion.get("scale", []), dtype=np.float32)
        if mean.shape != (MOTION_DIM,) or scale.shape != (MOTION_DIM,) or np.any(scale <= 0):
            raise ValueError("motion preprocessing parameters must each contain 19 safe values")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("motion preprocessing parameters contain NaN/Inf")
        return cls(mean=mean, scale=scale, fitted=True)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != MOTION_DIM:
            raise ValueError("motion input must contain 19 features")
        transformed = (array - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("scaled motion input contains NaN/Inf")
        return transformed.astype(np.float32)

    def to_artifact(self) -> dict:
        return {
            "schema_version": "1.0",
            "feature_version": FEATURE_VERSION,
            "motion": {
                "method": "zscore",
                "fit_on": "training_split_only",
                "fitted": bool(self.fitted),
                "mean": self.mean.astype(float).tolist(),
                "scale": self.scale.astype(float).tolist(),
                "passthrough_indices": [17, 18],
            },
        }


def preprocess_observation(observation: Mapping, contract: Mapping, scaler: MotionScaler) -> dict[str, np.ndarray]:
    return {
        WIFI_INPUT_NAME: build_wifi_vector(observation.get("wifi", []), contract),
        MOTION_INPUT_NAME: scaler.transform(build_motion_vector(observation)),
    }
