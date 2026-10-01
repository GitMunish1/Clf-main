"""CLF V1 real-survey dataset connector.

Rows represent node-labelled observations/windows. Wi-Fi is always expanded to
the fixed 13-BSSID registry, motion is always the canonical 19-value vector, and
survey sessions—not adjacent rows—define train/validation/test boundaries.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clf_features import (
    MOTION_NAMES,
    build_motion_vector,
    load_feature_contract,
    normalize_bssid,
    ordered_bssids,
)
from data.data_connector import DatasetConnector
from utils.definitions import get_project_root


class CLFDataConnector(DatasetConnector):
    def __init__(
        self,
        csv_path="datasets/clf/fingerprints.csv",
        feature_contract_path="artifacts/contracts/feature_contract.json",
        coordinate_calibration_path="artifacts/models/clf_v1/coordinate_calibration.json",
        missing_rssi=-110.0,
        require_calibrated_coordinates=True,
        split_seed=1234,
        **_legacy_options,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feature_contract_path = feature_contract_path
        self.coordinate_calibration_path = coordinate_calibration_path
        self.missing_rssi = float(missing_rssi)
        self.require_calibrated_coordinates = bool(require_calibrated_coordinates)
        self.split_seed = int(split_seed)
        self.feature_names = []
        self.feature_groups = {}
        self.metadata = None
        self.contract = None

    @staticmethod
    def _project_path(value):
        path = Path(value)
        return path if path.is_absolute() else Path(get_project_root()) / path

    def _validate_calibration(self):
        if not self.require_calibrated_coordinates:
            return
        path = self._project_path(self.coordinate_calibration_path)
        try:
            calibration = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                "coordinate calibration is required before CLF training; "
                f"missing {path}"
            ) from exc
        if calibration.get("calibrated") is not True or calibration.get("units") != "metres":
            raise ValueError(
                "coordinate calibration is not ready: provide measured local-metre "
                "calibration instead of GeoJSON/drawing units"
            )

    @staticmethod
    def _first_present(row, *names, default=0.0):
        for name in names:
            if name in row and pd.notna(row[name]):
                return float(row[name])
        return float(default)

    def _motion_for_row(self, row):
        def vector(prefix):
            return {
                axis: self._first_present(row, f"{prefix}_{axis}", f"imu_{prefix}_{axis}")
                for axis in ("x", "y", "z")
            }
        return build_motion_vector({
            "accelerometer": vector("accel"),
            "gyroscope": vector("gyro"),
            "linear_acceleration": vector("linear_accel"),
            "magnetometer": vector("magnet"),
            "step_delta": self._first_present(row, "step_delta"),
            "heading_deg": self._first_present(row, "heading_deg"),
        })

    def _wifi_matrix(self, data):
        approved = ordered_bssids(self.contract)
        candidates = {bssid: [] for bssid in approved}
        for column in data.columns:
            if not column.lower().startswith("wifi_"):
                continue
            key = normalize_bssid(column[5:])
            if key in candidates:
                candidates[key].append(column)
        matrix = np.full((len(data), len(approved)), self.missing_rssi, dtype=np.float32)
        for index, bssid in enumerate(approved):
            columns = candidates[bssid]
            if not columns:
                continue
            values = data[columns].apply(pd.to_numeric, errors="coerce")
            matrix[:, index] = values.max(axis=1, skipna=True).fillna(self.missing_rssi)
        return matrix

    def _build_session_split(self, data):
        if "split" in data.columns:
            split = data["split"].astype(str).str.lower()
            unknown = set(split) - {"train", "val", "test"}
            if unknown:
                raise ValueError(f"unsupported split labels: {sorted(unknown)}")
            if "session_id" in data.columns:
                leakage = data.assign(_split=split).groupby("session_id")["_split"].nunique()
                if (leakage > 1).any():
                    raise ValueError("a survey session cannot span multiple data partitions")
        else:
            if "session_id" not in data.columns:
                raise ValueError(
                    "real CLF training data needs explicit split=train/val/test or session_id"
                )
            sessions = np.asarray(sorted(data["session_id"].astype(str).unique()))
            if len(sessions) < 3:
                raise ValueError("at least three survey sessions are required for 70/15/15 splitting")
            rng = np.random.default_rng(self.split_seed)
            rng.shuffle(sessions)
            val_count = max(1, int(round(len(sessions) * 0.15)))
            test_count = max(1, int(round(len(sessions) * 0.15)))
            if val_count + test_count >= len(sessions):
                val_count = test_count = 1
            test_sessions = set(sessions[:test_count])
            val_sessions = set(sessions[test_count:test_count + val_count])
            split = data["session_id"].astype(str).map(
                lambda session: "test" if session in test_sessions else (
                    "val" if session in val_sessions else "train"
                )
            )
        indices = {
            name: np.where(split.to_numpy() == name)[0]
            for name in ("train", "val", "test")
        }
        if any(len(indices[name]) == 0 for name in indices):
            raise ValueError("train, val, and test partitions must all be non-empty")
        self.split_indices = [indices]

    def load_dataset(self):
        self.contract = load_feature_contract(self._project_path(self.feature_contract_path))
        self._validate_calibration()
        path = self._project_path(self.csv_path)
        if path.name == "fingerprints.example.csv":
            raise ValueError("fingerprints.example.csv is documentation only, not production training data")
        data = pd.read_csv(path)
        required = {"node_id", "floor", "x_m", "y_m"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"CLF dataset missing required columns: {sorted(missing)}")

        coordinates = data[["x_m", "y_m"]].apply(pd.to_numeric, errors="coerce")
        if coordinates.isna().any().any() or not np.isfinite(coordinates.to_numpy()).all():
            raise ValueError("CLF metre coordinates contain NaN/Inf")
        if (coordinates.to_numpy() < 0).any():
            raise ValueError("CLF local-metre coordinates must use a non-negative calibrated origin")

        wifi = self._wifi_matrix(data)
        motion = np.vstack([self._motion_for_row(row) for _, row in data.iterrows()]).astype(np.float32)
        if not np.isfinite(wifi).all() or not np.isfinite(motion).all():
            raise ValueError("CLF sensor features contain NaN/Inf")

        wifi_names = [f"wifi_{bssid}" for bssid in ordered_bssids(self.contract)]
        self.feature_names = wifi_names + list(MOTION_NAMES)
        self.feature_groups = {
            "wifi": list(range(13)),
            "motion": list(range(13, 32)),
            "ble": [],
        }
        self.rss = np.concatenate([wifi, motion], axis=1).astype(np.float32)
        self.pos = coordinates.to_numpy(dtype=np.float32)
        self.floor = pd.to_numeric(data["floor"], errors="raise").to_numpy()
        self.floors = np.sort(np.unique(self.floor))
        self.num_floors = len(self.floors)
        self.floorplan_width = []
        self.floorplan_height = []
        for floor in self.floors:
            positions = self.pos[self.floor == floor]
            self.floorplan_width.append(float(np.max(positions[:, 0]) + 1e-6))
            self.floorplan_height.append(float(np.max(positions[:, 1]) + 1e-6))
        self.metadata = data[[column for column in (
            "node_id", "floor", "x_m", "y_m", "session_id", "timestamp", "window_id"
        ) if column in data.columns]].copy()
        if "timestamp" in data.columns:
            self.time = data["timestamp"].to_numpy()
        self._build_session_split(data)
        return self

    def get_dataset_identifier(self):
        return "clf"
