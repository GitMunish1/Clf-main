import numpy as np

from data.mcel_data_provider import (
    MCELdataProvider,
    compute_grid_cell_origins_of_encoding,
)


class CLFSensorFusionDataProvider(MCELdataProvider):
    """CLF multi-input provider with numerically safe Multi-CEL decoding."""

    def _clf_param(self, name, default):
        return self.pr.param_dict.get(name, default)

    def build_sensor_inputs(self):
        raw = self.rss.astype(np.float32, copy=True)
        groups = self.dc.feature_groups
        feature_names = self.dc.feature_names
        n = len(raw)

        wifi = self._slice_or_zero(raw, groups.get('wifi', []), n)
        ble = self._slice_or_zero(raw, groups.get('ble', []), n)
        motion_idx = groups.get('motion', [])
        motion = self._slice_or_zero(raw, motion_idx, n)

        wifi = self._normalize_rssi(wifi)
        ble = self._normalize_rssi(ble)
        motion = self._normalize_motion(
            motion, [feature_names[i] for i in motion_idx]
        )

        self.x = {
            'wifi_input': wifi.astype(np.float32),
            'ble_input': ble.astype(np.float32),
            'motion_input': motion.astype(np.float32),
        }
        return self

    @staticmethod
    def _slice_or_zero(raw, indices, n):
        if not indices:
            return np.zeros((n, 1), dtype=np.float32)
        return raw[:, indices]

    @staticmethod
    def _normalize_rssi(values):
        result = (values + 110.0) / 110.0
        return np.clip(result, 0.0, 1.0)

    def _normalize_motion(self, values, names):
        result = values.astype(np.float32, copy=True)
        if not names:
            return result

        train_idx = self.split_indices[self.split_idx]['train']
        for col, name in enumerate(names):
            if name in ('heading_sin', 'heading_cos'):
                continue
            train_values = (
                result[train_idx, col] if len(train_idx) else result[:, col]
            )
            scale = max(
                float(np.max(np.abs(train_values)))
                if len(train_values) else 0.0,
                1.0,
            )
            result[:, col] /= scale
        return result

    def get_x(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        return {name: values[subset] for name, values in self.x.items()}

    def get_input_dim(self):
        return {
            name: int(values.shape[1])
            for name, values in self.x.items()
        }

    def _inverse_distance_weights(self, distances):
        """Return finite normalized inverse-distance weights."""
        epsilon = float(self._clf_param('label_weight_epsilon', 1e-6))
        safe = np.maximum(
            np.asarray(distances, dtype=np.float64), epsilon
        )
        inverse = 1.0 / safe
        total = float(np.sum(inverse))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(
                len(safe), 1.0 / max(len(safe), 1), dtype=np.float64
            )
        return inverse / total

    def compute_multilabel_aug_data(self, weighted_grid_labels=True):
        """Build Multi-CEL labels without 1/0 or NaN target weights."""
        labels = self.y
        num_grids = int(self.get_num_grid_cells())
        multi_labels = np.zeros((len(labels), num_grids, 3))
        multi_grid_cell_labels = np.zeros((len(labels), num_grids))
        multi_grid_cell_labels_no_pad = np.zeros((len(labels), num_grids))
        multi_labels_no_pad = np.zeros((len(labels), num_grids, 3))

        for idx in range(len(labels)):
            primary_cell = int(labels[idx, 2])
            candidate_cells = [primary_cell]
            candidate_xy = [labels[idx, :2]]

            multi_grid_cell_labels_no_pad[idx, primary_cell] = 1.0
            multi_labels_no_pad[idx, primary_cell, :2] = labels[idx, :2]
            multi_labels_no_pad[idx, primary_cell, 2] = 1.0

            if idx in self.aug_encoding:
                for aug in self.aug_encoding[idx]:
                    candidate_cells.append(int(aug[2]))
                    candidate_xy.append(aug[:2])

            distances = [
                float(np.sum(np.square(xy))) for xy in candidate_xy
            ]
            class_weights = self._inverse_distance_weights(distances)

            for cell, xy, class_weight in zip(
                candidate_cells, candidate_xy, class_weights
            ):
                multi_labels[idx, cell, :2] = xy
                multi_labels[idx, cell, 2] = (
                    class_weight if weighted_grid_labels else 1.0
                )
                multi_grid_cell_labels[idx, cell] = class_weight

        self.multi_labels = np.reshape(
            multi_labels, [len(multi_labels), num_grids * 3]
        )
        self.multi_grid_cell_labels = multi_grid_cell_labels
        self.multi_labels_no_pad = np.reshape(
            multi_labels_no_pad,
            [len(multi_labels_no_pad), num_grids * 3]
        )
        self.multi_grid_cell_labels_no_pad = (
            multi_grid_cell_labels_no_pad
        )
        return self

    def convert_from_2dim_overlapping_grid(
        self, grid_pred, within_cell_reg, width, height, offset=0
    ):
        """Decode X/Y with neighbor-aware probability blending."""
        grid_size = self.pr.get_param('grid_size')
        padding_ratio = self.pr.get_param('padding_ratio')
        origins, num_local = compute_grid_cell_origins_of_encoding(
            width, height, grid_size
        )
        scale = grid_size / 2.0 + grid_size * padding_ratio
        top_k = max(1, int(self._clf_param('decoder_top_k', 1)))
        top_k = min(top_k, num_local)
        secondary_ratio = float(
            self._clf_param('decoder_secondary_ratio', 0.35)
        )
        neighbor_radius = (
            float(self._clf_param('decoder_neighbor_radius_cells', 1.5))
            * grid_size
        )

        pred_fold = np.zeros((len(grid_pred), 2))

        for idx in range(len(grid_pred)):
            local_probs = np.asarray(
                grid_pred[idx, offset:offset + num_local],
                dtype=np.float64,
            )
            primary_local = int(np.argmax(local_probs))
            primary_prob = float(local_probs[primary_local])

            if top_k == 1:
                candidate_ids = [primary_local]
            else:
                ranked = np.argsort(local_probs)[::-1][:top_k]
                candidate_ids = [primary_local]
                for local_id in ranked:
                    local_id = int(local_id)
                    if local_id == primary_local:
                        continue
                    if primary_prob > 0.0:
                        relative = (
                            float(local_probs[local_id]) / primary_prob
                        )
                    else:
                        relative = 0.0
                    distance = np.linalg.norm(
                        origins[local_id] - origins[primary_local]
                    )
                    if (
                        relative >= secondary_ratio
                        and distance <= neighbor_radius
                    ):
                        candidate_ids.append(local_id)

            weights = np.array(
                [max(float(local_probs[i]), 0.0) for i in candidate_ids],
                dtype=np.float64,
            )
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                weights = np.full(
                    len(candidate_ids), 1.0 / len(candidate_ids)
                )
            else:
                weights /= weight_sum

            candidates = []
            for local_id in candidate_ids:
                global_id = offset + local_id
                encoded_pred = within_cell_reg[
                    idx, global_id * 2:(global_id + 1) * 2
                ]
                candidates.append(
                    origins[local_id] + encoded_pred[:2] * scale
                )

            pred_fold[idx] = np.sum(
                np.asarray(candidates)
                * weights.reshape(-1, 1),
                axis=0,
            )

        return pred_fold

    def get_prediction_confidence(self, grid_pred):
        """Return confidence, top-2 margin, and acceptance per sample."""
        probs = np.asarray(grid_pred, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] == 0:
            raise ValueError('grid_pred must be [batch, grid_cells]')

        chosen = np.argmax(probs, axis=1)
        confidence = probs[np.arange(len(probs)), chosen]

        if probs.shape[1] == 1:
            margin = confidence.copy()
        else:
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        min_confidence = float(
            self._clf_param('decoder_min_confidence', 0.0)
        )
        min_margin = float(
            self._clf_param('decoder_min_margin', 0.0)
        )
        accepted = np.logical_and(
            confidence >= min_confidence,
            margin >= min_margin,
        )
        return {
            'cell_id': chosen,
            'confidence': confidence,
            'margin': margin,
            'accepted': accepted,
        }

    def decode_live_predictions(self, grid_pred, within_cell_reg):
        """Decode live predictions and reject uncertain marker updates."""
        status = self.get_prediction_confidence(grid_pred)
        chosen = status['cell_id']
        floor_ids = self.get_floors_of_grid_cells(chosen)
        xy = np.full((len(grid_pred), 2), np.nan, dtype=np.float64)

        offset = 0
        for f_idx in range(self.num_floors):
            sub_idx = np.where(floor_ids == f_idx)[0]
            if f_idx > 0:
                offset = int(
                    np.cumsum(self.grid_per_floor)[f_idx - 1]
                )
            if len(sub_idx) == 0:
                continue
            xy[sub_idx] = self.convert_from_2dim_overlapping_grid(
                grid_pred[sub_idx],
                within_cell_reg[sub_idx],
                offset=offset,
                height=self.floorplan_height[f_idx],
                width=self.floorplan_width[f_idx],
            )

        rejected = ~status['accepted']
        xy[rejected] = np.nan
        return {
            'xy': xy,
            'floor_id': np.where(status['accepted'], floor_ids, -1),
            'cell_id': np.where(status['accepted'], chosen, -1),
            'confidence': status['confidence'],
            'margin': status['margin'],
            'accepted': status['accepted'],
        }
