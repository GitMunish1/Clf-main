"""CLF adapter: ordered raw RSSI and calibrated labels, grouped session splits."""
from types import SimpleNamespace
import numpy as np
from data.mcel_data_provider import MCELdataProvider
from model.clf_artifact import make_feature_contract


class CLFdataProvider(MCELdataProvider):
    def __init__(self, params, rss_dbm, xy_m, floors, session_ids, partitions,
                 floor_dimensions, contract):
        super().__init__(params, SimpleNamespace(split_indices=None))
        size, padding = float(params.get('grid_size', 3)), float(params.get('padding_ratio', .3))
        if not np.isfinite([size, padding]).all() or size <= 0 or not 0 <= padding < 1:
            raise ValueError('invalid grid size or padding')
        # Explicit parameters prevent provider defaults from silently changing encoding.
        self.pr.param_dict = {**params, 'grid_size': size, 'padding_ratio': padding}
        expected = make_feature_contract([x['bssid'] for x in contract['wifi']['access_points']])
        if contract != expected:
            raise ValueError('incompatible CLF feature contract')
        rss, xy, floors = np.asarray(rss_dbm, dtype=float), np.asarray(xy_m, dtype=float), np.asarray(floors)
        sessions = np.asarray(session_ids, dtype=str)
        n = len(rss)
        if (n < 3 or rss.shape != (n, 24) or xy.shape != (n, 2) or
                floors.shape != (n,) or sessions.shape != (n,) or
                not np.isfinite(rss).all() or not np.isfinite(xy).all() or
                not np.isin(floors, [0, 1]).all()):
            raise ValueError('invalid ordered RSSI or calibrated labels')
        if set(partitions) != {'train', 'val', 'test'}:
            raise ValueError('train, val and test session lists are required')
        groups = {key: set(map(str, value)) for key, value in partitions.items()}
        if (any(not v for v in groups.values()) or
                any(groups[a] & groups[b] for a, b in [('train','val'), ('train','test'), ('val','test')]) or
                set.union(*groups.values()) != set(sessions)):
            raise ValueError('session partitions must be nonempty, disjoint and exhaustive')
        split = {key: np.where(np.isin(sessions, list(value)))[0] for key, value in groups.items()}
        if sorted(floor_dimensions) != [0, 1]:
            raise ValueError('explicit calibrated dimensions for floors 0 and 1 required')
        self.rss, self.pos, self.floor = rss, xy, floors
        self.floors, self.num_floors = [0, 1], 2
        self.floorplan_width = [floor_dimensions[f][0] for f in self.floors]
        self.floorplan_height = [floor_dimensions[f][1] for f in self.floors]
        for i, floor in enumerate(self.floors):
            width, height = self.floorplan_width[i], self.floorplan_height[i]
            points = xy[floors == floor]
            if (not np.isfinite([width, height]).all() or min(width, height) <= 0 or
                    np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height)):
                raise ValueError('label coordinates must lie in calibrated floor extents')
        self.split_indices = [split]
        self.feature_contract = expected
        self.standardize_data('clf_fixed_rssi')
        self.transform_to_grid_encoding().compute_multilabel_aug_data(
            weighted_grid_labels=params.get('weighted_grid_labels', False)).set_labels(
                class_pad=True, reg_pad=True)
