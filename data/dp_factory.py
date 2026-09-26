import math

from data.gia_vslam_data_connector import GiaVSLAMdataConnector
from data.mcel_data_provider import MCELdataProvider
from data.clf_sensor_fusion_data_provider import CLFSensorFusionDataProvider
from data.reg_data_provider import RegDataProvider
from data.tampere_data_connector import TampereDataConnector
from data.uji_data_connector import UJIdataConnector
from data.uts_data_connector import UTSdataConnector
from data.clf_data_connector import CLFDataConnector


def _validate_clf_grid_scale(dp, d_params):
    """Fail early when map units would create an impractical Multi-CEL head."""
    grid_size = float(d_params.get('grid_size', 3))
    if grid_size <= 0:
        raise ValueError('grid_size must be positive')

    max_grid_cells = int(d_params.get('max_grid_cells', 5000))
    total_cells = 0
    for width, height in zip(
        dp.floorplan_width, dp.floorplan_height
    ):
        rows = int(math.ceil(float(height) / grid_size) + 1)
        cols = int(math.ceil(float(width) / grid_size) + 1)
        total_cells += rows * cols

    if total_cells > max_grid_cells:
        raise ValueError(
            'CLF grid would create {} cells (limit {}). '
            'The x/y coordinates are probably still raw GeoJSON/map units. '
            'Calibrate/convert coordinates to local metres before training, '
            'or choose a grid_size expressed in the same calibrated units.'
            .format(total_cells, max_grid_cells)
        )


def get_data_provider(dataset_params, m_type):

    d_params = dataset_params['params']
    dataset = dataset_params['dataset']

    if dataset == 'tampere':
        conn = TampereDataConnector()
    elif dataset == 'uji':
        conn = UJIdataConnector()
    elif dataset == 'uts':
        conn = UTSdataConnector()
    elif dataset == 'gia_vslam':
        conn = GiaVSLAMdataConnector(
            floors=d_params['floors'],
            devices=d_params['devices'],
            test_devices=d_params['test_devices']
            if 'test_devices' in d_params else None,
            test_trajectories=d_params['test_trajectories']
            if 'test_trajectories' in d_params else None
        ).load_dataset()
    elif dataset == 'clf':
        conn = CLFDataConnector(
            csv_path=d_params.get('csv_path', 'datasets/clf/fingerprints.csv'),
            wifi_prefix=d_params.get('wifi_prefix', 'wifi_'),
            ble_prefix=d_params.get('ble_prefix', 'ble_'),
            imu_prefix=d_params.get('imu_prefix', 'imu_'),
            motion_features=d_params.get('motion_features'),
            missing_rssi=d_params.get('missing_rssi', -110.0),
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    if m_type in ('mCEL', 'mCEL_sensor_fusion'):
        if m_type == 'mCEL_sensor_fusion':
            if dataset != 'clf':
                raise ValueError(
                    'mCEL_sensor_fusion currently requires dataset: clf'
                )
            dp = CLFSensorFusionDataProvider(
                dataset_params['params'], dc=conn
            )
        else:
            dp = MCELdataProvider(dataset_params['params'], dc=conn)

        dp = (
            dp.load_dataset()
            .generate_split_indices()
            .generate_validation_indices()
        )

        if dataset == 'clf':
            _validate_clf_grid_scale(dp, d_params)

        if dataset != 'clf':
            dp = dp.replace_missing_values()

        if m_type == 'mCEL_sensor_fusion':
            dp = dp.build_sensor_inputs()
        elif dataset == 'clf':
            x = dp.rss.astype('float32').copy()
            groups = conn.feature_groups
            radio_idx = groups['wifi'] + groups['ble']
            if radio_idx:
                x[:, radio_idx] = (x[:, radio_idx] + 110.0) / 110.0
                x[:, radio_idx] = x[:, radio_idx].clip(0.0, 1.0)
            dp.x = x
        else:
            dp = dp.standardize_data(scaling_type=d_params['scaling'])

        dp = dp.transform_to_grid_encoding()

        class_pad = bool(d_params.get('class_pad', False))
        reg_pad = bool(d_params.get('reg_pad', False))
        weighted_grid_labels = bool(
            d_params.get('weighted_grid_labels', False)
        )

        dp = dp.compute_multilabel_aug_data(weighted_grid_labels)
        dp = dp.set_labels(class_pad, reg_pad)

    elif m_type == '3D':
        dp = (
            RegDataProvider(dataset_params['params'], dc=conn)
            .load_dataset()
            .generate_split_indices()
            .generate_validation_indices()
        )
        dp = (
            dp.replace_missing_values()
            .standardize_data(scaling_type=d_params['scaling'])
        )
        dp = dp.set_labels(scale_labels=True)

    else:
        raise ValueError('Unsupported model type: {}'.format(m_type))

    return dp
