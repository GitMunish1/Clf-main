from data.gia_vslam_data_connector import GiaVSLAMdataConnector
from data.mcel_data_provider import MCELdataProvider
from data.reg_data_provider import RegDataProvider
from data.tampere_data_connector import TampereDataConnector
from data.uji_data_connector import UJIdataConnector
from data.uts_data_connector import UTSdataConnector
from data.clf_data_connector import CLFDataConnector


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
        conn = GiaVSLAMdataConnector(floors=d_params['floors'],
                                     devices=d_params['devices'],
                                     test_devices=d_params['test_devices'] if 'test_devices' in d_params else None,
                                     test_trajectories=d_params['test_trajectories'] if 'test_trajectories' in d_params else None
                                     ).load_dataset()
    elif dataset == 'clf':
        conn = CLFDataConnector(
            csv_path=d_params.get('csv_path', 'datasets/clf/fingerprints.csv'),
            wifi_prefix=d_params.get('wifi_prefix', 'wifi_'),
            ble_prefix=d_params.get('ble_prefix', 'ble_'),
            motion_features=d_params.get('motion_features'),
            missing_rssi=d_params.get('missing_rssi', -110.0),
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    if m_type == 'mCEL':
        dp = MCELdataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices().generate_validation_indices()
        dp = dp.replace_missing_values()

        # CLF has mixed feature units. Scale RSSI to approximately [0, 1] while
        # preserving cyclic heading and normalizing motion separately.
        if dataset == 'clf':
            x = dp.rss.astype('float32').copy()
            groups = conn.feature_groups
            radio_idx = groups['wifi'] + groups['ble']
            if radio_idx:
                x[:, radio_idx] = (x[:, radio_idx] + 110.0) / 110.0
                x[:, radio_idx] = x[:, radio_idx].clip(0.0, 1.0)
            motion_names = [conn.feature_names[i] for i in groups['motion']]
            for i, name in zip(groups['motion'], motion_names):
                if name in ('heading_sin', 'heading_cos'):
                    continue
                scale = max(float(abs(x[:, i]).max()), 1.0)
                x[:, i] /= scale
            dp.x = x
        else:
            dp = dp.standardize_data(scaling_type=d_params['scaling'])

        dp = dp.transform_to_grid_encoding()

        class_pad = bool(d_params.get('class_pad', False))
        reg_pad = bool(d_params.get('reg_pad', False))
        weighted_grid_labels = bool(d_params.get('weighted_grid_labels', False))

        dp = dp.compute_multilabel_aug_data(weighted_grid_labels)
        dp = dp.set_labels(class_pad, reg_pad)

    elif m_type == '3D':
        dp = RegDataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices().generate_validation_indices()
        dp = dp.replace_missing_values().standardize_data(scaling_type=d_params['scaling'])
        dp = dp.set_labels(scale_labels=True)

    else:
        dp = None

    return dp
