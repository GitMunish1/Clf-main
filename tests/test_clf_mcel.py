import json
from types import SimpleNamespace
import numpy as np
import pytest
import tensorflow as tf
from data.base_data_provider import BaseDataProvider
from data.mcel_data_provider import MCELdataProvider
from data.clf_data_provider import CLFdataProvider
from model.clf_artifact import (build_clf_model, make_feature_contract,
                               make_grid_metadata, write_untrained_package)
from model.mcel_model import get_mcel_loss


def contract():
    return make_feature_contract([f'02:00:00:00:00:{i:02x}' for i in range(24)])


def grid():
    return make_grid_metadata([{'floor': f, 'width_m': 3, 'height_m': 3} for f in (0, 1)], 3, .3)


def test_authentic_architecture_24_ap_forward_gradients_and_loss():
    model = build_clf_model(8)
    assert model.input_shape == (None, 24)
    assert model.output_shape == [(None, 8), (None, 16)]
    assert model.count_params() == 550424
    x = tf.constant(np.linspace(0, 1, 48).reshape(2, 24), dtype=tf.float32)
    y_class = tf.one_hot([0, 5], 8)
    targets = np.zeros((2, 8, 3), dtype=np.float32)
    targets[0, 0] = [.1, -.1, 1]
    targets[1, 5] = [.2, .3, 1]
    with tf.GradientTape() as tape:
        scores, offsets = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_class, scores))
        loss += tf.reduce_mean(get_mcel_loss(8)(targets.reshape(2, -1), offsets))
    grads = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
    assert np.isfinite(loss.numpy())
    np.testing.assert_allclose(tf.reduce_sum(scores, axis=1), 1, atol=1e-6)
    assert np.max(np.abs(offsets)) <= 1


def test_masked_loss_ignores_inactive_cells():
    target = np.array([[[.5, -.5, 1], [0, 0, 0]]], dtype=np.float32)
    pred = tf.Variable([[0., 0., .9, .9]])
    with tf.GradientTape() as tape:
        loss = get_mcel_loss(2)(target.reshape(1, -1), pred)
    np.testing.assert_allclose(loss.numpy(), [.5])
    np.testing.assert_allclose(tape.gradient(loss, pred).numpy(), [[-1, 1, 0, 0]])


def test_exact_centre_overlap_has_finite_labels():
    dp = MCELdataProvider({'grid_size': 3, 'padding_ratio': .6}, SimpleNamespace())
    dp.pos, dp.floor = np.array([[3., 3.]]), np.array([0])
    dp.floors, dp.num_floors = [0], 1
    dp.floorplan_width, dp.floorplan_height = [6], [6]
    dp.transform_to_grid_encoding().compute_multilabel_aug_data(True)
    assert np.isfinite(dp.multi_grid_cell_labels).all()
    assert np.isfinite(dp.multi_labels).all()
    np.testing.assert_allclose(dp.multi_grid_cell_labels.sum(axis=1), 1)


@pytest.mark.parametrize('method', ['min_max', 'standard'])
def test_preprocessing_fits_training_rows_only(method):
    dp = BaseDataProvider({}, SimpleNamespace())
    dp.rss = np.array([[-100., -90.], [-80., -70.], [900., 900.]])
    dp.split_indices = [{'train': [0, 1], 'val': [], 'test': [2]}]
    dp.standardize_data(method)
    expected = [0, 1/3] if method == 'min_max' else [-1, -1]
    np.testing.assert_allclose(dp.x[0], expected)


def test_grid_coordinates_and_loss_encoding_round_trip():
    meta = grid()
    assert len(meta['cells']) == 8
    np.testing.assert_allclose(meta['regression_scale_m'], 2.4)
    dp = MCELdataProvider({'grid_size': 3, 'padding_ratio': .3}, SimpleNamespace())
    point = np.array([[1.1, 2.2]])
    labels, _, count = dp.transform_to_grid_encoding_(point, 3, 3, 0)
    probabilities = np.zeros((1, count))
    probabilities[0, int(labels[0, 2])] = 1
    offsets = np.zeros((1, 2 * count))
    j = int(labels[0, 2]); offsets[0, 2*j:2*j+2] = labels[0, :2]
    np.testing.assert_allclose(dp.convert_from_2dim_overlapping_grid(probabilities, offsets, 3, 3), point)


def test_real_keras_serialization_is_untrained_and_round_trips(tmp_path):
    model = build_clf_model(8)
    calibration = {'calibrated': False, 'units': 'metres', 'map_to_local_m': {}}
    package = write_untrained_package(model, tmp_path/'package', contract(), grid(), calibration)
    restored = tf.keras.models.load_model(package/'clf_localization.keras', compile=False, safe_mode=True)
    x = np.zeros((1, 24), dtype=np.float32)
    for before, after in zip(model(x, training=False), restored(x, training=False)):
        np.testing.assert_array_equal(before.numpy(), after.numpy())
    assert json.loads((package/'model_metadata.json').read_text())['status'] == 'untrained'
    with pytest.raises(FileExistsError):
        write_untrained_package(model, package, contract(), grid(), calibration)


def provider(partitions):
    return CLFdataProvider({'grid_size':3, 'padding_ratio':.3},
        np.full((6,24), -65.), np.array([[1,1]]*6), [0,1,0,1,0,1],
        ['a','a','b','b','c','c'], partitions, {0:(3,3),1:(3,3)}, contract())


def test_clf_session_splits_and_fixed_wifi_contract():
    dp = provider({'train':['a'], 'val':['b'], 'test':['c']})
    assert dp.get_input_dim() == 24 and dp.get_output_dim() == [8,16]
    assert dp.get_x('train').shape == (2,24)
    np.testing.assert_allclose(dp.x, .5)
    assert np.isfinite(dp.get_y('train')[1]).all()


def test_clf_rejects_session_leakage():
    with pytest.raises(ValueError, match='disjoint'):
        provider({'train':['a','b'], 'val':['b'], 'test':['c']})


def test_checkpoint_paths_support_new_saves_and_legacy_loads(tmp_path):
    from model.base_model import BaseModel
    model = BaseModel({}, None, str(tmp_path))
    assert model.weights_path().endswith('model.weights.h5')
    tf.keras.callbacks.ModelCheckpoint(model.weights_path(), save_weights_only=True)
    (tmp_path/'model.hdf5').write_text('legacy test path only')
    assert model.weights_path(for_loading=True).endswith('model.hdf5')


def test_authentic_training_wrapper_compiles_without_training(tmp_path):
    import yaml
    from pathlib import Path
    from model.model_factory import get_model
    dp=provider({'train':['a'],'val':['b'],'test':['c']})
    conf=yaml.safe_load((Path(__file__).parents[1]/'config/clf/wifi24.yml').read_text())
    wrapped=get_model(conf,str(tmp_path),dp)
    assert wrapped.model.optimizer is not None
    assert wrapped.model.input_shape==(None,24)
    assert set(wrapped.model.loss)=={'output_class','output_reg'}
