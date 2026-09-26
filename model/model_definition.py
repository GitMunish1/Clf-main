import tensorflow as tf


def _apply_mlp(x, conf, prefix):
    if conf is None:
        return x
    for idx, units in enumerate(conf.get('layers', [])):
        x = tf.keras.layers.Dense(
            units, name='{}_dense_{}'.format(prefix, idx)
        )(x)
        x = tf.keras.layers.Activation(
            conf.get('activation', 'relu'),
            name='{}_act_{}'.format(prefix, idx)
        )(x)
        if 'dropout' in conf and conf['dropout']:
            x = tf.keras.layers.Dropout(
                conf['dropout'],
                name='{}_dropout_{}'.format(prefix, idx)
            )(x)
    return x


def _apply_sensor_augmentation(x, conf, prefix):
    """Training-only noise/dropout for real phone sensor gaps."""
    if conf is None:
        return x

    noise = float(conf.get('input_noise', 0.0) or 0.0)
    if noise > 0.0:
        x = tf.keras.layers.GaussianNoise(
            noise, name='{}_input_noise'.format(prefix)
        )(x)

    input_dropout = float(
        conf.get('input_dropout', 0.0) or 0.0
    )
    if input_dropout > 0.0:
        x = tf.keras.layers.Dropout(
            input_dropout,
            name='{}_input_dropout'.format(prefix)
        )(x)

    return x


def _apply_modality_dropout(x, conf, prefix):
    """Randomly remove an entire modality during training.

    One dropout mask is sampled per example and broadcast over the encoded
    feature dimension. This teaches fusion to survive temporary Wi-Fi/BLE/
    motion loss instead of assuming every sensor is always available.
    """
    if conf is None:
        return x
    rate = float(conf.get('modality_dropout', 0.0) or 0.0)
    if rate <= 0.0:
        return x
    return tf.keras.layers.Dropout(
        rate,
        noise_shape=(None, 1),
        name='{}_modality_dropout'.format(prefix),
    )(x)


def _build_mcel_heads(head, h_conf, output_dim):
    class_conf = h_conf['classification']
    c_head = _apply_mlp(head, class_conf, 'class')
    c_output = tf.keras.layers.Dense(
        output_dim[0], name='class_logits'
    )(c_head)
    c_output = tf.keras.layers.Activation(
        'softmax', name='output_class'
    )(c_output)

    reg_conf = h_conf['regression']
    r_head = _apply_mlp(head, reg_conf, 'reg')
    r_output = tf.keras.layers.Dense(
        output_dim[1], name='reg_values'
    )(r_head)
    r_output = tf.keras.layers.Activation(
        'tanh', name='output_reg'
    )(r_output)
    return [c_output, r_output]


def _build_sensor_fusion_backbone(conf, input_dim):
    if not isinstance(input_dim, dict):
        raise ValueError(
            'mCEL_sensor_fusion requires named input dimensions'
        )

    encoders = conf.get('encoders', {})
    inputs = {}
    encoded = []

    for modality in ('wifi', 'ble', 'motion'):
        input_name = '{}_input'.format(modality)
        if input_name not in input_dim:
            raise ValueError(
                'Missing sensor-fusion input: {}'.format(input_name)
            )

        encoder_conf = encoders.get(modality) or {}
        inp = tf.keras.layers.Input(
            shape=(input_dim[input_name],),
            name=input_name
        )
        inputs[input_name] = inp

        augmented = _apply_sensor_augmentation(
            inp, encoder_conf, '{}_encoder'.format(modality)
        )
        modality_features = _apply_mlp(
            augmented,
            encoder_conf,
            '{}_encoder'.format(modality)
        )
        modality_features = _apply_modality_dropout(
            modality_features,
            encoder_conf,
            '{}_encoder'.format(modality)
        )
        encoded.append(modality_features)

    fused = tf.keras.layers.Concatenate(
        name='sensor_fusion_concat'
    )(encoded)

    if conf.get('fusion_layer_norm', True):
        fused = tf.keras.layers.LayerNormalization(
            name='sensor_fusion_norm'
        )(fused)

    fused = _apply_mlp(
        fused,
        conf.get('fusion', conf.get('backbone')),
        'fusion'
    )
    return inputs, fused


def get_model_from_yaml_definition(conf, input_dim, output_dim):
    model_type = conf['type']

    if model_type == 'mCEL_sensor_fusion':
        inputs, head = _build_sensor_fusion_backbone(conf, input_dim)
        outputs = _build_mcel_heads(head, conf['head'], output_dim)
        return tf.keras.models.Model(
            inputs=inputs,
            outputs=outputs,
            name='clf_mcel_sensor_fusion'
        )

    input_layer = tf.keras.layers.Input(shape=input_dim, name='input')
    bb = input_layer
    bb_conf = conf['backbone']

    if bb_conf is not None and bb_conf['type'] == 'MLP':
        bb = _apply_mlp(bb, bb_conf, 'backbone')

    if model_type == 'mCEL':
        outputs = _build_mcel_heads(bb, conf['head'], output_dim)
        return tf.keras.models.Model(input_layer, outputs)

    if model_type == '3D' or model_type == '2D':
        head = _apply_mlp(bb, conf['head'], 'regression')
        output = tf.keras.layers.Dense(output_dim)(head)
        output = tf.keras.layers.Activation('linear')(output)
        return tf.keras.models.Model(input_layer, output)

    raise ValueError('Unsupported model type: {}'.format(model_type))
