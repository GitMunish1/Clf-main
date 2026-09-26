# CLF Multi-CEL sensor-fusion extension

This branch adapts the original Wi-Fi Multi-CEL indoor-localization model for CLF while preserving the core grid-cell classification + within-cell regression design.

## Architecture

```text
Wi-Fi RSSI -> Wi-Fi encoder ---\
BLE RSSI   -> BLE encoder ------> sensor fusion -> Multi-CEL classification -> cell/floor
Motion     -> Motion encoder ---/              \-> Multi-CEL regression     -> X/Y
```

The motion input can contain steps, step delta, heading, walked distance and flattened IMU features. Heading is converted to sine/cosine so 359 degrees remains close to 0 degrees.

The new model type is `mCEL_sensor_fusion`. Original `mCEL` remains available for the public Wi-Fi-only datasets.

## Training CSV

Each row is one labelled observation/window. Required columns:

`x,y,floor`

Radio columns:

- `wifi_<stable-bssid-or-id>`
- `ble_<stable-beacon-id>`

Motion columns may include:

- `steps`
- `step_delta`
- `heading_deg`
- `distance_m`
- any `imu_*` columns such as `imu_accel_mean_x` or `imu_gyro_std_z`

Missing RSSI is represented as `-110` dBm. Missing motion values default to zero.

An optional `split` column with `train`, `val`, and `test` is strongly recommended. Complete survey walks/sessions should stay inside one split to avoid leakage.

## Training

Place the exported CLF dataset at:

`datasets/clf/fingerprints.csv`

Run:

`python pipeline.py -c config/clf/config.yml`

The pipeline performs:

1. Wi-Fi encoding
2. BLE encoding
3. motion/IMU encoding
4. sensor fusion
5. Multi-CEL grid-cell classification
6. within-grid-cell X/Y regression
7. floor derivation from the predicted grid cell

## Production backend input

The backend must create the same ordered inputs used during training:

```text
wifi_input   = stable Wi-Fi RSSI vector
ble_input    = stable BLE RSSI vector
motion_input = [steps, step_delta, sin(heading), cos(heading), distance, imu...]
```

The model should be loaded once when the backend starts. Live location packets should only run preprocessing + inference; they should never retrain or reload the model.

## Evaluation

Use the same held-out survey walks for these ablations:

1. Wi-Fi only
2. Wi-Fi + BLE
3. Wi-Fi + BLE + steps/heading
4. Wi-Fi + BLE + steps/heading + IMU

Report floor accuracy, grid-cell accuracy, mean/median position error, P95 error and inference latency. Do not claim a fusion accuracy improvement until held-out results confirm it.
