# CLF V1 Multi-CEL localization

CLF V1 preserves the original Multi-CEL grid-cell classification plus
within-cell X/Y regression design. The CLF production model has exactly two
named inputs:

```text
13 whitelisted Wi-Fi RSSI features -> Dense 256 -> Dense 128 --\
                                                               -> LayerNorm -> Dense 256 -> Dense 256
19 motion features              -> Dense 64  -> Dense 64  ----/                   |-> cell classification
                                                                                  \-> within-cell X/Y regression
```

Floor is derived from the predicted global grid cell. There is no separate
floor network. BLE is intentionally disabled for V1 and no dummy BLE tensor is
accepted. The original public-dataset `mCEL` model remains available.

## Feature contract

`artifacts/contracts/feature_contract.json` is the machine-readable source of
truth (`clf-features-v2`). Wi-Fi order is fixed by normalized BSSID, never AP
name or scan strength. Unknown APs are ignored; duplicate observations retain
the strongest RSSI; missing approved APs use `-110 dBm`.

```text
clipped = clip(rssi, -110, -20)
wifi_input = (clipped + 110) / 90
```

The motion input is always:

```text
accel x/y/z/magnitude,
gyro x/y/z/magnitude,
linear acceleration x/y/z/magnitude,
magnetometer x/y/z/magnitude,
step delta,
heading sin/cos
```

Magnitudes are `sqrt(x²+y²+z²)`. Heading uses
`radians(heading_deg % 360)`. A missing magnetometer becomes four zeros.
Motion indexes 0–16 use z-score parameters fitted only on the training-session
split; heading sine/cosine pass through. The exported `preprocessing.json` is
reused unchanged by the backend. No live per-packet fitting is allowed.

## Real training data

Each Collector row/window must include `node_id`, `floor`, calibrated `x_m` and
`y_m`, the 13 fixed Wi-Fi columns, the raw motion fields, `session_id`, a
timestamp/window identifier, and either an explicit `split` or enough sessions
for deterministic 70/15/15 session splitting. One session can never span
partitions.

`datasets/clf/fingerprints.example.csv` is documentation/test data only. The
connector explicitly refuses to train on that filename. Synthetic RSSI is used
only in software-contract tests.

Training cannot begin until all three conditions are met:

1. Real node-labelled survey fingerprints exist.
2. `coordinate_calibration.json` confirms measured local-metre coordinates.
3. Train/validation/test survey sessions are defined.

The checked-in calibration, preprocessing, grid, and model metadata files are
templates marked not ready. No physical scale, fingerprint, model, coordinate,
or accuracy has been fabricated.

## Export and evaluation

`clf_export.export_model_package` writes a complete `artifacts/models/clf_v1`
package and rejects uncalibrated or incompatible inputs. It exports the Keras
model, feature contract, AP registry, training-only scaler, generated grid
cells, calibration, and versioned model metadata.

`clf_evaluation.py` reports floor accuracy, grid-cell accuracy, mean/median/P95
position error, confidence acceptance and rejection rates, and inference
latency on held-out sessions. It supports comparable Wi-Fi-only (zeroed motion)
and Wi-Fi+motion evaluation. Any claim that motion improves localization must
come from held-out real survey sessions.
