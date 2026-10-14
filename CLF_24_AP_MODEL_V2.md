# CLF 24-AP model contract (model v2, features v3)

The new model input contract is `clf-features-v3`, with exactly 24 Wi-Fi
columns in fixed BSSID order and the existing 19-element motion input.
The neural architecture remains two named input tensors:
`wifi_input[24]` and `motion_input[19]`, with BLE disabled.

## 24 AP identities

- Positions 0–12: original 13 verified BSSIDs, unchanged in order.
- Positions 13–23: 11 actual campus *hardware MAC* identities provided in
  the Ground Floor inventory. They are **candidate broadcast BSSIDs**, not
  invented Wi-Fi access-point measurements.
- A radio observation matches those new positions only if the *real observed*
  BSSID equals that candidate MAC. If the AP broadcasts a different BSSID,
  the identity must be confirmed and the versioned contract updated in
  the collector, training repo and backend before using it in training.
- Floor labels: 17 Ground Floor (6 original + 11 candidate), 7 First Floor.
- Duplicates retain strongest RSSI; missing values map to `-110 dBm` and
  normalize to 0; no synthetic RSSI readings are created.

## Readiness gates

The included `artifacts/models/clf_v2` package consists of **templates**,
with `status: model_not_trained`, empty grid cells, and unfitted motion
scaler. A real trained model cannot be claimed yet.

Training and export still require field-measured coordinates, real
node-labelled samples, train/validation/test splits by survey session, a fitted
training-only motion scaler, and evaluation on real held-out sessions.
The old v1 model folder remains unmodified for reference/rollback.

Backend runtime must reject any 13-input v1 model as incompatible with v3.
