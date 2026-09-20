# CLF Multi-CEL sensor-fusion extension

This branch adapts the original Multi-CEL indoor-localization pipeline for CLF fingerprints.

## Input contract
Each row is a labelled observation. Required columns are `x`, `y`, and `floor`. Wi-Fi columns begin with `wifi_`; BLE columns begin with `ble_`. Motion can include `steps`, `heading_deg`, and `distance_m`. Heading is converted to sine/cosine before training to avoid the 0/360-degree discontinuity.

Use stable identifiers in sensor column names (for example BSSID/beacon ID), not SSID names. Missing RSSI is represented as -110 dBm.

An optional `split` column with train/val/test is strongly recommended. Put complete collection walks/sessions into only one split so adjacent samples from the same walk do not leak into both training and testing.

## Training
Place the exported CLF CSV at `datasets/clf/fingerprints.csv`, then run:

`python pipeline.py -c config/clf/config.yml`

The original mCEL grid-cell classification + within-cell regression heads remain intact. The change is the multimodal feature vector used by the backbone.

## Evaluation
Compare at least these ablations on the same held-out walks:
1. Wi-Fi only
2. Wi-Fi + BLE
3. Wi-Fi + BLE + motion

Report floor accuracy, mean/median position error, and latency. Do not claim that BLE/motion improves accuracy until held-out results show it.
