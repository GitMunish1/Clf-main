# CLF Runtime Backend

This backend completes the non-training side of CLF now: phone sensor reception, validation,
fixed tensor creation, asynchronous SQLite persistence, WebSocket/REST transport, model wiring,
model status, and GeoJSON map matching.

## Runtime flow

Flutter -> sensor JSON -> validate -> queue raw storage -> build Wi-Fi/BLE/motion tensors ->
CLF model runtime -> map match -> response.

The existing research/training code is not replaced.

## Safe untrained-model behavior

If `backend/artifacts/clf_fusion.keras` is missing, the server starts in one of two safe states:

- `untrained`: TensorFlow is installed, so the intended Wi-Fi Encoder + BLE Encoder + Motion
  Encoder -> Fusion -> Floor Head + Cell Head + XY Head network is instantiated.
- `wiring_only`: TensorFlow is not installed, but receiving, processing and storage still work.

Random coordinates from an untrained network are never returned to Flutter.

## Start

From repository root:

    pip install -r backend/requirements.txt
    uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

API docs: http://SERVER_IP:8000/docs

## Endpoints

- GET /health
- GET /ready
- POST /api/v1/sessions
- POST /api/v1/localize
- GET /api/v1/model/status
- GET /api/v1/debug/packets/{packet_id}
- WS /ws/location

## Sensor packet example

    {
      "schema_version": "1.0",
      "session_id": "7b3f113c-434c-4499-bd9f-e6cb585ad938",
      "device_id": "android-phone",
      "timestamp_ms": 1790780000000,
      "wifi": [
        {"bssid": "aa:bb:cc:dd:ee:01", "ssid": "CU-WIFI", "rssi_dbm": -52, "frequency_mhz": 5180}
      ],
      "ble": [
        {"beacon_id": "ground-lobby-01", "rssi_dbm": -67, "tx_power": -59}
      ],
      "motion": {
        "accelerometer": {"x": 0.08, "y": 0.12, "z": 9.75},
        "gyroscope": {"x": 0.01, "y": 0.03, "z": -0.02},
        "linear_acceleration": {"x": 0.02, "y": 0.01, "z": 0.06}
      },
      "steps": {"total": 1240, "delta": 1},
      "heading_deg": 92.4
    }

BLE is optional. Wi-Fi + motion-only packets are accepted.

## Stored data

SQLite stores three layers separately:
1. raw sensor packet,
2. processed Wi-Fi/BLE/motion feature vectors,
3. model/prediction metadata.

The write queue runs outside the live localization path, so prediction does not wait for DB disk writes.

## When the model is trained

Export the exact AP/beacon vocabulary used during training to
`backend/artifacts/ap_registry.json`, save the Keras model as
`backend/artifacts/clf_fusion.keras`, and restart the server.

The response can then contain floor, cell_id, x, y, confidence and map_matched.

## Test

    cd backend
    PYTHONPATH=. pytest -q tests/test_server.py
