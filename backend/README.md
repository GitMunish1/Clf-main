# CLF Runtime Backend

This backend completes the non-training side of CLF now: phone sensor reception, validation,
fixed tensor creation, asynchronous SQLite persistence, WebSocket/REST transport, model wiring,
model status, and GeoJSON map matching.

## Runtime flow

Flutter -> sensor JSON (+ optional ordered route nodes) -> validate -> queue raw storage ->
build Wi-Fi/BLE/motion tensors -> CLF model runtime -> optional map match ->
match predicted position to the supplied route nodes -> response.

For the POC, the backend does **not** run A* or calculate a route. Flutter owns the map,
destination lookup and ordered node sequence. The location engine only predicts the user's
position and reports progress on that supplied sequence.

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
      "heading_deg": 92.4,
      "route": {
        "destination_node_id": "ROOM_204",
        "current_index_hint": 0,
        "arrival_radius": 2.0,
        "nodes": [
          {"node_id": "N12", "floor": 1, "x": 12.4, "y": 8.1},
          {"node_id": "N13", "floor": 1, "x": 18.2, "y": 8.1},
          {"node_id": "N17", "floor": 1, "x": 24.0, "y": 10.5},
          {"node_id": "ROOM_204", "floor": 1, "x": 29.1, "y": 12.0}
        ]
      }
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


## POC navigation contract

The app already contains the floor map, POIs and node coordinates. When a user chooses a
destination, Flutter supplies the ordered route nodes. The server does not search the graph.

Example:

    N12 -> N13 -> N17 -> N21 -> ROOM_204

After localization, the server compares the predicted floor/X/Y with only those supplied nodes
and can return:

    {
      "route_progress": {
        "destination_node_id": "ROOM_204",
        "current_node_id": "N17",
        "next_node_id": "N21",
        "current_index": 2,
        "total_nodes": 5,
        "distance_to_current_node": 1.1,
        "reached_destination": false
      }
    }

This is route-progress tracking, not A* pathfinding. The node coordinates sent by Flutter must
use the same floor coordinate system as the localization model/GeoJSON.
