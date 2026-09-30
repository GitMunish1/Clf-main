import os
from uuid import uuid4

os.environ["CLF_DATABASE_PATH"] = "/tmp/clf-test-runtime.db"
os.environ["CLF_MODEL_PATH"] = "/tmp/no-trained-clf-model.keras"

from fastapi.testclient import TestClient
from server import SensorPacket, SensorPreprocessor, app


def test_fixed_feature_shapes():
    packet = SensorPacket.model_validate({
        "session_id": str(uuid4()),
        "device_id": "phone",
        "timestamp_ms": 1790780000000,
        "wifi": [{"bssid": "AA:BB:CC:DD:EE:01", "rssi_dbm": -55}],
        "steps": {"delta": 2},
        "heading_deg": 450,
        "motion": {"accelerometer": {"x": 0.1, "y": 0.2, "z": 9.7}}
    })
    batch = SensorPreprocessor().transform(packet)
    assert batch.wifi.shape == (128,)
    assert batch.ble.shape == (64,)
    assert batch.motion.shape == (12,)
    assert packet.heading_deg == 90


def test_end_to_end_untrained_backend():
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", json={"device_id": "phone"})
        assert session.status_code == 201
        response = client.post("/api/v1/localize", json={
            "session_id": session.json()["session_id"],
            "device_id": "phone",
            "timestamp_ms": 1790780000000,
            "wifi": [{"bssid": "aa:bb:cc:dd:ee:ff", "rssi_dbm": -52}],
            "steps": {"delta": 1},
            "heading_deg": 90
        })
        assert response.status_code == 200
        body = response.json()
        assert body["model"]["state"] in {"untrained", "wiring_only"}
        assert body["location"] is None
        stored = client.get(f"/api/v1/debug/packets/{body['packet_id']}")
        assert stored.status_code == 200
        assert stored.json()["feature_version"] == "clf-features-v1"
