from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.getenv("CLF_DATABASE_PATH", str(BASE_DIR / "runtime" / "clf_runtime.db")))
MODEL_PATH = Path(os.getenv("CLF_MODEL_PATH", str(BASE_DIR / "artifacts" / "clf_fusion.keras")))
AP_REGISTRY_PATH = Path(os.getenv("CLF_AP_REGISTRY_PATH", str(BASE_DIR / "artifacts" / "ap_registry.json")))
MAPS_DIR = Path(os.getenv("CLF_MAPS_DIR", str(BASE_DIR / "artifacts" / "maps")))
WIFI_DIM = int(os.getenv("CLF_WIFI_DIM", "128"))
BLE_DIM = int(os.getenv("CLF_BLE_DIM", "64"))
MOTION_DIM = int(os.getenv("CLF_MOTION_DIM", "12"))
NUM_FLOORS = int(os.getenv("CLF_NUM_FLOORS", "2"))
NUM_CELLS = int(os.getenv("CLF_NUM_CELLS", "128"))
EXPOSE_UNTRAINED_LOGITS = os.getenv("CLF_EXPOSE_UNTRAINED_LOGITS", "false").lower() == "true"
FEATURE_VERSION = "clf-features-v1"


class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class MotionSample(BaseModel):
    accelerometer: Vector3 | None = None
    gyroscope: Vector3 | None = None
    linear_acceleration: Vector3 | None = None
    magnetometer: Vector3 | None = None


class WiFiObservation(BaseModel):
    bssid: str = Field(min_length=2, max_length=64)
    rssi_dbm: float = Field(ge=-150, le=0)
    ssid: str | None = Field(default=None, max_length=128)
    frequency_mhz: int | None = Field(default=None, ge=2000, le=8000)

    @field_validator("bssid")
    @classmethod
    def normalize_bssid(cls, value: str) -> str:
        return value.strip().lower()


class BLEObservation(BaseModel):
    beacon_id: str = Field(min_length=1, max_length=160)
    rssi_dbm: float = Field(ge=-150, le=0)
    tx_power: float | None = Field(default=None, ge=-150, le=50)

    @field_validator("beacon_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip().lower()


class StepSample(BaseModel):
    total: int | None = Field(default=None, ge=0)
    delta: int = Field(default=0, ge=0, le=1000)


class SensorPacket(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = "1.0"
    packet_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    device_id: str = Field(min_length=1, max_length=128)
    timestamp_ms: int = Field(gt=0)
    wifi: list[WiFiObservation] = Field(default_factory=list, max_length=256)
    ble: list[BLEObservation] = Field(default_factory=list, max_length=256)
    motion: MotionSample | None = None
    steps: StepSample | None = None
    heading_deg: float | None = None

    @field_validator("heading_deg")
    @classmethod
    def normalize_heading(cls, value: float | None) -> float | None:
        return None if value is None else value % 360.0


class SessionCreate(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    app_version: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelStatus(BaseModel):
    state: Literal["trained", "untrained", "wiring_only", "error"]
    model_version: str
    weights_loaded: bool
    backend: str
    detail: str


class LocationEstimate(BaseModel):
    floor: int | None = None
    cell_id: int | None = None
    x: float | None = None
    y: float | None = None
    confidence: float | None = None
    map_matched: bool = False


class LocalizeResponse(BaseModel):
    packet_id: UUID
    session_id: UUID
    accepted: bool = True
    model: ModelStatus
    location: LocationEstimate | None = None
    inference_ms: float
    feature_version: str = FEATURE_VERSION
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


@dataclass(slots=True)
class StorageJob:
    kind: str
    data: dict[str, Any]


class StorageManager:
    def __init__(self, path: Path):
        self.path = path
        self.queue: asyncio.Queue[StorageJob | None] = asyncio.Queue(maxsize=5000)
        self.worker: asyncio.Task | None = None

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._init_db)
        self.worker = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        if self.worker:
            await self.queue.put(None)
            await self.worker
            self.worker = None

    async def enqueue(self, kind: str, data: dict[str, Any]) -> None:
        await self.queue.put(StorageJob(kind, data))

    async def get_packet(self, packet_id: str) -> dict[str, Any] | None:
        await self.queue.join()
        return await asyncio.to_thread(self._get_packet, packet_id)

    async def _writer_loop(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                await asyncio.to_thread(self._write, job)
            finally:
                self.queue.task_done()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, app_version TEXT,
                metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sensor_packets(
                packet_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, device_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL, received_at TEXT NOT NULL, payload_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_packets_session_time ON sensor_packets(session_id, timestamp_ms);
            CREATE TABLE IF NOT EXISTS processed_features(
                packet_id TEXT PRIMARY KEY, feature_version TEXT NOT NULL, wifi_json TEXT NOT NULL,
                ble_json TEXT NOT NULL, motion_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(
                packet_id TEXT PRIMARY KEY, model_version TEXT NOT NULL, model_state TEXT NOT NULL,
                floor INTEGER, cell_id INTEGER, x REAL, y REAL, confidence REAL,
                map_matched INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    def _write(self, job: StorageJob) -> None:
        d = job.data
        with self._connect() as conn:
            if job.kind == "session":
                conn.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?)",
                             (d["session_id"], d["device_id"], d.get("app_version"), d["metadata_json"], d["created_at"]))
            elif job.kind == "raw":
                conn.execute("INSERT OR REPLACE INTO sensor_packets VALUES(?,?,?,?,?,?)",
                             (d["packet_id"], d["session_id"], d["device_id"], d["timestamp_ms"], d["received_at"], d["payload_json"]))
            elif job.kind == "processed":
                conn.execute("INSERT OR REPLACE INTO processed_features VALUES(?,?,?,?,?,?)",
                             (d["packet_id"], FEATURE_VERSION, d["wifi_json"], d["ble_json"], d["motion_json"], d["created_at"]))
            elif job.kind == "prediction":
                conn.execute("INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                    d["packet_id"], d["model_version"], d["model_state"], d.get("floor"), d.get("cell_id"),
                    d.get("x"), d.get("y"), d.get("confidence"), int(d.get("map_matched", False)),
                    d["metadata_json"], d["created_at"]))

    def _get_packet(self, packet_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("""SELECT s.*, p.feature_version, p.wifi_json, p.ble_json, p.motion_json,
                r.model_version, r.model_state, r.floor, r.cell_id, r.x, r.y, r.confidence, r.map_matched,
                r.metadata_json AS prediction_metadata_json
                FROM sensor_packets s
                LEFT JOIN processed_features p ON p.packet_id=s.packet_id
                LEFT JOIN predictions r ON r.packet_id=s.packet_id
                WHERE s.packet_id=?""", (packet_id,)).fetchone()
            return dict(row) if row else None


@dataclass(slots=True)
class FeatureBatch:
    wifi: np.ndarray
    ble: np.ndarray
    motion: np.ndarray

    def model_inputs(self) -> dict[str, np.ndarray]:
        return {"wifi": self.wifi[None, :], "ble": self.ble[None, :], "motion": self.motion[None, :]}


class SensorPreprocessor:
    def __init__(self):
        self.registry = self._load_registry()
        self.wifi_index = {k.lower(): int(v) for k, v in self.registry.get("wifi", {}).items()}
        self.ble_index = {k.lower(): int(v) for k, v in self.registry.get("ble", {}).items()}

    def transform(self, packet: SensorPacket) -> FeatureBatch:
        missing = self._norm_rssi(-110)
        wifi = np.full(WIFI_DIM, missing, dtype=np.float32)
        ble = np.full(BLE_DIM, missing, dtype=np.float32)
        for obs in packet.wifi:
            idx = self._slot(obs.bssid, self.wifi_index, WIFI_DIM, "wifi")
            wifi[idx] = max(wifi[idx], self._norm_rssi(obs.rssi_dbm))
        for obs in packet.ble:
            idx = self._slot(obs.beacon_id, self.ble_index, BLE_DIM, "ble")
            ble[idx] = max(ble[idx], self._norm_rssi(obs.rssi_dbm))

        a = packet.motion.accelerometer if packet.motion and packet.motion.accelerometer else Vector3()
        g = packet.motion.gyroscope if packet.motion and packet.motion.gyroscope else Vector3()
        l = packet.motion.linear_acceleration if packet.motion and packet.motion.linear_acceleration else Vector3()
        h = math.radians(packet.heading_deg or 0.0)
        values = [a.x, a.y, a.z, self._mag(a), g.x, g.y, g.z, self._mag(g), self._mag(l),
                  float(packet.steps.delta if packet.steps else 0), math.sin(h), math.cos(h)]
        motion = np.asarray(values[:MOTION_DIM], dtype=np.float32)
        if len(motion) < MOTION_DIM:
            motion = np.pad(motion, (0, MOTION_DIM-len(motion))).astype(np.float32)
        return FeatureBatch(wifi, ble, motion)

    @staticmethod
    def _norm_rssi(value: float) -> float:
        v = min(-30.0, max(-110.0, float(value)))
        return ((v + 110.0) / 80.0) * 2.0 - 1.0

    @staticmethod
    def _mag(v: Vector3) -> float:
        return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    @staticmethod
    def _slot(key: str, registry: dict[str, int], dim: int, namespace: str) -> int:
        if key in registry and 0 <= registry[key] < dim:
            return registry[key]
        digest = hashlib.blake2b(f"{namespace}:{key}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % dim

    @staticmethod
    def _load_registry() -> dict[str, Any]:
        if not AP_REGISTRY_PATH.exists():
            return {"wifi": {}, "ble": {}}
        try:
            return json.loads(AP_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"wifi": {}, "ble": {}}


@dataclass(slots=True)
class Prediction:
    floor: int | None
    cell_id: int | None
    x: float | None
    y: float | None
    confidence: float | None
    inference_ms: float
    debug: dict[str, Any]


class CLFModelRuntime:
    def __init__(self):
        self.model = None
        self.backend = "numpy"
        self.weights_loaded = False
        self.state: Literal["trained", "untrained", "wiring_only", "error"] = "wiring_only"
        self.version = "clf-fusion-untrained-v1"
        self.detail = "TensorFlow not installed; receive/process/store pipeline is active."
        self._load()

    def _load(self) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            return
        self.backend = "tensorflow"
        try:
            if MODEL_PATH.exists():
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                self.state, self.weights_loaded, self.version = "trained", True, MODEL_PATH.stem
                self.detail = f"Loaded trained model: {MODEL_PATH}"
            else:
                self.model = self._build_untrained(tf)
                self.state = "untrained"
                self.detail = "Untrained CLF fusion model loaded for wiring validation; location output is suppressed."
        except Exception as exc:
            self.model = None
            self.state = "error"
            self.detail = f"Model initialization failed: {exc}"

    @staticmethod
    def _build_untrained(tf):
        wi = tf.keras.Input((WIFI_DIM,), name="wifi")
        bi = tf.keras.Input((BLE_DIM,), name="ble")
        mi = tf.keras.Input((MOTION_DIM,), name="motion")
        w = tf.keras.layers.Dense(128, activation="relu")(wi)
        w = tf.keras.layers.Dense(64, activation="relu")(w)
        b = tf.keras.layers.Dense(64, activation="relu")(bi)
        b = tf.keras.layers.Dense(32, activation="relu")(b)
        m = tf.keras.layers.Dense(32, activation="relu")(mi)
        m = tf.keras.layers.Dense(16, activation="relu")(m)
        f = tf.keras.layers.Concatenate(name="sensor_fusion")([w, b, m])
        f = tf.keras.layers.Dense(128, activation="relu")(f)
        f = tf.keras.layers.Dense(64, activation="relu")(f)
        return tf.keras.Model({"wifi": wi, "ble": bi, "motion": mi}, {
            "floor": tf.keras.layers.Dense(NUM_FLOORS, activation="softmax", name="floor_head")(f),
            "cell": tf.keras.layers.Dense(NUM_CELLS, activation="softmax", name="cell_head")(f),
            "position": tf.keras.layers.Dense(2, name="position_head")(f),
        }, name="clf_sensor_fusion")

    def status(self) -> ModelStatus:
        return ModelStatus(state=self.state, model_version=self.version, weights_loaded=self.weights_loaded,
                           backend=self.backend, detail=self.detail)

    def predict(self, inputs: dict[str, np.ndarray]) -> Prediction:
        start = perf_counter()
        debug = {"input_shapes": {k: list(v.shape) for k, v in inputs.items()}}
        if self.model is None:
            return Prediction(None, None, None, None, None, (perf_counter()-start)*1000, debug)
        out = self.model(inputs, training=False)
        floor = np.asarray(out["floor"])[0]
        cell = np.asarray(out["cell"])[0]
        pos = np.asarray(out["position"])[0]
        if self.state != "trained":
            if EXPOSE_UNTRAINED_LOGITS:
                debug.update({"untrained_floor_argmax": int(np.argmax(floor)), "untrained_cell_argmax": int(np.argmax(cell)),
                              "untrained_position": [float(pos[0]), float(pos[1])]})
            return Prediction(None, None, None, None, None, (perf_counter()-start)*1000, debug)
        return Prediction(int(np.argmax(floor)), int(np.argmax(cell)), float(pos[0]), float(pos[1]),
                          float(np.max(floor)*np.max(cell)), (perf_counter()-start)*1000, debug)


class MapMatcher:
    def __init__(self):
        self.features: dict[int, list[tuple[str | None, Any]]] = {}
        self.enabled = False
        try:
            from shapely.geometry import shape  # type: ignore
            self._shape = shape
            self.enabled = True
        except Exception:
            return
        if not MAPS_DIR.exists():
            return
        for path in MAPS_DIR.glob("*.geojson"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for feature in data.get("features", []):
                props = feature.get("properties", {}) or {}
                if props.get("navigable") is False:
                    continue
                kind = str(props.get("kind", props.get("type", ""))).lower()
                if kind and kind not in {"corridor", "walkway", "room", "stairs", "lift", "entrance", "lobby"}:
                    continue
                try:
                    floor = int(props.get("floor", props.get("level", 0)))
                    geom = self._shape(feature["geometry"])
                except Exception:
                    continue
                feature_id = props.get("id") or props.get("node_id") or props.get("name")
                self.features.setdefault(floor, []).append((str(feature_id) if feature_id else None, geom))

    def match(self, floor: int, x: float, y: float) -> tuple[float, float, bool, str | None]:
        if not self.enabled or floor not in self.features:
            return x, y, False, None
        from shapely.geometry import Point  # type: ignore
        from shapely.ops import nearest_points  # type: ignore
        p = Point(x, y)
        for fid, geom in self.features[floor]:
            if geom.contains(p) or geom.touches(p):
                return x, y, True, fid
        fid, geom = min(self.features[floor], key=lambda item: item[1].distance(p))
        snap = nearest_points(p, geom)[1]
        return float(snap.x), float(snap.y), True, fid


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.storage = StorageManager(DATABASE_PATH)
    await app.state.storage.start()
    app.state.preprocessor = SensorPreprocessor()
    app.state.model = CLFModelRuntime()
    app.state.matcher = MapMatcher()
    try:
        yield
    finally:
        await app.state.storage.stop()


app = FastAPI(title="CLF Location Engine", version="0.1.0", lifespan=lifespan)


def storage(request: Request) -> StorageManager:
    return request.app.state.storage


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CLF Location Engine", "version": "0.1.0"}


@app.get("/ready")
async def ready(request: Request):
    status = request.app.state.model.status()
    return {"ready": status.state != "error", "model_state": status.state,
            "weights_loaded": status.weights_loaded, "storage_queue_depth": request.app.state.storage.queue.qsize()}


@app.get("/api/v1/model/status", response_model=ModelStatus)
async def model_status(request: Request):
    return request.app.state.model.status()


@app.post("/api/v1/sessions", status_code=201)
async def create_session(payload: SessionCreate, db: StorageManager = Depends(storage)):
    sid = uuid4()
    await db.enqueue("session", {"session_id": str(sid), "device_id": payload.device_id, "app_version": payload.app_version,
                                 "metadata_json": json.dumps(payload.metadata, separators=(",", ":")), "created_at": utcnow()})
    return {"session_id": str(sid), "created_at": utcnow()}


async def process_packet(request: Request, packet: SensorPacket) -> LocalizeResponse:
    db: StorageManager = request.app.state.storage
    received = utcnow()
    await db.enqueue("raw", {"packet_id": str(packet.packet_id), "session_id": str(packet.session_id),
                             "device_id": packet.device_id, "timestamp_ms": packet.timestamp_ms, "received_at": received,
                             "payload_json": json.dumps(packet.model_dump(mode="json"), separators=(",", ":"))})
    features = request.app.state.preprocessor.transform(packet)
    await db.enqueue("processed", {"packet_id": str(packet.packet_id),
        "wifi_json": json.dumps(features.wifi.tolist(), separators=(",", ":")),
        "ble_json": json.dumps(features.ble.tolist(), separators=(",", ":")),
        "motion_json": json.dumps(features.motion.tolist(), separators=(",", ":")), "created_at": received})

    pred = request.app.state.model.predict(features.model_inputs())
    location = None
    warnings: list[str] = []
    meta = dict(pred.debug)
    if request.app.state.model.state == "trained" and pred.floor is not None and pred.x is not None and pred.y is not None:
        x, y, matched, fid = request.app.state.matcher.match(pred.floor, pred.x, pred.y)
        location = LocationEstimate(floor=pred.floor, cell_id=pred.cell_id, x=x, y=y,
                                    confidence=pred.confidence, map_matched=matched)
        meta["map_feature_id"] = fid
    else:
        warnings.append("Model is not trained/loaded yet; data was received, tensorized and stored, but random location output was suppressed.")

    await db.enqueue("prediction", {"packet_id": str(packet.packet_id), "model_version": request.app.state.model.version,
        "model_state": request.app.state.model.state, "floor": location.floor if location else None,
        "cell_id": location.cell_id if location else None, "x": location.x if location else None,
        "y": location.y if location else None, "confidence": location.confidence if location else None,
        "map_matched": location.map_matched if location else False,
        "metadata_json": json.dumps(meta, separators=(",", ":")), "created_at": received})
    return LocalizeResponse(packet_id=packet.packet_id, session_id=packet.session_id,
                            model=request.app.state.model.status(), location=location,
                            inference_ms=round(pred.inference_ms, 3), warnings=warnings,
                            debug=pred.debug if EXPOSE_UNTRAINED_LOGITS else None)


@app.post("/api/v1/localize", response_model=LocalizeResponse)
async def localize(request: Request, packet: SensorPacket):
    return await process_packet(request, packet)


@app.get("/api/v1/debug/packets/{packet_id}")
async def packet_debug(packet_id: UUID, db: StorageManager = Depends(storage)):
    row = await db.get_packet(str(packet_id))
    if row is None:
        raise HTTPException(404, "Packet not found")
    return row


@app.websocket("/ws/location")
async def ws_location(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                packet = SensorPacket.model_validate(await websocket.receive_json())
                result = await process_packet(websocket, packet)
                await websocket.send_json(result.model_dump(mode="json"))
            except ValidationError as exc:
                await websocket.send_json({"accepted": False, "error": "validation_error", "detail": exc.errors()})
    except WebSocketDisconnect:
        return
