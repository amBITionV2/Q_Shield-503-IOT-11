import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import Device
from security import create_jwt, verify_jwt, kyber_generate_keypair
import mqtt_client
from config import GEOFENCE_CENTER, GEOFENCE_RADIUS
import math, uuid, json, requests, logging, hashlib
from datetime import datetime, timezone
from typing import List, Optional
import os
from collections import deque
import threading
from web3 import Web3

# --- Blockchain Config ---
HARDHAT_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
# Note: Ensure this path is correct for your environment
with open(r"/home/wini/qs/Q_Shield-503-IOT-11/qshield_hardhat/qshield_hardhat/artifacts/contracts/Counter.sol/Counter.json") as f:
    contract_json = json.load(f)
    CONTRACT_ABI = contract_json["abi"]
w3 = Web3(Web3.HTTPProvider(HARDHAT_RPC_URL))
w3.eth.default_account = w3.eth.accounts[0]
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
logging.info(f"Connected to blockchain, contract at {CONTRACT_ADDRESS}")

# --- ML Model Imports ---
try:
    from ml.edge_inference import IsolationForestDetector
    from ml.cloud_inference import LSTMAnomalyDetector
    EDGE_MODEL_AVAILABLE = True
    CLOUD_MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML models not available: {e}")
    EDGE_MODEL_AVAILABLE = False
    CLOUD_MODEL_AVAILABLE = False

# --- General Config ---
MAX_DEVICES = 100
TELEMETRY_INTERVAL_SECONDS = 1
TELEMETRY_BUFFER_SIZE = 50  # Store last 50 telemetry records per device

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)

# Root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)

# Custom loggers
def create_logger(name, filename):
    logger = logging.getLogger(name)
    if not logger.handlers: # Prevents duplicate logs on reload
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

input_logger = create_logger("input_logger", "logs/input_telemetry.log")
output_logger = create_logger("output_logger", "logs/output_telemetry.log")
debug_logger = create_logger("debug_logger", "logs/debug.log")
edge_logger = create_logger("edge_logger", "logs/edge_inference.log")
cloud_logger = create_logger("cloud_logger", "logs/cloud_inference.log")

debug_logger.info("Debug logger initialized (geo-fence alerts will appear here)")

# --- ML Models Global Instances ---
edge_detector = None
cloud_detector = None
telemetry_buffers = {}
buffer_lock = threading.Lock()

# --- Blockchain Global Instances ---
MERKLE_BATCH: List[str] = []

# --- ML Helper Functions ---
def initialize_ml_models():
    global edge_detector, cloud_detector
    if EDGE_MODEL_AVAILABLE:
        try:
            edge_detector = IsolationForestDetector()
            if edge_detector.model: edge_logger.info("Edge Isolation Forest model loaded successfully")
            else: edge_logger.warning("Edge model failed to load")
        except Exception as e:
            edge_logger.error(f"Failed to initialize edge model: {e}")
            edge_detector = None
    if CLOUD_MODEL_AVAILABLE:
        try:
            cloud_detector = LSTMAnomalyDetector()
            if cloud_detector.model and cloud_detector.scaler: cloud_logger.info("Cloud LSTM model loaded successfully")
            else: cloud_logger.warning("Cloud model failed to load")
        except Exception as e:
            cloud_logger.error(f"Failed to initialize cloud model: {e}")
            cloud_detector = None

def add_to_telemetry_buffer(device_id: str, telemetry_data: dict):
    with buffer_lock:
        if device_id not in telemetry_buffers:
            telemetry_buffers[device_id] = deque(maxlen=TELEMETRY_BUFFER_SIZE)
        telemetry_buffers[device_id].append(telemetry_data)

def run_ml_inference(telemetry_data: dict, device_id: str):
    inference_results = {"edge_anomaly": None, "cloud_anomaly": None, "combined_risk_level": "normal"}
    if edge_detector and edge_detector.model:
        try:
            edge_result = edge_detector.get_anomaly_score(telemetry_data)
            inference_results["edge_anomaly"] = edge_result
            edge_logger.info(f"Edge inference for {device_id}: {edge_result}")
        except Exception as e:
            edge_logger.error(f"Edge inference failed for {device_id}: {e}")
    if cloud_detector and cloud_detector.model:
        try:
            with buffer_lock:
                buffer_data = list(telemetry_buffers.get(device_id, []))
            if len(buffer_data) >= 30:
                cloud_result = cloud_detector.get_anomaly_score(buffer_data)
                inference_results["cloud_anomaly"] = cloud_result
                cloud_logger.info(f"Cloud inference for {device_id}: {cloud_result}")
        except Exception as e:
            cloud_logger.error(f"Cloud inference failed for {device_id}: {e}")
    
    edge_anomaly = inference_results.get("edge_anomaly")
    cloud_anomaly = inference_results.get("cloud_anomaly")

    if edge_anomaly and edge_anomaly.get("is_anomaly"):
        if cloud_anomaly and cloud_anomaly.get("is_anomaly"):
            if cloud_anomaly.get("severity") == "high": inference_results["combined_risk_level"] = "critical"
            else: inference_results["combined_risk_level"] = "high"
        else: inference_results["combined_risk_level"] = "medium"
    elif cloud_anomaly and cloud_anomaly.get("is_anomaly"):
        inference_results["combined_risk_level"] = "medium"
    return inference_results

# --- Blockchain Helper Functions ---
def log_event_to_blockchain(merkle_root: str, ipfs_cid: str) -> str:
    try:
        merkle_root_bytes = bytes.fromhex(merkle_root)
        tx_hash = contract.functions.logBatch(merkle_root_bytes, ipfs_cid).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logging.info(f"Logged batch to blockchain. Tx: {receipt.transactionHash.hex()}")
        return receipt.transactionHash.hex()
    except Exception as e:
        logging.error(f"Blockchain transaction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Blockchain transaction failed: {e}")

def hash_leaf(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def hash_pair(a: str, b: str) -> str:
    return hashlib.sha256((a + b).encode()).hexdigest()

def build_merkle_root(leaves: List[str]) -> str:
    if not leaves: return ""
    if len(leaves) == 1: return leaves[0]
    if len(leaves) % 2 == 1: leaves.append(leaves[-1])
    paired = [hash_pair(leaves[i], leaves[i + 1]) for i in range(0, len(leaves), 2)]
    return build_merkle_root(paired)

# --- FastAPI App ---
app = FastAPI(title="Q-Shield+ Backend")
Base.metadata.create_all(bind=engine)

@app.on_event("startup")
async def startup_event():
    initialize_ml_models()
    mqtt_client.start_mqtt()

# --- Pydantic Models ---
class RegisterRequest(BaseModel):
    device_name: str

class TelemetryIn(BaseModel):
    did: str
    current: float
    pressure: float
    temperature: float
    thermocouple: float
    voltage: float
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    timestamp: Optional[str] = None

# --- Geo-fence utility ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- API Endpoints ---
@app.post("/register_device")
def register_device(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(Device).count() >= MAX_DEVICES:
        raise HTTPException(status_code=403, detail="Device registration limit reached")
    existing = db.query(Device).filter(Device.device_name == req.device_name).first()
    if existing:
        token = create_jwt({"device_name": req.device_name, "did": existing.did})
        return {"did": existing.did, "jwt": token, "kyber_public_key": existing.kyber_pk}

    new_account = w3.eth.account.create()
    eth_address = new_account.address
    private_key = new_account.key.hex()
    did = f"did:ethr:{eth_address}"
    pk, sk = kyber_generate_keypair()
    device = Device(device_name=req.device_name, did=did, device_address=eth_address, kyber_pk=pk.hex(), kyber_sk=sk.hex())
    db.add(device)
    db.commit(); db.refresh(device)
    token = create_jwt({"device_name": req.device_name, "did": did})
    return {"did": did, "jwt": token, "kyber_public_key": pk.hex(), "device_private_key": private_key}

@app.post("/telemetry")
def ingest_telemetry(payload: TelemetryIn, db: Session = Depends(get_db)):
    input_logger.info(f"Received telemetry: {payload.dict()}")
    
    # Server-side validation (can be enabled as needed)
    # if not (1.8 <= payload.current <= 2.94): ...

    device = db.query(Device).filter(Device.did == payload.did).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    device.latitude = payload.latitude
    device.longitude = payload.longitude
    db.commit()

    distance = haversine(payload.latitude, payload.longitude, GEOFENCE_CENTER["lat"], GEOFENCE_CENTER["lon"])
    geo_alert = distance > GEOFENCE_RADIUS
    if geo_alert:
        debug_logger.warning(f"Geofence ALERT for {payload.did} | Distance: {distance:.2f} meters")

    timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()
    telemetry_for_ml = {k: v for k, v in payload.dict().items() if k in {"current", "pressure", "temperature", "thermocouple", "voltage"}}
    telemetry_for_ml["timestamp"] = timestamp
    
    add_to_telemetry_buffer(payload.did, telemetry_for_ml)
    ml_results = run_ml_inference(telemetry_for_ml, payload.did)

    telemetry_record = payload.dict()
    telemetry_record.update({
        "timestamp": timestamp,
        "geo_fence_alert": geo_alert,
        "distance_meters": distance,
        "ml_inference": ml_results
    })

    output_logger.info(f"Processed telemetry: {json.dumps(telemetry_record)}")
    MERKLE_BATCH.append(json.dumps(telemetry_record))
    
    return {"status": "ok", "telemetry": telemetry_record}

@app.post("/log_merkle_batch")
def log_merkle_batch(token=Depends(verify_jwt)):
    if not MERKLE_BATCH:
        raise HTTPException(status_code=400, detail="No telemetry to log in Merkle batch")
    leaves = [hash_leaf(entry) for entry in MERKLE_BATCH]
    merkle_root = build_merkle_root(leaves)
    batch_json = json.dumps(MERKLE_BATCH)
    files = {"file": ("batch_logs.json", batch_json)}
    try:
        resp = requests.post("http://127.0.0.1:5001/api/v0/add", files=files, timeout=10)
        resp.raise_for_status()
        batch_cid = resp.json()["Hash"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPFS batch upload failed: {e}")

    tx_hash = log_event_to_blockchain(merkle_root=merkle_root, ipfs_cid=batch_cid)
    MERKLE_BATCH.clear()
    return {"message": "Merkle batch logged", "batch_ipfs_cid": batch_cid, "merkle_root": merkle_root, "transaction_hash": tx_hash}

@app.get("/audit_log")
def get_audit_log():
    try:
        event_filter = contract.events.EventLogged.create_filter(fromBlock=0)
        events = event_filter.get_all_entries()
        log_entries = []
        for event in reversed(events):
            record_id = event['args']['recordId']
            record = contract.functions.eventRecords(record_id).call()
            log_entries.append({
                "recordId": record_id,
                "timestamp": datetime.fromtimestamp(record[0], tz=timezone.utc).isoformat(),
                "merkle_root": record[1].hex(),
                "ipfs_cid": record[2],
                "transaction_hash": event['transactionHash'].hex()
            })
        return {"status": "ok", "logs": log_entries}
    except Exception as e:
        logging.error(f"Failed to fetch audit log: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit log: {e}")

@app.get("/ml_status")
def get_ml_status():
    return {
        "edge_model_loaded": edge_detector is not None and edge_detector.model is not None,
        "cloud_model_loaded": cloud_detector is not None and cloud_detector.model is not None,
        "edge_model_available": EDGE_MODEL_AVAILABLE,
        "cloud_model_available": CLOUD_MODEL_AVAILABLE,
        "telemetry_buffers_count": len(telemetry_buffers)
    }

@app.get("/device/{device_id}/telemetry_history")
def get_device_telemetry_history(device_id: str):
    with buffer_lock:
        return {"device_id": device_id, "history": list(telemetry_buffers.get(device_id, []))}

@app.post("/analyze_telemetry")
def analyze_telemetry_manually(payload: TelemetryIn):
    telemetry_data = {k: v for k, v in payload.dict().items() if k in {"current", "pressure", "temperature", "thermocouple", "voltage"}}
    results = run_ml_inference(telemetry_data, payload.did)
    return {"telemetry": telemetry_data, "analysis": results}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)