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

# ♡ Config 
MAX_DEVICES = 100
TELEMETRY_INTERVAL_SECONDS = 1

# ♡ Logging Setup
os.makedirs("logs", exist_ok=True)

# ♡ Root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)

# ♡ Input / Output / Debug loggers
def create_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

input_logger = create_logger("input_logger", "logs/input_telemetry.log")
output_logger = create_logger("output_logger", "logs/output_telemetry.log")
debug_logger = create_logger("debug_logger", "logs/debug.log")

debug_logger.info("Debug logger initialized (geo-fence alerts will appear here)")

# ♡ FastAPI 
app = FastAPI(title="Q-Shield+ Backend")
Base.metadata.create_all(bind=engine)
mqtt_client.start_mqtt() 

# ♡ Models 
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

# ♡ Geo-fence utility 
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi, delta_lambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ♡ Device Registration
@app.post("/register_device")
def register_device(req: RegisterRequest, db: Session = Depends(get_db)):
    total_devices = db.query(Device).count()
    if total_devices >= MAX_DEVICES:
        raise HTTPException(status_code=403, detail="Device registration limit reached")

    existing = db.query(Device).filter(Device.device_name == req.device_name).first()
    if existing:
        token = create_jwt({"device_name": req.device_name, "did": existing.did})
        return {"did": existing.did, "jwt": token, "kyber_public_key": existing.kyber_pk}

    did = str(uuid.uuid4())
    pk, sk = kyber_generate_keypair()
    device = Device(device_name=req.device_name, did=did, kyber_pk=pk.hex(), kyber_sk=sk.hex())
    db.add(device)
    db.commit()
    db.refresh(device)

    token = create_jwt({"device_name": req.device_name, "did": did})
    return {"did": did, "jwt": token, "kyber_public_key": pk.hex()}

# ♡ Telemetry Ingestion 
@app.post("/telemetry")
def ingest_telemetry(payload: TelemetryIn, db: Session = Depends(get_db)):
    # ♡ Input log 
    input_logger.info(f"Received telemetry: {payload.dict()}")
    for handler in input_logger.handlers:
        handler.flush()

    # ♡ Validate telemetry
    if not (1.8 <= payload.current <= 2.94):
        raise HTTPException(status_code=422, detail="Current out of range")
    if not (-0.274 <= payload.pressure <= 0.383):
        raise HTTPException(status_code=422, detail="Pressure out of range")
    if not (89.86 <= payload.temperature <= 90.29):
        raise HTTPException(status_code=422, detail="Temperature out of range")
    if not (26.765 <= payload.thermocouple <= 26.7875):
        raise HTTPException(status_code=422, detail="Thermocouple out of range")
    if not (219.45 <= payload.voltage <= 248.54):
        raise HTTPException(status_code=422, detail="Voltage out of range")

    # ♡ Get device from DB 
    device = db.query(Device).filter(Device.did == payload.did).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    device.latitude = payload.latitude
    device.longitude = payload.longitude
    db.commit()

    # ♡ Geo-fence check 
    distance = haversine(payload.latitude, payload.longitude,
                         GEOFENCE_CENTER["lat"], GEOFENCE_CENTER["lon"])
    geo_alert = distance > GEOFENCE_RADIUS

    if geo_alert:
        debug_logger.warning(f"Geofence ALERT for {payload.did} | Distance: {distance:.2f} meters")
        for handler in debug_logger.handlers:
            handler.flush()

    # ♡ Prepare telemetry record 
    timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()
    telemetry_record = {
        "did": payload.did,
        "current": payload.current,
        "pressure": payload.pressure,
        "temperature": payload.temperature,
        "thermocouple": payload.thermocouple,
        "voltage": payload.voltage,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "altitude": payload.altitude,
        "timestamp": timestamp,
        "geo_fence_alert": geo_alert,
        "distance_meters": distance,
    }

    # ♡ Output log 
    output_logger.info(f"Processed telemetry: {telemetry_record}")
    for handler in output_logger.handlers:
        handler.flush()

    return {"status": "ok", "telemetry": telemetry_record}

# ♡ Run Server 
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
