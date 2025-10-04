import requests
import time
import random
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# ♡ Configurable server URL 
SERVER_URL = "http://127.0.0.1:8000"  

BASE_LAT = 12.9716
BASE_LON = 77.5946

DEVICE_CACHE_FILE = Path("device_cache.json")

# ♡ Utility: clamp values to server limits 
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# ♡ Save registered device locally 
def save_device(device_name, did, jwt):
    DEVICE_CACHE_FILE.write_text(json.dumps({
        "device_name": device_name,
        "did": did,
        "jwt": jwt
    }))

# ♡ Load registered device if exists 
def load_device():
    if DEVICE_CACHE_FILE.exists():
        data = json.loads(DEVICE_CACHE_FILE.read_text())
        return data["device_name"], data["did"], data["jwt"]
    return None, None, None

# ♡ Device registration
def register_device(name: str):
    cached_name, cached_did, cached_jwt = load_device()
    if cached_name == name:
        print(f"[*] Using cached device '{name}' with ID: {cached_did}")
        return cached_did, cached_jwt

    print(f"[*] Registering device '{name}' with server...")
    resp = requests.post(
        f"{SERVER_URL}/register_device",
        json={"device_name": name}
    )
    resp.raise_for_status()
    data = resp.json()
    did = data["did"]
    jwt = data["jwt"]
    save_device(name, did, jwt)
    print(f"[+] Device '{name}' registered with ID: {did}")
    return did, jwt

# ♡ Send telemetry safely 
def send_telemetry(did: str, jwt: str):
    # ♡ Determine if this telemetry is an anomaly
    is_anomaly = False  

    # ♡ Normal ranges
    current_range = (1.81019, 2.93381)
    pressure_range = (-0.273216, 0.382638)
    temperature_range = (89.8666, 90.2836)
    thermocouple_range = (26.7658, 26.787)
    voltage_range = (219.454, 248.533)

    # ♡ Widened anomaly ranges
    current_anomaly = (2.94, 3.2)
    pressure_anomaly = (0.383, 0.5)
    temperature_anomaly = (90.284, 91)
    thermocouple_anomaly = (26.788, 26.8)
    voltage_anomaly = (248.534, 250)

    # ♡ Choose ranges based on anomaly
    current = random.uniform(*current_anomaly if is_anomaly else current_range)
    pressure = random.uniform(*pressure_anomaly if is_anomaly else pressure_range)
    temperature = random.uniform(*temperature_anomaly if is_anomaly else temperature_range)
    thermocouple = random.uniform(*thermocouple_anomaly if is_anomaly else thermocouple_range)
    voltage = random.uniform(*voltage_anomaly if is_anomaly else voltage_range)

    payload = {
        "did": did,
        "current": current,
        "pressure": pressure,
        "temperature": temperature,
        "thermocouple": thermocouple,
        "voltage": voltage,
        "latitude": BASE_LAT + random.uniform(-0.0005, 0.0005),
        "longitude": BASE_LON + random.uniform(-0.0005, 0.0005),
        "altitude": 920.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulated_anomaly": is_anomaly  # optional field for testing
    }

    headers = {"Authorization": f"Bearer {jwt}"}
    resp = requests.post(f"{SERVER_URL}/telemetry", json=payload, headers=headers)
    resp.raise_for_status()
    print(f"[+] Telemetry sent for device {did} ({'ANOMALY' if is_anomaly else 'normal'}): {resp.json()}")

# ♡ Main: simulate device continuously 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python device_sim.py <device_name> [server_url]")
        sys.exit(1)

    device_name = sys.argv[1]
    if len(sys.argv) > 2:
        SERVER_URL = sys.argv[2]

    device_id, jwt = register_device(device_name)

    while True:
        try:
            send_telemetry(device_id, jwt)
        except requests.exceptions.HTTPError as e:
            print(f"[!] Error sending telemetry: {e}")
        except Exception as ex:
            print(f"[!] Unexpected error: {ex}")
        time.sleep(1)
