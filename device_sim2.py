import requests
import time
import random
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np # Import numpy for sine wave generation

# --- All server/device configuration is unchanged ---
SERVER_URL = "http://127.0.0.1:8000"

BASE_LAT = 12.9716
BASE_LON = 77.5946

DEVICE_CACHE_FILE = Path("device_cache.json")

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def save_device(device_name, did, jwt):
    DEVICE_CACHE_FILE.write_text(json.dumps({
        "device_name": device_name,
        "did": did,
        "jwt": jwt
    }))

def load_device():
    if DEVICE_CACHE_FILE.exists():
        data = json.loads(DEVICE_CACHE_FILE.read_text())
        return data["device_name"], data["did"], data["jwt"]
    return None, None, None

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

# In device_simulator2.py

def send_telemetry(did: str, jwt: str, time_step: int):
    """
    Generates and sends telemetry.
    - Normal data is a smooth sine wave.
    - Anomalous data is a random value from a wider, out-of-bounds range.
    """
    # Change this line back to generate random anomalies
    is_anomaly = random.random() < 0.05

    if is_anomaly:
        # --- GENERATE OBVIOUS ANOMALY ---
        current_anomaly = (2.94, 3.2)
        pressure_anomaly = (0.383, 0.5)
        temperature_anomaly = (90.284, 91)
        thermocouple_anomaly = (26.788, 26.8)
        voltage_anomaly = (248.534, 250)
        
        current = random.uniform(*current_anomaly)
        pressure = random.uniform(*pressure_anomaly)
        temperature = random.uniform(*temperature_anomaly)
        thermocouple = random.uniform(*thermocouple_anomaly)
        voltage = random.uniform(*voltage_anomaly)
        
    else:
        # --- GENERATE SMOOTH NORMAL DATA ---
        current = 2.37 + 0.5 * np.sin(0.1 * time_step)
        pressure = 0.05 + 0.3 * np.sin(0.05 * time_step + 1)
        temperature = 90.07 + 0.2 * np.sin(0.2 * time_step + 2)
        thermocouple = 26.776 + 0.01 * np.sin(0.15 * time_step + 3)
        voltage = 234 + 14 * np.sin(0.08 * time_step + 4)

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
    }
    
    # Put this logic back in to label the anomalies correctly
    if is_anomaly:
        payload["simulated_anomaly"] = True

    headers = {"Authorization": f"Bearer {jwt}"}
    resp = requests.post(f"{SERVER_URL}/telemetry", json=payload, headers=headers)
    resp.raise_for_status()
    print(f"[+] Telemetry sent for device {did} ({'ANOMALY' if is_anomaly else 'normal'}): {resp.json()}")


# --- The main execution block is unchanged ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python device_simulator2.py <device_name> [server_url]")
        sys.exit(1)

    device_name = sys.argv[1]
    if len(sys.argv) > 2:
        SERVER_URL = sys.argv[2]

    device_id, jwt = register_device(device_name)

    time_step = 0
    while True:
        try:
            send_telemetry(device_id, jwt, time_step)
            time_step += 1
        except requests.exceptions.HTTPError as e:
            print(f"[!] Error sending telemetry: {e}")
        except Exception as ex:
            print(f"[!] Unexpected error: {ex}")
        time.sleep(1)