import paho.mqtt.client as mqtt
from security import decrypt_aes_gcm
from config import MQTT_BROKER, MQTT_PORT

# ♡ MQTT setup
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with code", rc)
    client.subscribe("devices/telemetry")

def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload}")

client.on_connect = on_connect
client.on_message = on_message

def start_mqtt():
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
