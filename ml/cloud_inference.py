import os
import joblib
import pandas as pd
import logging
import numpy as np
from typing import Dict, List
from tensorflow.keras.models import load_model

# Suppress tensorflow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

FEATURE_COLS = ['current', 'pressure', 'temperature', 'thermocouple', 'voltage']
SEQUENCE_LENGTH = 30
RECONSTRUCTION_THRESHOLD = 0.05  # empirically set or tune

class LSTMAnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """Loads the Keras model and the scaler."""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH, compile=False)
                logging.info(f"Successfully loaded Keras model from {MODEL_PATH}.")
            else:
                logging.error(f"Keras model not found at {MODEL_PATH}.")

            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logging.info(f"Successfully loaded Scaler from {SCALER_PATH}.")
            else:
                logging.error(f"Scaler not found at {SCALER_PATH}.")

        except Exception as e:
            logging.error(f"Error loading models: {e}")
            self.model = None
            self.scaler = None

    def classify_severity(self, score: float) -> str:
        if score < 0.25:
            return "low"
        elif score < 0.6:
            return "medium"
        else:
            return "high"

    def get_anomaly_score(self, telemetry_sequence: List[Dict]) -> Dict:
        if self.model is None or self.scaler is None:
            logging.error("Model or scaler not loaded.")
            return {'score': 0.0, 'is_anomaly': False, 'severity': 'unloaded'}

        if len(telemetry_sequence) < SEQUENCE_LENGTH:
            logging.warning(f"Input sequence length {len(telemetry_sequence)} less than required {SEQUENCE_LENGTH}.")
            return {'score': 0.0, 'is_anomaly': False, 'severity': 'insufficient_data'}

        try:
            df = pd.DataFrame(telemetry_sequence[-SEQUENCE_LENGTH:])[FEATURE_COLS]
            data_scaled = self.scaler.transform(df.values)
            sequence = np.expand_dims(data_scaled, axis=0).astype(np.float32)
        except KeyError as e:
            logging.error(f"Missing feature in input data: {e}")
            return {'score': 0.0, 'is_anomaly': False, 'severity': 'error'}
        except Exception as e:
            logging.error(f"Failed to prepare data: {e}")
            return {'score': 0.0, 'is_anomaly': False, 'severity': 'error'}

        reconstructed = self.model.predict(sequence, verbose=0)

        if hasattr(reconstructed, 'numpy'):
            reconstructed = reconstructed.numpy()
            
        if reconstructed.shape != sequence.shape:
             reconstructed = np.array(reconstructed)

        # --- THIS IS THE CORRECTED LINE ---
        mse = np.mean(np.power(sequence - reconstructed, 2))

        is_anomaly = mse > RECONSTRUCTION_THRESHOLD
        normalized_score = min(1.0, mse / (RECONSTRUCTION_THRESHOLD * 2))

        return {
            'score': float(normalized_score),
            'is_anomaly': bool(is_anomaly),
            'severity': self.classify_severity(normalized_score) if is_anomaly else "none",
            'raw_mse': float(mse)
        }

if __name__ == "__main__":
    detector = LSTMAnomalyDetector()

    if detector.model and detector.scaler:
        print("--- Testing Cloud LSTM Anomaly Detection ---")
        normal_data = {'current': 2.1, 'pressure': 0.15, 'temperature': 90.1, 'thermocouple': 26.77, 'voltage': 240.0}
        anomaly_data = {'current': 3.5, 'pressure': 0.8, 'temperature': 95.0, 'thermocouple': 30.0, 'voltage': 255.0}

        normal_sequence = [normal_data] * SEQUENCE_LENGTH
        anomaly_sequence = [normal_data] * (SEQUENCE_LENGTH - 1) + [anomaly_data]

        print(f"[CloudLSTM] Normal sequence test: {detector.get_anomaly_score(normal_sequence)}")
        print(f"[CloudLSTM] Anomalous sequence test: {detector.get_anomaly_score(anomaly_sequence)}")