import os
import joblib
import pandas as pd
import logging
import numpy as np
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "isolation_forest_model.pkl")

FEATURE_COLS = ['current', 'pressure', 'temperature', 'thermocouple', 'voltage']

class IsolationForestDetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the Isolation Forest model from the specified path."""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                logging.info(f"Successfully loaded Isolation Forest model from {MODEL_PATH}.")
            else:
                logging.error(f"Isolation Forest model not found at {MODEL_PATH}.")
        except Exception as e:
            logging.error(f"An error occurred during model loading: {e}")
            self.model = None

    def _to_native(self, value):
        """Convert NumPy types to native Python types."""
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def get_anomaly_score(self, telemetry_dict: Dict) -> Dict:
        """Performs inference on a single telemetry data point."""
        if self.model is None:
            logging.error("Model not loaded. Cannot perform inference.")
            return {'is_anomaly': False, 'score': 0.0}

        try:
            df = pd.DataFrame([telemetry_dict])[FEATURE_COLS]
        except KeyError as e:
            logging.error(f"A required feature is missing from the input data: {e}")
            return {'is_anomaly': False, 'score': 0.0, 'error': 'Missing feature'}

        prediction = self.model.predict(df)[0]
        raw_score = self.model.score_samples(df)[0]
        is_anomaly = (prediction == -1)

        normalized_score = min(1.0, max(0.0, -0.5 * raw_score))

        # Convert NumPy types to native Python types
        result = {
            'is_anomaly': is_anomaly,
            'score': normalized_score,
            'raw_score': raw_score
        }
        return {k: self._to_native(v) for k, v in result.items()}

# --- Main execution block for testing ---
if __name__ == "__main__":
    detector = IsolationForestDetector()

    if detector.model:
        logging.info("--- Testing Edge Isolation Forest Anomaly Detection ---")

        normal_data = {'current': 2.1, 'pressure': 0.15, 'temperature': 90.1, 'thermocouple': 26.77, 'voltage': 240.0}
        anomaly_data = {'current': 3.5, 'pressure': 0.8, 'temperature': 95.0, 'thermocouple': 30.0, 'voltage': 255.0}

        print(f"\n[EdgeIF] Normal data point test: {detector.get_anomaly_score(normal_data)}")
        print(f"[EdgeIF] Anomalous data point test: {detector.get_anomaly_score(anomaly_data)}")
