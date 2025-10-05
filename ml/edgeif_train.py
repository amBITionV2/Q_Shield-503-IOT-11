import os
import argparse
import pandas as pd
import json
import re
import logging
from sklearn.ensemble import IsolationForest
import joblib

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature columns to use for training (lowercase to match logs) ---
FEATURE_COLS = ['current', 'pressure', 'temperature', 'thermocouple', 'voltage']

def parse_log_file(file_path):
    """
    Parses the output_telemetry.log file, extracts JSON data from each line,
    and filters for non-anomalous entries.
    """
    logging.info(f"Parsing log file from {file_path}...")
    records = []
    json_pattern = re.compile(r'\{.*\}')

    with open(file_path, 'r') as f:
        for line in f:
            match = json_pattern.search(line)
            if not match:
                continue

            try:
                json_str = match.group(0)
                data = json.loads(json_str)

                # A record is considered NORMAL for training if the 'simulated_anomaly' key
                # is NOT PRESENT in the dictionary.
                if 'simulated_anomaly' not in data:
                    if all(col in data for col in FEATURE_COLS):
                        records.append(data)

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping malformed or incomplete log line: {line.strip()}. Error: {e}")
    
    if not records:
        logging.error("No valid, non-anomalous data records were found in the log file.")
        raise ValueError("Could not parse any valid data from the log file.")
        
    logging.info(f"Successfully parsed {len(records)} valid, non-anomalous records.")
    return pd.DataFrame(records)

def load_and_preprocess_data(file_path):
    """Loads and preprocesses data from the specified log file."""
    try:
        df = parse_log_file(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        raise
    
    data = df[FEATURE_COLS]
    logging.info(f"Data shape for training: {data.shape}")
    
    # Handle any potential missing values
    data = data.ffill().bfill()
    
    return data

def train_isolation_forest(data):
    """Trains the Isolation Forest model on the provided data."""
    logging.info("Training Isolation Forest model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,  # Expectation of 1% anomalies in real data
        max_samples='auto',
        random_state=42,
        verbose=1,
        n_jobs=-1 # Use all available CPU cores
    )
    model.fit(data)
    logging.info("Training finished.")
    return model

def save_model(model, path):
    """Saves the trained model to a specified file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved to: {path}")

def main(args):
    """Main function to run the training pipeline."""
    data = load_and_preprocess_data(args.data_file)
    model = train_isolation_forest(data)
    save_model(model, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolation Forest Training Script")
    
    parser.add_argument('--data-file', type=str, default="logs/output_telemetry.log",
                        help='Path to the input training data log file.')
    parser.add_argument('--model-path', type=str, default="ml/saved_models/isolation_forest_model.pkl",
                        help='Path to save the trained model file.')
                        
    args = parser.parse_args()
    main(args)