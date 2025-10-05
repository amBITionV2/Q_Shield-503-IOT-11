import os
import argparse
import joblib
import logging
import numpy as np
import pandas as pd
import json
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

                # --- MODIFIED LOGIC ---
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
    logging.info(f"Loading data from log file: {file_path}...")
    try:
        df = parse_log_file(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        raise

    df = df[FEATURE_COLS]
    logging.info(f"Data shape for training: {df.shape}")
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df.values)
    
    logging.info("Data loading and preprocessing complete.")
    return normalized_data, scaler

def create_sequences(data, seq_length):
    """
    Convert 2D array data into overlapping sequences.
    """
    logging.info(f"Creating sequences with length {seq_length}...")
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    
    if not sequences:
        raise ValueError("Could not create any sequences. Check data length and sequence length.")
        
    return np.array(sequences)

def build_lstm_autoencoder(seq_length, num_features, encoding_dim=64):
    """
    Builds the LSTM Autoencoder model.
    (This is the new, improved version)
    """
    logging.info("Building LSTM autoencoder model with 'tanh' activation...")
    inputs = Input(shape=(seq_length, num_features))
    
    # Encoder
    encoded = LSTM(128, activation='tanh', return_sequences=True)(inputs)
    encoded = LSTM(encoding_dim, activation='tanh')(encoded)
    
    # Decoder
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(encoding_dim, activation='tanh', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='tanh', return_sequences=True)(decoded)
    
    # Output layer
    output = TimeDistributed(Dense(num_features, activation='sigmoid'))(decoded)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    model.summary(print_fn=logging.info)
    return model

def main(args):
    data, scaler = load_and_preprocess_data(args.data_file)
    sequences = create_sequences(data, args.seq_length)
    logging.info(f"Final sequences shape: {sequences.shape}")
    
    num_features = sequences.shape[2]
    model = build_lstm_autoencoder(args.seq_length, num_features)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    logging.info("Starting model training...")
    history = model.fit(
        sequences, sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    logging.info("Training finished.")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "lstm_autoencoder.h5")
    scaler_path = os.path.join(args.model_dir, "scaler.joblib")
    
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Autoencoder Training Script")
    
    parser.add_argument('--data-file', type=str, default="logs/output_telemetry.log",
                        help='Path to the input training data log file.')
    parser.add_argument('--model-dir', type=str, default="ml/saved_models/",
                        help='Directory to save the trained model and scaler.')
                        
    parser.add_argument('--seq-length', type=int, default=30,
                        help='Length of the input sequences for the LSTM.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training.')
                        
    args = parser.parse_args()
    main(args)