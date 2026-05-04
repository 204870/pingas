"""
Prosody model trainer for pitch (F0) and intensity delta prediction.

Trains an LSTM to predict F0 *delta* (change from previous phone in cents)
and intensity delta from phone features and prosodic annotations.

The model learns: "when a phone is marked prominent/boundary, how should
F0 change?" This maps directly to synthesis: apply predicted delta to
introduce pitch accents (*) and boundary tones (. ? !) in the output.

Usage:
    python trainer.py [--burnc PATH] [--epochs N] [--batch-size N] [--output PATH]
"""

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Bidirectional, Masking, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from pin.burnc_parser import (
    parse_corpus,
    utterances_to_sequences,
    hz_to_cents,
    cents_to_hz,
    F0_MIN_HZ,
    F0_MAX_HZ
)


def build_model(
    sequence_length: int,
    n_features: int,
    n_targets: int = 2,
    lstm_units: int = 128,
    lstm_layers: int = 3,
    dense_units: list[int] = [128, 64, 32],
    bidirectional: bool = True
) -> Model:
    """
    Build LSTM model for prosody prediction.

    Args:
        sequence_length: Max sequence length (for input shape)
        n_features: Number of input features per timestep
        n_targets: Number of output targets (default 2: f0_delta, intensity_delta)
        lstm_units: Number of LSTM units per layer
        lstm_layers: Number of LSTM layers
        dense_units: List of dense layer sizes
        bidirectional: Whether to use bidirectional LSTM

    Returns:
        Keras model
    """
    model = Sequential()

    # Masking layer to handle padded sequences (mask value = 0)
    model.add(Masking(mask_value=0.0, input_shape=(sequence_length, n_features)))

    # LSTM layers
    for i in range(lstm_layers):
        lstm = LSTM(lstm_units, return_sequences=True)
        if bidirectional:
            model.add(Bidirectional(lstm))
        else:
            model.add(lstm)

    # Dense layers for each timestep
    for units in dense_units:
        model.add(TimeDistributed(Dense(units, activation='relu')))

    # Output layer
    model.add(TimeDistributed(Dense(n_targets, activation='linear')))

    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    masks_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    masks_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lstm_units: int = 64,
    bidirectional: bool = True,
    patience: int = 10,
    model_path: Path = None
) -> tuple[Model, dict]:
    """
    Train the prosody model.

    Returns:
        Trained model and training history
    """
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    n_targets = y_train.shape[2]

    model = build_model(
        sequence_length=seq_len,
        n_features=n_features,
        n_targets=n_targets,
        lstm_units=lstm_units,
        bidirectional=bidirectional
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print(model.summary())

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if model_path:
        callbacks.append(ModelCheckpoint(
            str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))

    # Apply masks by zeroing out padded targets
    # (The Masking layer handles input, but we need to handle output too)
    y_train_masked = y_train * masks_train[:, :, np.newaxis]
    y_val_masked = y_val * masks_val[:, :, np.newaxis]

    # Use sample weights to ignore padded timesteps in loss
    history = model.fit(
        X_train, y_train_masked,
        validation_data=(X_val, y_val_masked),
        sample_weight=masks_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history.history


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    masks_test: np.ndarray,
    meta: dict
) -> dict:
    """
    Evaluate model performance.

    Returns metrics for F0 delta (in cents) and intensity delta.
    """
    y_pred = model.predict(X_test, verbose=0)

    # Flatten and filter by mask
    mask_flat = masks_test.flatten().astype(bool)
    y_true_flat = y_test.reshape(-1, 2)[mask_flat]
    y_pred_flat = y_pred.reshape(-1, 2)[mask_flat]

    # F0 metrics (in cents)
    f0_true_cents = y_true_flat[:, 0]
    f0_pred_cents = y_pred_flat[:, 0]
    f0_mae_cents = np.mean(np.abs(f0_true_cents - f0_pred_cents))
    f0_rmse_cents = np.sqrt(np.mean((f0_true_cents - f0_pred_cents) ** 2))

    # Intensity metrics
    int_true = y_true_flat[:, 1]
    int_pred = y_pred_flat[:, 1]
    int_mae = np.mean(np.abs(int_true - int_pred))

    # Convert cents error to approximate semitones for interpretability
    f0_mae_semitones = f0_mae_cents / 100.0

    metrics = {
        'f0_delta_mae_cents': float(f0_mae_cents),
        'f0_delta_rmse_cents': float(f0_rmse_cents),
        'f0_delta_mae_semitones': float(f0_mae_semitones),
        'intensity_delta_mae': float(int_mae),
        'n_samples': int(np.sum(mask_flat))
    }

    return metrics


def save_model_with_meta(
    model: Model,
    meta: dict,
    output_path: Path
) -> None:
    """Save model (.h5) and metadata (.json) for inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save Keras model as .h5
    model.save(output_path)

    # Save metadata alongside the model
    meta_path = output_path.with_suffix('.json')
    meta_serializable = {
        'f0_delta_stats': meta.get('f0_delta_stats', {}),
        'intensity_delta_stats': meta.get('intensity_delta_stats', {}),
        'feature_names': meta['feature_names'],
        'target_names': meta['target_names'],
        'sequence_length': meta['sequence_length'],
        'f0_unit': meta['f0_unit'],
        'f0_bounds_hz': meta['f0_bounds_hz']
    }

    with open(meta_path, 'w') as f:
        json.dump(meta_serializable, f, indent=2)

    print(f"Model saved to {output_path}")
    print(f"Metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Train prosody prediction model')
    parser.add_argument('--burnc', type=Path, default=Path('../bu_radio/data'),
                        help='Path to BURNC data directory')
    parser.add_argument('--output', type=Path, default=Path('pin/prosody_model.h5'),
                        help='Output path for trained model (.h5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lstm-units', type=int, default=64,
                        help='Number of LSTM units')
    parser.add_argument('--seq-length', type=int, default=200,
                        help='Max sequence length (longer sequences truncated)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Fraction of data for testing')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Fraction of training data for validation')
    args = parser.parse_args()

    # Parse corpus
    print(f"Loading BURNC corpus from {args.burnc}...")
    utterances = parse_corpus(args.burnc, verbose=True)

    if not utterances:
        print("No utterances found. Check the BURNC path.")
        return

    # Convert to sequences
    print(f"\nConverting to sequences (max length {args.seq_length})...")
    X, y, masks, meta = utterances_to_sequences(utterances, sequence_length=args.seq_length)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Valid phones: {int(masks.sum())}")

    # F0 delta statistics in cents
    y_f0_delta = y[:, :, 0][masks.astype(bool)]
    print(f"\nF0 delta distribution (cents, phone-to-phone change):")
    print(f"  Range: [{y_f0_delta.min():.0f}, {y_f0_delta.max():.0f}] cents")
    print(f"  Std: {y_f0_delta.std():.0f} cents ({y_f0_delta.std()/100:.1f} semitones)")
    print(f"  Mean: {y_f0_delta.mean():.1f} cents (should be near 0)")

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test, m_trainval, m_test = train_test_split(
        X, y, masks, test_size=args.test_size, random_state=42
    )

    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_trainval, y_trainval, m_trainval,
        test_size=args.val_size / (1 - args.test_size),
        random_state=42
    )

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} utterances")
    print(f"  Val:   {len(X_val)} utterances")
    print(f"  Test:  {len(X_test)} utterances")

    # Train
    print(f"\nTraining model...")
    model, history = train_model(
        X_train, y_train, m_train,
        X_val, y_val, m_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        patience=args.patience
    )

    # Evaluate
    print(f"\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, m_test, meta)

    print(f"\nTest metrics (delta prediction):")
    print(f"  F0 delta MAE:  {metrics['f0_delta_mae_cents']:.1f} cents ({metrics['f0_delta_mae_semitones']:.2f} semitones)")
    print(f"  F0 delta RMSE: {metrics['f0_delta_rmse_cents']:.1f} cents")
    print(f"  Intensity delta MAE: {metrics['intensity_delta_mae']:.3f} (normalized)")

    # Save
    meta['metrics'] = metrics
    meta['training_history'] = {
        'final_train_loss': history['loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'epochs_trained': len(history['loss'])
    }

    save_model_with_meta(model, meta, args.output)

    # Also save metrics separately for reference
    metrics_path = args.output.with_name('training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_metrics': metrics,
            'training_history': meta['training_history']
        }, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
