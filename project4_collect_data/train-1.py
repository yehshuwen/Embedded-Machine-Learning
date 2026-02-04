import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from model import build_model

# Setting
data_dir = 'data'
classes = ['sitting', 'supine', 'prone', 'side', 'unknown']
num_classes = len(classes)

# Columns for each sensor
sensor_cols = {
    'accel': ['ax', 'ay', 'az'],
    'gyro':  ['gx', 'gy', 'gz'],
    'mag':   ['mx', 'my', 'mz']
}

# Map label
lebel_map = {
    'sitting': 0,
    'supine':  1,
    'prone':   2,
    'side':    3,
    'unknown': 4
}

input_channels   = 3          # x, y, z
sample_rate_hz   = 50         # collecting data sampling rate
window_seconds   = 2          # window length in seconds
window_size      = int(sample_rate_hz * window_seconds)  # samples per window
step             = int(window_size * 0.5)  # 50% overlap

model_save_path  = "posture_model.keras"
tflite_model_path = "model.tflite"
header_file_path  = "model.h"

# Load CSV files and build sensor-specific window datasets
print("Loading data...")
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

# We keep separate lists for each sensor type
X_lists = {'accel': [], 'gyro': [], 'mag': []}
y_lists = {'accel': [], 'gyro': [], 'mag': []}

for file in all_files:
    file_name = os.path.basename(file)
    label_name = file_name.split('_')[0]

    if label_name in lebel_map:
        label_id = lebel_map[label_name]
        df = pd.read_csv(file)

        # For each sensor create sliding windows of x,y,z
        for sensor_name, cols in sensor_cols.items():
            if not all(col in df.columns for col in cols):
                continue

            sensor_data = df[cols].values
            for i in range(0, len(sensor_data) - window_size, step):
                window = sensor_data[i: i + window_size]
                X_lists[sensor_name].append(window)
                y_lists[sensor_name].append(label_id)

# Normalize each sensor and merge
X_scaled_list = []
y_scaled_list = []

print("\n" + "=" * 40)
print(" Computing normalization constants for Arduino code...")
print("=" * 40)

for sensor_name in sensor_cols.keys():
    X_sensor = np.array(X_lists[sensor_name])
    y_sensor = np.array(y_lists[sensor_name])

    if len(X_sensor) == 0:
        continue

    # 1. Flatten windows from shape (num_windows, window_size, 3) -> (N, 3)
    X_flat = X_sensor.reshape(-1, 3)

    # 2. Create a new StandardScaler for this sensor only
    scaler = StandardScaler()

    # 3. Fit and transform only on this sensor's data
    X_flat_scaled = scaler.fit_transform(X_flat)

    # 4. Restore original window shape
    X_scaled = X_flat_scaled.reshape(X_sensor.shape)

    # 5. Print per-sensor normalization constants in C-style for Arduino
    print(f"// Constants for {sensor_name.upper()}")
    print(
        f"const float NORM_MEAN_{sensor_name.upper()}[] = "
        f"{{ {scaler.mean_[0]:.6f}f, {scaler.mean_[1]:.6f}f, {scaler.mean_[2]:.6f}f }};"
    )
    print(
        f"const float NORM_SCALE_{sensor_name.upper()}[] = "
        f"{{ {scaler.scale_[0]:.6f}f, {scaler.scale_[1]:.6f}f, {scaler.scale_[2]:.6f}f }};"
    )
    print("-" * 20)

    # 6. Add normalized data and labels to the combined list
    X_scaled_list.append(X_scaled)
    y_scaled_list.append(y_sensor)

print("=" * 40 + "\n")

# Merge normalized data into one dataset

X_scaled = np.concatenate(X_scaled_list)
y = np.concatenate(y_scaled_list)  # 1D integer labels

print(f"End data loading. Built {X_scaled.shape[0]} windows")
print(f"X shape: {X_scaled.shape}")
print(f"y shape: {y.shape}")

# One-hot encode labels
y_one_hot = to_categorical(y, num_classes)  # shape: (N, num_classes)

# Train/val/test split: 80% train, 10% val, 10% test
X_train, X_non_train, y_train, y_non_train = train_test_split(
    X_scaled, y_one_hot, test_size=0.2, random_state=42, stratify=y
)

# For second split we need 1D labels for stratify argument
X_test, X_validate, y_test, y_validate = train_test_split(
    X_non_train, y_non_train, test_size=0.5,
    random_state=42, stratify=np.argmax(y_non_train, axis=1)
)

print(f"Training set:   {X_train.shape}")
print(f"Validation set: {X_validate.shape}")
print(f"Testing set:    {X_test.shape}")

# Build and train the model
model_input_shape = (window_size, input_channels)
model = build_model(
    hidden_activation='relu',
    input_shape=model_input_shape,
    num_classes=num_classes
)
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_validate, y_validate),
    callbacks=[early_stopping, checkpoint],
    verbose=2
)

# Evaluate model (testting set)
print("Evaluate model on test set...")
best_model = tf.keras.models.load_model(model_save_path)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Testing set Loss:     {test_loss:.4f}")
print(f"Testing set Accuracy: {test_accuracy:.4f}")
print(f"Best model saved to:  {model_save_path}")


# Plot training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')

# Convert best Keras model to quantized INT8 TFLite model and C header
print(f"Converting model to TFLite: {tflite_model_path}")

def representative_dataset():
    sample_count = min(200, X_train.shape[0])
    for i in range(sample_count):
        yield [X_train[i:i + 1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite (int8) size: {os.path.getsize(tflite_model_path)} bytes")

# Generate model.h using xxd -i
print(f"Generating {header_file_path} ...")
try:
    os_command = f"xxd -i {tflite_model_path} > {header_file_path}"
    if os.name == 'nt':
        os_command = f"xxd -i {tflite_model_path} > {header_file_path}"

    os.system(os_command)

    if os.path.exists(header_file_path) and os.path.getsize(header_file_path) > 0:
        print(f"{header_file_path} built successfully!")
    else:
        raise Exception(
            f"xxd command failed or produced an empty file. (Command: {os_command})"
        )

except Exception as e:
    print(f"Error generating {header_file_path}: {e}")
    print("Please ensure 'xxd' is installed and available in your system PATH.")