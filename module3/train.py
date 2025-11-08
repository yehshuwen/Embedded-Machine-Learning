import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from model import build_model

# Setting
data_dir = 'data'
num_classes = 5
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
lebal_map = {
    'sitting': 0,
    'supine': 1,
    'prone': 2,
    'side': 3,
    'unknown': 4
}

noise_level = 0.05

# Combine all data set
print("loading data...")
all_files = glob.glob(os.path.join(data_dir, "*.csv"))
X_list = []
y_list = []

for file in all_files:
    file_name = os.path.basename(file)
    label_name = file_name.split('_')[0]

    if label_name in lebal_map:
        label_id = lebal_map[label_name]

        df = pd.read_csv(file)
        data = df[features].values

        noise  = np.random.normal(0, noise_level, data.shape)
        data_with_noise = data + noise

        X_list.append(data)
        y_list.append(np.full(data.shape[0], label_id))

        X_list.append(data_with_noise)
        y_list.append(np.full(data_with_noise.shape[0], label_id))

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
print(f"Total sample data: {X.shape[0]}")

# normolized
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# add label, transe y to one-hot coding e.g., 3 -> [0, 0, 1, 0, 0]
y_one_hot = to_categorical(y, num_classes=num_classes)

# split data, 70% for trainning and 15% for validation and 15% for testing
X_train, X_non_train, y_train, y_non_train = train_test_split(
    X_scaled, y_one_hot, test_size=0.3, random_state=42, stratify=y_one_hot
)
X_test, X_validate, y_test, y_validate = train_test_split(
    X_non_train, y_non_train, test_size=0.5, random_state=42, stratify=y_non_train
)

print(f"Training set: {X_train.shape}, Validation set: {X_validate.shape}, Testing set: {X_test.shape}")

activations = ['relu', 'sigmoid', 'tanh']
histories = {}
models = {}

# invoid Overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for act in activations:
    print(f"\n--- Trainning...: {act} ---")
    model = build_model(hidden_activation=act, num_features=X_train.shape[1], num_classes=num_classes)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_validate, y_validate),
        callbacks=[early_stopping],
        verbose=1
    )
    # Evaluate 
    print(f"\n--- Evaluate {act} model in testing set ---")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Testing set Loss: {test_loss:.4f}")
    print(f"Testing set Accuracy: {test_accuracy:.4f}")

    histories[act] = history
    models[act] = model
    

# Plotting result
print("\nPlotting...")
plt.figure(figsize=(12, 8))
for act, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{act} - val_accuracy')
plt.title('model varify compared')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('activation_comparison.png')
print("Saved as activation_comparison.png")

best_activation = None
best_val_accuracy = 0

for act, history in histories.items():
    current_max_val_acc = max(history.history['val_accuracy'])
    print(f"model {act} best acc: {current_max_val_acc:.4f}")
    if current_max_val_acc > best_val_accuracy:
        best_val_accuracy = current_max_val_acc
        best_activation = act

print(f"\n--- best model: {best_activation} (acc: {best_val_accuracy:.4f}) ---")
best_model = models[best_activation]
model_save_path = "my_best_posture_model.keras"
print(f"\n---Saving model to {model_save_path} ---")
best_model.save(model_save_path)
print("Saved")