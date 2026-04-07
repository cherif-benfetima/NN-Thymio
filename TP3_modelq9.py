import glob
import os

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


dataset_candidates = glob.glob("line_follower/line_follow_obstacles_dataset_*.h5")
if not dataset_candidates:
    raise FileNotFoundError(
        "No transformed Q9 dataset found. Run transform_q9_dataset.py first."
    )

file_path = max(dataset_candidates, key=os.path.getmtime)
print("Dataset loaded:", file_path)

with h5py.File(file_path, "r") as f:
    X = np.array(f["X"], dtype=np.float32)
    y = np.array(f["y"], dtype=np.float32)
    feature_order_attr = f.attrs.get("feature_order", "")

if isinstance(feature_order_attr, bytes):
    feature_order_attr = feature_order_attr.decode("utf-8")

feature_order = [x.strip() for x in str(feature_order_attr).split(",") if x.strip()]
if not feature_order:
    feature_order = [
        "ground_left_norm",
        "ground_right_norm",
        "prox_h0_norm",
        "prox_h1_norm",
        "prox_h2_norm",
        "prox_h3_norm",
        "prox_h4_norm",
    ]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature order:", feature_order)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(24, 16),
    activation="tanh",
    solver="adam",
    max_iter=2500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=60,
    validation_fraction=0.2,
)

mlp.fit(X_train_scaled, y_train)


def evaluate(X_eval, y_eval, name):
    pred = mlp.predict(X_eval)
    mse = mean_squared_error(y_eval, pred)
    r2 = mlp.score(X_eval, y_eval)
    print(f"{name} -> MSE: {mse:.6f} | R2: {r2:.4f}")


evaluate(X_train_scaled, y_train, "Train")
evaluate(X_test_scaled, y_test, "Test")

print("Final loss:", mlp.loss_)
print("Iterations:", mlp.n_iter_)

plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Q9 training loss")
plt.tight_layout()
plt.show()

bundle = {
    "model": mlp,
    "scaler_x": scaler_x,
    "normalization": {
        "ground_black": 0.0,
        "ground_white": 1000.0,
        "prox_min": 0.0,
        "prox_max": 4500.0,
        "motor_max": 500.0,
    },
    "feature_order": feature_order,
}

joblib.dump(bundle, "model_mlp_q9.pkl")
print("Model saved: model_mlp_q9.pkl")
