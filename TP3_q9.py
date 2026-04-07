import os
import time

import joblib
import numpy as np
from serial.tools import list_ports
from thymiodirect import Thymio


loaded = joblib.load("model_mlp_q9.pkl")
if isinstance(loaded, dict) and "model" in loaded:
    mlp = loaded["model"]
    scaler_x = loaded.get("scaler_x")
    norm_cfg = loaded.get("normalization", {})
    feature_order = loaded.get("feature_order", [])
else:
    mlp = loaded
    scaler_x = None
    norm_cfg = {}
    feature_order = []

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

GROUND_BLACK = float(norm_cfg.get("ground_black", 0.0))
GROUND_WHITE = float(norm_cfg.get("ground_white", 1000.0))
PROX_MIN = float(norm_cfg.get("prox_min", 0.0))
PROX_MAX = float(norm_cfg.get("prox_max", 4500.0))
MOTOR_TARGET_MAX = float(norm_cfg.get("motor_max", 500.0))
SMOOTHING_ALPHA = 0.25


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_ground(value):
    denom = max(1e-9, GROUND_WHITE - GROUND_BLACK)
    return clamp((value - GROUND_BLACK) / denom, 0.0, 1.0)


def normalize_prox(value):
    denom = max(1e-9, PROX_MAX - PROX_MIN)
    return clamp((value - PROX_MIN) / denom, 0.0, 1.0)


def denormalize_motor_speed(speed_norm):
    return int(clamp(speed_norm, -1.0, 1.0) * MOTOR_TARGET_MAX)


def set_leds(th, node_id, r, g, b):
    th[node_id]["leds.top"] = [r, g, b]


def on_comm_error(error):
    print("Communication error:", error)
    os._exit(1)


def get_feature(name, ground, prox_h):
    if name == "ground_left_norm":
        return normalize_ground(ground[0])
    if name == "ground_right_norm":
        return normalize_ground(ground[1])
    if name == "prox_h0_norm":
        return normalize_prox(prox_h[0])
    if name == "prox_h1_norm":
        return normalize_prox(prox_h[1])
    if name == "prox_h2_norm":
        return normalize_prox(prox_h[2])
    if name == "prox_h3_norm":
        return normalize_prox(prox_h[3])
    if name == "prox_h4_norm":
        return normalize_prox(prox_h[4])
    raise ValueError(f"Unknown feature: {name}")


done = False
left_cmd_smoothed = 0.0
right_cmd_smoothed = 0.0


def obs(node_id):
    global done, left_cmd_smoothed, right_cmd_smoothed
    if done:
        return

    ground = th[node_id]["prox.ground.reflected"]
    prox_h = th[node_id]["prox.horizontal"]

    sensors = [get_feature(name, ground, prox_h) for name in feature_order]
    features = np.array([sensors], dtype=np.float32)
    if scaler_x is not None:
        features = scaler_x.transform(features)

    motor_norm = mlp.predict(features)[0]
    left_target = denormalize_motor_speed(float(motor_norm[0]))
    right_target = denormalize_motor_speed(float(motor_norm[1]))

    left_cmd_smoothed = (1.0 - SMOOTHING_ALPHA) * left_cmd_smoothed + SMOOTHING_ALPHA * left_target
    right_cmd_smoothed = (1.0 - SMOOTHING_ALPHA) * right_cmd_smoothed + SMOOTHING_ALPHA * right_target

    th[node_id]["motor.left.target"] = int(left_cmd_smoothed)
    th[node_id]["motor.right.target"] = int(right_cmd_smoothed)

    if th[node_id]["button.center"]:
        print("Center button pressed, stopping.")
        th[node_id]["motor.left.target"] = 0
        th[node_id]["motor.right.target"] = 0
        set_leds(th, node_id, 0, 0, 0)
        done = True


ports = list(list_ports.comports())
if not ports:
    print("No Thymio port found")
    raise SystemExit(1)

thymio_ports = []
for port in ports:
    info = f"{port.description} {port.manufacturer} {port.hwid}".lower()
    if "thymio" in info:
        thymio_ports.append(port)

candidate_ports = thymio_ports if thymio_ports else ports
serial_port = candidate_ports[0].device
print("Thymio port:", serial_port)

th = Thymio(
    use_tcp=False,
    serial_port=serial_port,
    refreshing_coverage={
        "prox.ground.reflected",
        "prox.horizontal",
        "button.center",
    },
)
th.on_comm_error = on_comm_error
th.connect()
node_id = th.first_node()

set_leds(th, node_id, 255, 120, 0)
th.set_variable_observer(node_id, obs)
print("Q9 model running...")

while not done:
    time.sleep(0.1)

print("Disconnecting Thymio")
th.disconnect()
