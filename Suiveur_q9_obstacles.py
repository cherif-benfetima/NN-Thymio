from thymiodirect import Thymio
import numpy as np
import sys
import os
import time
from serial.tools import list_ports

try:
    import h5py
except ImportError:
    h5py = None

from datetime import datetime


# ======================
# Q9: Line follow + obstacle avoidance (single policy)
# Stable baseline: line-follow first, bounded obstacle override
# ======================

BASE_SPEED = 95
KP_LINE = 75.0
ERROR_ALPHA = 0.30
ERROR_DEADBAND = 0.04
MAX_CORRECTION = 85
MIN_FORWARD_SPEED = 55
MAX_MOTOR_SPEED = 500

BLACK_THRESHOLD = 531
RECOVER_INNER = 20
RECOVER_OUTER = 220
RECOVER_STRAIGHT = 120

# Obstacle thresholds (prox.horizontal)
OBS_CENTER_ON = 1200
OBS_SIDE_ON = 1500
OBS_CENTER_OFF = 700
OBS_VERY_NEAR = 2200
OBS_CLEAR_STEPS = 10
OBS_MIN_STEPS = 18
REJOIN_STEPS = 52

# Normalization
GROUND_BLACK = 0.0
GROUND_WHITE = 1000.0
PROX_MIN = 0.0
PROX_MAX = 4500.0

# Smoothing
SMOOTH_ALPHA = 0.30


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_ground(value):
    denom = max(1e-9, GROUND_WHITE - GROUND_BLACK)
    return clamp((value - GROUND_BLACK) / denom, 0.0, 1.0)


def normalize_prox(value):
    denom = max(1e-9, PROX_MAX - PROX_MIN)
    return clamp((value - PROX_MIN) / denom, 0.0, 1.0)


def normalize_motor(speed):
    return clamp(speed / MAX_MOTOR_SPEED, -1.0, 1.0)


def clamp_motor(speed):
    return int(clamp(speed, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED))


def set_leds(th, node_id, r, g, b):
    th[node_id]["leds.top"] = [r, g, b]


def on_comm_error(error):
    print(error)
    os._exit(1)


# Logs
steps = []
ground_l_raw = []
ground_r_raw = []
prox_h0_raw = []
prox_h1_raw = []
prox_h2_raw = []
prox_h3_raw = []
prox_h4_raw = []
motor_l_raw = []
motor_r_raw = []
mode_log = []

# Runtime
# mode: 0=line, 1=avoid, 2=rejoin
done = False
mode_state = 0
mode_timer = 0
clear_count = 0
last_avoid_turn = "none"
last_turn = "none"

filtered_error = 0.0
left_cmd_smoothed = 0.0
right_cmd_smoothed = 0.0


def compute_line_follow(ground_l, ground_r):
    global filtered_error, last_turn

    if ground_l > BLACK_THRESHOLD and ground_r > BLACK_THRESHOLD:
        if last_turn == "left":
            return RECOVER_INNER, RECOVER_OUTER, 0.0, True
        if last_turn == "right":
            return RECOVER_OUTER, RECOVER_INNER, 0.0, True
        return RECOVER_STRAIGHT, RECOVER_STRAIGHT, 0.0, True

    left_dark = 1.0 - normalize_ground(ground_l)
    right_dark = 1.0 - normalize_ground(ground_r)

    raw_error = left_dark - right_dark
    filtered_error = (1.0 - ERROR_ALPHA) * filtered_error + ERROR_ALPHA * raw_error

    if abs(filtered_error) < ERROR_DEADBAND:
        correction = 0.0
    else:
        correction = clamp(KP_LINE * filtered_error, -MAX_CORRECTION, MAX_CORRECTION)

    if filtered_error > ERROR_DEADBAND:
        last_turn = "left"
    elif filtered_error < -ERROR_DEADBAND:
        last_turn = "right"
    else:
        last_turn = "none"

    left = clamp(BASE_SPEED - correction, MIN_FORWARD_SPEED, MAX_MOTOR_SPEED)
    right = clamp(BASE_SPEED + correction, MIN_FORWARD_SPEED, MAX_MOTOR_SPEED)
    return left, right, filtered_error, False


def compute_avoidance(prox_h):
    left_pressure = prox_h[0] + prox_h[1]
    right_pressure = prox_h[3] + prox_h[4]
    center = prox_h[2]

    if left_pressure >= right_pressure:
        turn_dir = "right"
    else:
        turn_dir = "left"

    if center > OBS_VERY_NEAR:
        # Emergency: reverse + rotate away
        if turn_dir == "right":
            return -120, 140, turn_dir
        return 140, -120, turn_dir

    # Normal avoid: forward arc away from obstacle
    if turn_dir == "right":
        return 95, 25, turn_dir
    return 25, 95, turn_dir


def obs(node_id):
    global done, mode_state, mode_timer, clear_count, last_avoid_turn
    global left_cmd_smoothed, right_cmd_smoothed

    if done:
        return

    ground = th[node_id]["prox.ground.reflected"]
    prox_h = th[node_id]["prox.horizontal"]

    g_left = int(ground[0])
    g_right = int(ground[1])
    h = [int(prox_h[i]) for i in range(5)]

    front_center = h[2]
    side_score = max(h[1], h[3])
    obstacle_present = (front_center > OBS_CENTER_ON) or (side_score > OBS_SIDE_ON)

    if mode_state == 0 and obstacle_present:
        mode_state = 1
        mode_timer = OBS_MIN_STEPS
        clear_count = 0

    if mode_state == 1:
        target_left, target_right, last_avoid_turn = compute_avoidance(h)
        err = 0.0
        lost_line = False

        mode_timer -= 1
        if front_center < OBS_CENTER_OFF and side_score < OBS_CENTER_OFF:
            clear_count += 1
        else:
            clear_count = 0

        if mode_timer <= 0 and clear_count >= OBS_CLEAR_STEPS:
            mode_state = 2
            mode_timer = REJOIN_STEPS

    elif mode_state == 2:
        line_left, line_right, err, lost_line = compute_line_follow(g_left, g_right)
        if last_avoid_turn == "right":
            bias_left = 60 + 22
            bias_right = 60 - 22
        elif last_avoid_turn == "left":
            bias_left = 60 - 22
            bias_right = 60 + 22
        else:
            bias_left = 60
            bias_right = 60

        blend = (mode_timer / float(REJOIN_STEPS)) ** 2.6
        target_left = blend * bias_left + (1.0 - blend) * min(line_left, 68)
        target_right = blend * bias_right + (1.0 - blend) * min(line_right, 68)

        mode_timer -= 1
        if mode_timer <= 0:
            mode_state = 0

    else:
        target_left, target_right, err, lost_line = compute_line_follow(g_left, g_right)

    left_cmd_smoothed = (1.0 - SMOOTH_ALPHA) * left_cmd_smoothed + SMOOTH_ALPHA * target_left
    right_cmd_smoothed = (1.0 - SMOOTH_ALPHA) * right_cmd_smoothed + SMOOTH_ALPHA * target_right

    cmd_left = clamp_motor(left_cmd_smoothed)
    cmd_right = clamp_motor(right_cmd_smoothed)

    th[node_id]["motor.left.target"] = cmd_left
    th[node_id]["motor.right.target"] = cmd_right

    idx = len(steps)
    steps.append(idx)
    ground_l_raw.append(g_left)
    ground_r_raw.append(g_right)
    prox_h0_raw.append(h[0])
    prox_h1_raw.append(h[1])
    prox_h2_raw.append(h[2])
    prox_h3_raw.append(h[3])
    prox_h4_raw.append(h[4])
    motor_l_raw.append(cmd_left)
    motor_r_raw.append(cmd_right)
    mode_log.append(int(mode_state))

    mode_name = "line" if mode_state == 0 else ("avoid" if mode_state == 1 else "rejoin")
    print(
        f"i={idx} mode={mode_name} t={mode_timer} "
        f"g=({g_left},{g_right}) h=({h[0]},{h[1]},{h[2]},{h[3]},{h[4]}) "
        f"cmd=({cmd_left},{cmd_right})"
    )

    if th[node_id]["button.center"]:
        print("Center button pressed -> stop")
        th[node_id]["motor.left.target"] = 0
        th[node_id]["motor.right.target"] = 0
        set_leds(th, node_id, 0, 0, 0)
        done = True


ports = list(list_ports.comports())
if not ports:
    print("No Thymio detected.")
    sys.exit(1)

thymio_ports = []
for port in ports:
    info = f"{port.description} {port.manufacturer} {port.hwid}".lower()
    if "thymio" in info:
        thymio_ports.append(port)

candidate_ports = thymio_ports if thymio_ports else ports
serial_port = candidate_ports[0].device
print(f"Detected port: {serial_port}")

try:
    th = Thymio(
        use_tcp=False,
        serial_port=serial_port,
        refreshing_coverage={
            "prox.ground.reflected",
            "prox.horizontal",
            "button.center",
        },
    )
except Exception as error:
    print(error)
    sys.exit(1)

th.on_comm_error = on_comm_error
th.connect()
node_id = th.first_node()

set_leds(th, node_id, 0, 255, 255)
th.set_variable_observer(node_id, obs)
print("Q9 collector running (center button to stop)...")

while not done:
    time.sleep(0.1)

try:
    if getattr(th, "thymio_proxy", None) is not None:
        th.thymio_proxy.loop.call_soon_threadsafe(th.thymio_proxy.loop.stop)
    if hasattr(th, "thread"):
        th.thread.join(timeout=1.0)
except Exception:
    pass

os.makedirs("line_follower", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"line_follower/donnees_q9_{timestamp}.h5"

if h5py is None:
    print("h5py not installed: skipping save.")
    sys.exit(0)

arr_steps = np.array(steps, dtype=np.int32)
arr_gl = np.array(ground_l_raw, dtype=np.int32)
arr_gr = np.array(ground_r_raw, dtype=np.int32)
arr_h0 = np.array(prox_h0_raw, dtype=np.int32)
arr_h1 = np.array(prox_h1_raw, dtype=np.int32)
arr_h2 = np.array(prox_h2_raw, dtype=np.int32)
arr_h3 = np.array(prox_h3_raw, dtype=np.int32)
arr_h4 = np.array(prox_h4_raw, dtype=np.int32)
arr_ml = np.array(motor_l_raw, dtype=np.int32)
arr_mr = np.array(motor_r_raw, dtype=np.int32)
arr_mode = np.array(mode_log, dtype=np.int8)

with h5py.File(output_file, "w") as f:
    g_in = f.create_group("entrees")
    g_in.create_dataset("temps", data=arr_steps)
    g_in.create_dataset("ground_left_raw", data=arr_gl)
    g_in.create_dataset("ground_right_raw", data=arr_gr)
    g_in.create_dataset("ground_left_norm", data=np.array([normalize_ground(v) for v in arr_gl], dtype=np.float32))
    g_in.create_dataset("ground_right_norm", data=np.array([normalize_ground(v) for v in arr_gr], dtype=np.float32))

    g_in.create_dataset("prox_h0_raw", data=arr_h0)
    g_in.create_dataset("prox_h1_raw", data=arr_h1)
    g_in.create_dataset("prox_h2_raw", data=arr_h2)
    g_in.create_dataset("prox_h3_raw", data=arr_h3)
    g_in.create_dataset("prox_h4_raw", data=arr_h4)
    g_in.create_dataset("prox_h0_norm", data=np.array([normalize_prox(v) for v in arr_h0], dtype=np.float32))
    g_in.create_dataset("prox_h1_norm", data=np.array([normalize_prox(v) for v in arr_h1], dtype=np.float32))
    g_in.create_dataset("prox_h2_norm", data=np.array([normalize_prox(v) for v in arr_h2], dtype=np.float32))
    g_in.create_dataset("prox_h3_norm", data=np.array([normalize_prox(v) for v in arr_h3], dtype=np.float32))
    g_in.create_dataset("prox_h4_norm", data=np.array([normalize_prox(v) for v in arr_h4], dtype=np.float32))

    g_out = f.create_group("sorties")
    g_out.create_dataset("moteur_gauche_raw", data=arr_ml)
    g_out.create_dataset("moteur_droit_raw", data=arr_mr)
    g_out.create_dataset("moteur_gauche_norm", data=np.array([normalize_motor(v) for v in arr_ml], dtype=np.float32))
    g_out.create_dataset("moteur_droit_norm", data=np.array([normalize_motor(v) for v in arr_mr], dtype=np.float32))

    g_meta = f.create_group("metadata")
    g_meta.attrs["n_samples"] = int(len(arr_steps))
    g_meta.attrs["timestamp"] = timestamp
    g_meta.attrs["mode_desc"] = "0=line_follow,1=obstacle_avoid,2=rejoin"
    g_meta.create_dataset("mode", data=arr_mode)

print(f"Q9 data saved: {output_file}")
