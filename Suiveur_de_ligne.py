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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


# Parameters line follower
BASE_SPEED       = 140

# Proportional control
KP               = 70.0
ERROR_DEADBAND   = 0.03
ERROR_ALPHA      = 0.30
MAX_CORRECTION   = 90
MIN_FORWARD_SPEED = 55
STEER_SIGN       = 1.0

# Recovery when line is lost
RECOVER_INNER    = 20
RECOVER_OUTER    = 260
RECOVER_STRAIGHT = 140

BLACK_THRESHOLD  = 531
MAX_MOTOR_SPEED  = 500

BLACK_VALUE = 0.0
WHITE_VALUE = 1000.0

# Adaptive smoothing
DELTA_STRAIGHT = 12
DELTA_TURN     = 20
DELTA_RECOVER  = 40


def normalize_ground(value: float) -> float:
    x = (value - BLACK_VALUE) / (WHITE_VALUE - BLACK_VALUE)
    return max(0.0, min(1.0, x))

def normalize_motor(speed: float) -> float:
    x = speed / MAX_MOTOR_SPEED
    return max(-1.0, min(1.0, x))

def clamp_motor(speed: float) -> int:
    return int(max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, speed)))

def clamp_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def ramp_towards(current: float, target: float, max_delta: float, step: float = 10.0) -> float:
    delta = target - current

    if abs(delta) < 1e-9:
        return target

    direction = 1.0 if delta > 0 else -1.0
    allowed_move = min(abs(delta), max_delta)

    moved = 0.0
    value = current

    while moved + step <= allowed_move:
        value += direction * step
        moved += step

    remaining = allowed_move - moved
    if remaining > 0:
        value += direction * remaining

    return value


def set_leds(th, id, R, G, B):
    src = """
        dc end_toc
        dc _ev.init, init
    end_toc:

    init:
        push.s 0
        store counter
        push.s """
    src2 = """
        store _userdata
        push.s _userdata
        push.s """
    src3 = """
        store _userdata+1
        push.s _userdata+1
        push.s """
    src4 = """
        store _userdata+2
        push.s _userdata+2
        callnat _nf."""
    src5 = """
        stop

    counter:
        equ _userdata+3
    """
    th.run_asm(id, src + str(B) + src2 + str(G) + src3 + str(R) + src4 + 'leds.top' + src5)


def on_comm_error(error):
    print(error)
    os._exit(1)


# Global variables
time_steps      = []
sensor_l_values = []
sensor_r_values = []
motor_l_values  = []
motor_r_values  = []

last_turn = "none"
current_left_speed  = 0.0
current_right_speed = 0.0
filtered_error = 0.0


def obs(node_id):
    global done, last_turn, current_left_speed, current_right_speed, filtered_error

    if not done:
        left  = th[node_id]["prox.ground.reflected"][0]
        right = th[node_id]["prox.ground.reflected"][1]

        left_on_line  = left < BLACK_THRESHOLD
        right_on_line = right < BLACK_THRESHOLD

        if not left_on_line and not right_on_line:
            if last_turn == "left":
                target_left  = RECOVER_INNER
                target_right = RECOVER_OUTER
            elif last_turn == "right":
                target_left  = RECOVER_OUTER
                target_right = RECOVER_INNER
            else:
                target_left  = RECOVER_STRAIGHT
                target_right = RECOVER_STRAIGHT

            max_delta = DELTA_RECOVER
            error = 0.0
            correction = 0.0
            left_dark = 0.0
            right_dark = 0.0
            raw_error = 0.0
        else:
            left_dark = 1.0 - normalize_ground(left)
            right_dark = 1.0 - normalize_ground(right)

            raw_error = float(STEER_SIGN * (left_dark - right_dark))

            filtered_error = (1.0 - ERROR_ALPHA) * filtered_error + ERROR_ALPHA * raw_error
            error = filtered_error

            if abs(error) < ERROR_DEADBAND:
                correction = 0.0
            else:
                correction = clamp_value(KP * error, -MAX_CORRECTION, MAX_CORRECTION)

            target_left  = BASE_SPEED - correction
            target_right = BASE_SPEED + correction

            target_left = clamp_value(target_left, MIN_FORWARD_SPEED, MAX_MOTOR_SPEED)
            target_right = clamp_value(target_right, MIN_FORWARD_SPEED, MAX_MOTOR_SPEED)

            if error > ERROR_DEADBAND:
                last_turn = "left"
                max_delta = DELTA_TURN
            elif error < -ERROR_DEADBAND:
                last_turn = "right"
                max_delta = DELTA_TURN
            else:
                last_turn = "none"
                max_delta = DELTA_STRAIGHT

        current_left_speed  = ramp_towards(current_left_speed, target_left, max_delta)
        current_right_speed = ramp_towards(current_right_speed, target_right, max_delta)

        cmd_left  = clamp_motor(current_left_speed)
        cmd_right = clamp_motor(current_right_speed)

        th[node_id]["motor.left.target"]  = cmd_left
        th[node_id]["motor.right.target"] = cmd_right

        g0 = normalize_ground(left)
        g1 = normalize_ground(right)
        vl = normalize_motor(cmd_left)
        vr = normalize_motor(cmd_right)

        print(
            f"raw=({left},{right})  "
            f"norm=({g0:.3f},{g1:.3f})  "
            f"dark=({left_dark:.3f},{right_dark:.3f})  "
            f"raw_err={raw_error:.3f}  "
            f"err={error:.3f}  "
            f"corr={correction:.1f}  "
            f"target=({target_left},{target_right})  "
            f"cmd=({cmd_left},{cmd_right})  "
            f"motor=({vl:.3f},{vr:.3f})  "
            f"last_turn={last_turn}"
        )

        time_steps.append(len(time_steps))
        sensor_l_values.append(int(left))
        sensor_r_values.append(int(right))
        motor_l_values.append(int(cmd_left))
        motor_r_values.append(int(cmd_right))

        if th[node_id]["button.center"]:
            print("button.center pressed -> stop")
            th[node_id]["motor.left.target"]  = 0
            th[node_id]["motor.right.target"] = 0
            set_leds(th, node_id, 0, 0, 0)
            done = True


ports = list(list_ports.comports())
if not ports:
    print("No Thymio detected. Check the connection.")
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
        refreshing_coverage={"prox.ground.reflected", "button.center"},
    )
except Exception as error:
    print(error)
    sys.exit(1)

th.on_comm_error = on_comm_error
th.connect()
id = th.first_node()
done = False

set_leds(th, id, 0, 255, 0)
th.set_variable_observer(id, obs)
print("Thymio running (center button to stop)...")

while not done:
    time.sleep(0.1)

print("Disconnecting Thymio...")
try:
    if getattr(th, "thymio_proxy", None) is not None:
        th.thymio_proxy.loop.call_soon_threadsafe(th.thymio_proxy.loop.stop)
    if hasattr(th, "thread"):
        th.thread.join(timeout=1.0)
except Exception:
    pass

os.makedirs("line_follower", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

n = len(time_steps)

arr_sensor_l = np.array(sensor_l_values, dtype=np.int32)
arr_sensor_r = np.array(sensor_r_values, dtype=np.int32)
arr_motor_l  = np.array(motor_l_values, dtype=np.int32)
arr_motor_r  = np.array(motor_r_values, dtype=np.int32)

arr_norm_gl  = np.array([normalize_ground(v) for v in sensor_l_values], dtype=np.float32)
arr_norm_gr  = np.array([normalize_ground(v) for v in sensor_r_values], dtype=np.float32)
arr_norm_ml  = np.array([normalize_motor(v) for v in motor_l_values], dtype=np.float32)
arr_norm_mr  = np.array([normalize_motor(v) for v in motor_r_values], dtype=np.float32)

print(f"Recorded samples: {n}")

if time_steps:
    plt.figure(figsize=(12, 6))

    plt.plot(time_steps, motor_l_values, marker='o', linewidth=2, markersize=4, label='Left speed')
    plt.plot(time_steps, motor_r_values, marker='o', linewidth=2, markersize=4, label='Right speed')

    plt.axhline(y=0, linestyle='--', linewidth=0.8)
    plt.title('Motor speed evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (iterations)')
    plt.ylabel('Motor speed (Thymio units)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()

    plot_filename = f"line_follower/speeds_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Speed plot saved: '{plot_filename}'")
    plt.close()
else:
    print("No speed data to plot.")

print("\n" + "="*70)
print("SAVE DATA TO HDF5")
print("="*70)

hdf5_filename = f"line_follower/donnees_suiveur_{timestamp}.h5"

if h5py is None:
    print("h5py not installed: skipping HDF5 save (pip install h5py).")
else:
    try:
        with h5py.File(hdf5_filename, 'w') as f:
            grp_input = f.create_group('entrees')
            grp_input.create_dataset('temps', data=np.array(time_steps, dtype=np.int32))
            grp_input.create_dataset('capteur_gauche_brut', data=arr_sensor_l)
            grp_input.create_dataset('capteur_droit_brut', data=arr_sensor_r)
            grp_input.create_dataset('capteur_gauche_norm', data=arr_norm_gl)
            grp_input.create_dataset('capteur_droit_norm', data=arr_norm_gr)

            grp_output = f.create_group('sorties')
            grp_output.create_dataset('moteur_gauche_brut', data=arr_motor_l)
            grp_output.create_dataset('moteur_droit_brut', data=arr_motor_r)
            grp_output.create_dataset('moteur_gauche_norm', data=arr_norm_ml)
            grp_output.create_dataset('moteur_droit_norm', data=arr_norm_mr)

            grp_meta = f.create_group('metadata')
            grp_meta.attrs['nombre_echantillons'] = n
            grp_meta.attrs['timestamp'] = timestamp
            grp_meta.attrs['base_speed'] = BASE_SPEED
            grp_meta.attrs['kp'] = KP
            grp_meta.attrs['error_deadband'] = ERROR_DEADBAND
            grp_meta.attrs['error_alpha'] = ERROR_ALPHA
            grp_meta.attrs['max_correction'] = MAX_CORRECTION
            grp_meta.attrs['min_forward_speed'] = MIN_FORWARD_SPEED
            grp_meta.attrs['steer_sign'] = STEER_SIGN
            grp_meta.attrs['recover_inner'] = RECOVER_INNER
            grp_meta.attrs['recover_outer'] = RECOVER_OUTER
            grp_meta.attrs['recover_straight'] = RECOVER_STRAIGHT
            grp_meta.attrs['black_threshold'] = BLACK_THRESHOLD
            grp_meta.attrs['max_motor_speed'] = MAX_MOTOR_SPEED
            grp_meta.attrs['black_value'] = BLACK_VALUE
            grp_meta.attrs['white_value'] = WHITE_VALUE
            grp_meta.attrs['delta_straight'] = DELTA_STRAIGHT
            grp_meta.attrs['delta_turn'] = DELTA_TURN
            grp_meta.attrs['delta_recover'] = DELTA_RECOVER

            print(f"HDF5 file created: '{hdf5_filename}'")
            print(f"Recorded samples: {n}")

    except Exception as e:
        print(f"Error while saving HDF5: {e}")

print("="*70)
