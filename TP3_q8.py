from thymiodirect import Thymio
from serial.tools import list_ports
import os
import time
import joblib
import numpy as np

# ======================
# 1. Chargement du modèle MLP
# ======================
loaded = joblib.load("model_mlp_q8.pkl")
if isinstance(loaded, dict) and "model" in loaded:
    mlp = loaded["model"]
    scaler_x = loaded.get("scaler_x")
    norm_cfg = loaded.get("normalization", {})
else:
    # Compatibilite ancien format (modele seul)
    mlp = loaded
    scaler_x = None
    norm_cfg = {}

# ======================
# 2. Constantes
# ======================
GROUND_BLACK = float(norm_cfg.get("ground_black", 0.0))
GROUND_WHITE = float(norm_cfg.get("ground_white", 1000.0))
MOTOR_TARGET_MAX = float(norm_cfg.get("motor_max", 500.0))
SMOOTHING_ALPHA = 0.25

# ======================
# 3. Fonctions utilitaires
# ======================
def clamp(value, low, high):
    return max(low, min(value, high))

def normalize_ground(value):
    denom = max(1e-9, GROUND_WHITE - GROUND_BLACK)
    return clamp((value - GROUND_BLACK) / denom, 0.0, 1.0)

def denormalize_motor_speed(speed_norm):
    return int(clamp(speed_norm, -1.0, 1.0) * MOTOR_TARGET_MAX)

def set_leds(th, node_id, R, G, B):
    th[node_id]["leds.top"] = [R, G, B]

def on_comm_error(error):
    print("Erreur de communication:", error)
    os._exit(1)  # sortie forcée

# ======================
# 4. Callback Thymio
# ======================
done = False
left_cmd_smoothed = 0.0
right_cmd_smoothed = 0.0

def obs(node_id):
    global done, left_cmd_smoothed, right_cmd_smoothed
    if done:
        return

    # Lecture des capteurs sol
    reflected = th[node_id]["prox.ground.reflected"]
    ambient = th[node_id]["prox.ground.ambiant"]
    delta = th[node_id]["prox.ground.delta"]

    # Normalisation des 6 valeurs
    sensors = [
        normalize_ground(reflected[0]),
        normalize_ground(reflected[1]),
        normalize_ground(ambient[0]),
        normalize_ground(ambient[1]),
        normalize_ground(delta[0]),
        normalize_ground(delta[1])
    ]

    # Prédiction MLP
    features = np.array([sensors], dtype=np.float32)
    if scaler_x is not None:
        features = scaler_x.transform(features)

    motor_norm = mlp.predict(features)[0]
    left_target = denormalize_motor_speed(float(motor_norm[0]))
    right_target = denormalize_motor_speed(float(motor_norm[1]))

    # Lissage exponentiel des commandes pour limiter les oscillations.
    left_cmd_smoothed = (
        (1.0 - SMOOTHING_ALPHA) * left_cmd_smoothed + SMOOTHING_ALPHA * left_target
    )
    right_cmd_smoothed = (
        (1.0 - SMOOTHING_ALPHA) * right_cmd_smoothed + SMOOTHING_ALPHA * right_target
    )

    th[node_id]["motor.left.target"] = int(left_cmd_smoothed)
    th[node_id]["motor.right.target"] = int(right_cmd_smoothed)

    # Arrêt si bouton central pressé
    if th[node_id]["button.center"]:
        print("Bouton central pressé, arrêt du robot.")
        th[node_id]["motor.left.target"] = 0
        th[node_id]["motor.right.target"] = 0
        set_leds(th, node_id, 0, 0, 0)
        done = True

# ======================
# 5. Connexion Thymio
# ======================
ports = list(list_ports.comports())
if not ports:
    print("Aucun port Thymio trouvé")
    exit(1)

thymio_ports = []
for port in ports:
    info = f"{port.description} {port.manufacturer} {port.hwid}".lower()
    if "thymio" in info:
        thymio_ports.append(port)

candidate_ports = thymio_ports if thymio_ports else ports
serial_port = candidate_ports[0].device
print("Port Thymio trouvé:", serial_port)

th = Thymio(
    use_tcp=False,
    serial_port=serial_port,
    refreshing_coverage={
        "prox.ground.reflected",
        "prox.ground.ambiant",
        "prox.ground.delta",
        "button.center",
    }
)
th.on_comm_error = on_comm_error
th.connect()
node_id = th.first_node()

# ======================
# 6. Initialisation
# ======================
set_leds(th, node_id, 0, 0, 255)  # LED bleue pour signaler l’exécution
th.set_variable_observer(node_id, obs)
print("Thymio en exécution...")

# ======================
# 7. Boucle principale
# ======================
while not done:
    time.sleep(0.1)

print("Déconnexion Thymio")
th.disconnect()