# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error
# import joblib
# import matplotlib.pyplot as plt
# import h5py
# import numpy as np


# # ======================
# # 2. Recuperation des données
# # ======================

# file_path = "TP3/TP/datasets/line_follow_dataset_20260401_110925.h5"
# with h5py.File(file_path, "r") as f:
#     # lister les datasets
#     print("Contenu du fichier :", list(f.keys()))
    
#     X = np.array(f["X"])  
#     y = np.array(f["y"])  
# print(X[:5])  # Affiche les 5 premières lignes de X
# print(y[:5])  # Affiche les 5 premières valeurs de y
# print("Shape X :", X.shape)
# print("Shape y :", y.shape)

# # ======================
# # 2. Split des données
# # ======================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42
# )


# # ======================
# # 3. Modèle
# # ======================
# mlp = MLPRegressor(
#     hidden_layer_sizes=(3,2),
#     activation='tanh',
#     solver='adam',
#     max_iter=1000,
#     random_state=None)

# # ======================
# # 4. Entraînement
# # ======================
# mlp.fit(X_train, y_train)

# # ======================
# # 5. Évaluation
# # ======================
# def evaluate(X, y, name):
#     pred = mlp.predict(X)
#     mse = mean_squared_error(y, pred)
#     r2 = mlp.score(X, y)
#     print(f"{name} → MSE: {mse:.4f} | R²: {r2:.4f}")

# evaluate(X_train, y_train, "Train")
# evaluate(X_test, y_test, "Test")

# # ======================
# # 6. Loss
# # ======================
# print("Loss finale        :", mlp.loss_)
# print("Nombre d'itérations:", mlp.n_iter_)

# # ======================
# # 7. Courbe de loss
# # ======================
# plt.plot(mlp.loss_curve_)
# plt.xlabel("Itérations")
# plt.ylabel("Loss")
# plt.title("Évolution de la loss")
# plt.show()

# # ======================
# # 8. Sauvegarde modèle
# # ======================
# joblib.dump(mlp, "model_mlp_q7.pkl")

# # | Méthode              | Utilité               |
# # | -------------------- | --------------------- |
# # | `mlp.loss_`          | suivre l'entraînement |
# # | `mean_squared_error` | vraie erreur          |
# # | `score()` (R²)       | qualité globale       |


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
import os

# ======================
# 1. Récupération des données HDF5
# ======================
dataset_candidates = glob.glob("line_follower/line_follow_dataset_*.h5")
if not dataset_candidates:
    raise FileNotFoundError(
        "Aucun dataset transforme trouve. Lance d'abord transform_q8_dataset.py"
    )

file_path = max(dataset_candidates, key=os.path.getmtime)
print("Dataset charge:", file_path)
with h5py.File(file_path, "r") as f:
    print("Contenu du fichier :", list(f.keys()))
    X = np.array(f["X"])
    y = np.array(f["y"])

print("Shape X :", X.shape)
print("Shape y :", y.shape)
print("Exemple X :", X[:5])
print("Exemple y :", y[:5])

# ======================
# 2. Split des données
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ======================
# 3. Normalisation coherente (train/test + runtime)
# ======================
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# ======================
# 4. Modèle MLP
# ======================
mlp = MLPRegressor(
    hidden_layer_sizes=(12, 8),
    activation='tanh',
    solver='adam',
    max_iter=2000,
    random_state=42,
    early_stopping=True,       # arrête si pas d'amélioration
    n_iter_no_change=50,
    validation_fraction=0.2
)

# ======================
# 5. Entraînement
# ======================
mlp.fit(X_train_scaled, y_train)

# ======================
# 6. Évaluation
# ======================
def evaluate(X, y, name):
    pred = mlp.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = mlp.score(X, y)
    print(f"{name} → MSE: {mse:.6f} | R²: {r2:.4f}")

evaluate(X_train_scaled, y_train, "Train")
evaluate(X_test_scaled, y_test, "Test")

# ======================
# 7. Loss et courbe
# ======================
print("Loss finale        :", mlp.loss_)
print("Nombre d'itérations:", mlp.n_iter_)
print("Topologie (MLP):", mlp.hidden_layer_sizes)
print("Activation:", mlp.activation)
for i, w in enumerate(mlp.coefs_):
    print(f"Poids couche {i}: shape={w.shape}")
    print(w)
for i, b in enumerate(mlp.intercepts_):
    print(f"Biais couche {i}: shape={b.shape}")
    print(b)

plt.plot(mlp.loss_curve_)
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.title("Évolution de la loss")
plt.show()

# ======================
# 8. Sauvegarde du modèle + normalisation
# ======================
bundle = {
    "model": mlp,
    "scaler_x": scaler_x,
    "normalization": {
        "ground_black": 0.0,
        "ground_white": 1000.0,
        "motor_max": 500.0,
    },
    "feature_order": [
        "reflected_left",
        "reflected_right",
        "ambiant_left",
        "ambiant_right",
        "delta_left",
        "delta_right",
    ],
}
joblib.dump(bundle, "model_mlp_q8.pkl")
print("Modele sauvegarde: model_mlp_q8.pkl")

# ======================
# 9. Exemple de prédiction avec un nouveau jeu de données
# ======================
# data = [[x1, x2, x3, x4, x5, x6]]
# data_norm = scaler_X.transform(data)
# pred_norm = mlp.predict(data_norm)
# pred = scaler_y.inverse_transform(pred_norm)
# print("Prédiction :", pred)