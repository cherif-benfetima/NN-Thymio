from sklearn.neural_network import MLPRegressor

# Entrees possibles pour 2 variables binaires
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Sorties XOR (regression)
y_xor = [0.0, 1.0, 1.0, 0.0]


model = MLPRegressor(
    hidden_layer_sizes=(4, 2),  # au moins 2 couches cachees
    activation="tanh",
    solver="lbfgs",
    max_iter=10000,
    random_state=4,
)

model.fit(X, y_xor)

# Predictions continues (regression)
y_pred = model.predict(X)

# Verification logique XOR via arrondi a 0/1
y_pred_binary = [int(round(v)) for v in y_pred]

score = model.score(X, y_xor)

print("Exercice XOR avec MLPRegressor")
print("Configuration: hidden_layer_sizes=(4,2), activation=tanh, solver=lbfgs")
print(f"Score (R^2): {score:.6f}")
print("Predictions (continues):", [round(float(v), 6) for v in y_pred])
print("Predictions (arrondies):", y_pred_binary)
print("Attendus:", [int(v) for v in y_xor])


