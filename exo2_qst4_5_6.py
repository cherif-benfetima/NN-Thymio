from sklearn.neural_network import MLPClassifier

# Entrees possibles pour 2 variables binaires
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Labels XOR attendus
y_xor = [0, 1, 1, 0]


def train_mlp_xor(seed):
    model = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="tanh",
        max_iter=10000,
        random_state=seed,
    )
    model.fit(X, y_xor)

    preds = model.predict(X).tolist()
    score = model.score(X, y_xor)
    converged = model.n_iter_ < model.max_iter

    return model, preds, score, converged


def report_experiments():
    print("Exercice XOR avec MLPClassifier")
    print("Configuration: activation=tanh, 1 couche cachee, 2 neurones, max_iter=10000")
    print()

    print("Essais multiples (stabilite entre cycles d'apprentissage):")
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        _, preds, score, converged = train_mlp_xor(seed)
        print(
            f"seed={seed:>2} | convergence={converged} | predictions={preds} | score={score:.2f}"
        )

    print()
    print("Extraction des poids et biais (topologie de la figure), exemple seed=4:")
    model, preds, score, converged = train_mlp_xor(4)

    # Mapping de la figure:
    # w1a = x1 -> neurone cache 1, w1b = x1 -> neurone cache 2
    # w2a = x2 -> neurone cache 1, w2b = x2 -> neurone cache 2
    # w3 = neurone cache 1 -> sortie, w4 = neurone cache 2 -> sortie
    w1a = model.coefs_[0][0, 0]
    w1b = model.coefs_[0][0, 1]
    w2a = model.coefs_[0][1, 0]
    w2b = model.coefs_[0][1, 1]
    w3 = model.coefs_[1][0, 0]
    w4 = model.coefs_[1][1, 0]
    b1 = model.intercepts_[0][0]
    b2 = model.intercepts_[0][1]
    b3 = model.intercepts_[1][0]

    print(f"convergence={converged}")
    print(f"predictions={preds}")
    print(f"score={score:.2f}")
    print(f"w1a={w1a:.6f}, w1b={w1b:.6f}")
    print(f"w2a={w2a:.6f}, w2b={w2b:.6f}")
    print(f"w3={w3:.6f}, w4={w4:.6f}")
    print(f"b1={b1:.6f}, b2={b2:.6f}, b3={b3:.6f}")


if __name__ == "__main__":
    report_experiments()
