from sklearn.linear_model import Perceptron


# Entrees possibles pour 2 variables binaires
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Labels attendus
y_or = [0, 1, 1, 1]
y_and = [0, 0, 0, 1]
y_xor = [0, 1, 1, 0]


def train_and_report(name, y):
	model = Perceptron(max_iter=40, tol=1e-3, random_state=0, shuffle=False)
	model.fit(X, y)

	preds = model.predict(X).tolist()
	score = model.score(X, y)

	print(f"{name}")
	print(f"  labels attendus : {y}")
	print(f"  predictions     : {preds}")
	print(f"  score           : {score:.2f}")
	print()


if __name__ == "__main__":
	train_and_report("Operateur OR", y_or)
	train_and_report("Operateur AND", y_and)
	train_and_report("Operateur XOR", y_xor)