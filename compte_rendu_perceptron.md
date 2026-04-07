# Compte rendu - TP Apprentissage supervise

## 1. Objectif
Ce travail pratique a pour objectif de concevoir et entrainer des perceptrons pour trois operateurs logiques : OR, AND et XOR, puis d'analyser la qualite des predictions.

## 2. Methode
Les etapes suivies sont les suivantes :
1. Importer la classe Perceptron de scikit-learn.
2. Definir les entrees binaires possibles : [0,0], [0,1], [1,0], [1,1].
3. Definir les labels attendus pour OR, AND et XOR.
4. Entrainer un perceptron pour chaque operateur avec max_iter=40 et tol=1e-3.
5. Evaluer les predictions avec predict et la precision avec score.

## 3. Resultats obtenus
### Operateur OR
- Labels attendus : [0, 1, 1, 1]
- Predictions : [0, 1, 1, 1]
- Score : 1.00

### Operateur AND
- Labels attendus : [0, 0, 0, 1]
- Predictions : [0, 0, 0, 1]
- Score : 1.00

### Operateur XOR
- Labels attendus : [0, 1, 1, 0]
- Predictions : [0, 0, 0, 0]
- Score : 0.50

## 4. Reponses aux questions
### Question 1
Les perceptrons pour OR, AND et XOR ont bien ete concus et entraines.

### Question 2
Les predictions ne sont pas toutes correctes.
- OR : toutes les predictions sont correctes.
- AND : toutes les predictions sont correctes.
- XOR : les predictions sont incorrectes.

Explication : un perceptron simple realise une separation lineaire. Les donnees du XOR ne sont pas lineairement separables, donc un seul perceptron ne peut pas apprendre correctement cette fonction logique.

### Question 3
La fonction score mesure la precision moyenne de classification (accuracy), c'est-a-dire la proportion de bonnes predictions.
- Plage de valeurs : de 0 a 1.
- 1.00 signifie une classification parfaite sur les donnees testees.
- 0.50 signifie que seulement la moitie des predictions est correcte.

Dans ce TP, le score reflete bien la qualite des resultats :
- OR et AND : score eleve (1.00), donc apprentissage reussi.
- XOR : score faible (0.50), donc apprentissage insuffisant avec un perceptron simple.

## 5. Conclusion
Le perceptron simple est adapte aux problemes lineairement separables (OR, AND), mais il atteint ses limites sur XOR. Pour traiter XOR correctement, il faut un modele plus expressif, par exemple un perceptron multicouche (MLP).

## 6. Exercice XOR comme classifieur (Questions 4, 5 et 6)

### Question 4
Comme constate dans l'exercice precedent, XOR n'est pas lineairement separable. Un classifieur lineaire (perceptron simple) ne peut donc pas modeliser correctement cet operateur.

La solution adoptee est d'utiliser un perceptron multi-couche avec `MLPClassifier` de scikit-learn.

### Question 5
Configuration demandee :
- Fonction d'activation : `tanh`
- Topologie : 1 couche cachee de 2 neurones
- Limite d'iterations : `max_iter=10000`

Resultats des essais (plusieurs cycles d'apprentissage avec des initialisations differentes) :
- seed=0 : convergence=True, predictions=[1, 1, 1, 0], score=0.75
- seed=1 : convergence=True, predictions=[0, 1, 1, 1], score=0.75
- seed=2 : convergence=True, predictions=[0, 0, 0, 0], score=0.50
- seed=3 : convergence=True, predictions=[0, 1, 0, 1], score=0.50
- seed=4 : convergence=True, predictions=[0, 1, 1, 0], score=1.00
- seed=5 : convergence=True, predictions=[0, 1, 1, 0], score=1.00
- seed=6 : convergence=True, predictions=[0, 1, 1, 0], score=1.00
- seed=7 : convergence=True, predictions=[1, 0, 0, 0], score=0.25
- seed=8 : convergence=True, predictions=[1, 1, 1, 0], score=0.75
- seed=9 : convergence=True, predictions=[0, 0, 0, 1], score=0.25

Analyse :
- Le modele converge dans les essais observes (`convergence=True`).
- Le score peut atteindre 1 (cas parfait sur XOR, par exemple seeds 4, 5, 6).
- Les resultats ne sont pas identiques a chaque cycle d'apprentissage. Cela s'explique par l'initialisation aleatoire des poids (dependance au `random_state`) et l'optimisation iterative.

### Question 6
Pour l'exemple `seed=4` (score=1.00), les poids et biais extraits sont :

- Poids de la couche d'entree vers la couche cachee :
	- w1a = 3.245433
	- w1b = 1.539349
	- w2a = 3.195208
	- w2b = 1.579792

- Poids de la couche cachee vers la sortie :
	- w3 = 3.243366
	- w4 = -3.309809

- Biais :
	- b1 = -1.218101
	- b2 = -2.147005
	- b3 = -2.435114

Ces valeurs correspondent aux attributs `coefs_` (poids) et `intercepts_` (biais) du modele MLP, conformement a l'enonce.

## 7. Conclusion generale
Le TP montre clairement la difference entre un modele lineaire et un modele non lineaire :
- Le perceptron simple est adapte aux fonctions lineairement separables (OR, AND).
- Pour XOR, un reseau multi-couche est necessaire.
- Avec `MLPClassifier`, XOR peut etre appris parfaitement (score=1), mais la qualite finale depend de l'initialisation et peut varier d'un essai a l'autre.

## 8. Exercice XOR comme une regression (Question 7)

### Objectif
Representer l'operateur XOR comme un probleme de regression en utilisant `MLPRegressor`.

### Configuration du modele
- Modele : `MLPRegressor`
- Topologie : au moins 2 couches cachees (`hidden_layer_sizes=(4,2)`)
- Activation : `tanh`
- Optimisation : `lbfgs`
- Iterations max : `10000`

### Entrainement
Le modele est entraine avec :
- Entrees : `[[0,0], [0,1], [1,0], [1,1]]`
- Sorties cibles (XOR) : `[0.0, 1.0, 1.0, 0.0]`

### Resultats
- Score du modele (`R^2`) : `1.000000`
- Predictions continues : `[6e-05, 0.999948, 0.999891, 0.000205]`
- Predictions arrondies : `[0, 1, 1, 0]`
- Valeurs attendues : `[0, 1, 1, 0]`

### Interpretation
- Les predictions continues sont tres proches des cibles 0 et 1.
- Apres arrondi, les sorties correspondent exactement a XOR.
- Le score `R^2 = 1` indique un ajustement excellent sur les donnees d'apprentissage.
