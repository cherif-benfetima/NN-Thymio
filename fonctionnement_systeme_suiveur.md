# Fonctionnement du systeme de suiveur de ligne

## 1. Objectif general

Ce systeme permet de:

- piloter un robot Thymio pour suivre une ligne,
- enregistrer les donnees capteurs + moteurs,
- transformer ces donnees en dataset d entrainement,
- entrainer un modele MLP,
- reutiliser ce modele pour piloter le robot.

Le projet est donc organise en deux modes:

- mode controle classique (regles + proportionnel),
- mode apprentissage (modele neuronal).

## 2. Vue d ensemble des scripts

- Suiveur_de_ligne.py:
  - controle le robot en temps reel,
  - lit les capteurs sol,
  - calcule les commandes moteurs,
  - enregistre un fichier HDF5 source.

- transform_q8_dataset.py:
  - lit le fichier HDF5 source,
  - construit X (entrees) et y (sorties),
  - sauvegarde un dataset HDF5 pour l entrainement.

- TP3_modelq8.py:
  - charge le dataset transforme,
  - entraine un MLPRegressor,
  - sauvegarde le modele dans model_mlp_q8.pkl.

- TP3_q8.py:
  - charge le modele sauvegarde,
  - lit les capteurs du Thymio,
  - predit les vitesses moteur,
  - applique les commandes en temps reel.

## 3. Format des donnees HDF5

### 3.1 Fichier source genere par Suiveur_de_ligne.py

Le fichier est cree dans line_follower/donnees_suiveur_<timestamp>.h5.

Structure:

- groupe entrees/
  - temps
  - capteur_gauche_brut
  - capteur_droit_brut
  - capteur_gauche_ambiant_brut
  - capteur_droit_ambiant_brut
  - capteur_gauche_delta_brut
  - capteur_droit_delta_brut
  - capteur_gauche_norm
  - capteur_droit_norm
  - capteur_gauche_ambiant_norm
  - capteur_droit_ambiant_norm
  - capteur_gauche_delta_norm
  - capteur_droit_delta_norm

- groupe sorties/
  - moteur_gauche_brut
  - moteur_droit_brut
  - moteur_gauche_norm
  - moteur_droit_norm

- groupe metadata/
  - parametres de configuration (kp, base_speed, etc.)

### 3.2 Dataset transforme genere par transform_q8_dataset.py

Le fichier est cree dans line_follower/line_follow_dataset_<timestamp>.h5.

Structure:

- X: matrice des features (6 colonnes)
  - reflected_left
  - reflected_right
  - ambiant_left
  - ambiant_right
  - delta_left
  - delta_right

- y: matrice des cibles (2 colonnes)
  - moteur_gauche_norm
  - moteur_droit_norm

## 4. Logique de controle dans Suiveur_de_ligne.py

Le controle est base sur:

- une erreur laterale calculee avec les capteurs reflected,
- un correcteur proportionnel (KP),
- une zone morte (ERROR_DEADBAND),
- un filtrage exponentiel de l erreur (ERROR_ALPHA),
- un lissage progressif des consignes moteurs (ramp_towards).

Cas principaux:

- ligne detectee:
  - calcul de correction,
  - ajustement gauche/droite,
  - saturation des vitesses.

- ligne perdue:
  - strategie de recuperation selon le dernier sens de virage.

## 5. Pipeline complet d utilisation

1. Lancer Suiveur_de_ligne.py
   - enregistrer des trajectoires representatives.

2. Lancer transform_q8_dataset.py
   - produire X et y a partir du dernier HDF5 source.

3. Lancer TP3_modelq8.py
   - entrainer et sauvegarder model_mlp_q8.pkl.

4. Lancer TP3_q8.py
   - piloter le robot avec le modele appris.

## 6. Erreur frequente et cause

Erreur typique:

- KeyError: object 'capteur_gauche_ambiant_norm' doesn't exist

Cause:

- le fichier HDF5 source ne contient pas les 6 features attendues,
- souvent parce que le suiveur enregistre seulement reflected gauche/droite.

Solution:

- verifier que Suiveur_de_ligne.py enregistre bien les datasets ambiant et delta,
- regenerer un nouveau fichier donnees_suiveur_*.h5,
- relancer la transformation.

## 7. Conseils de validation rapide

- verifier les noms des datasets dans le HDF5 avant transformation,
- verifier la forme de X dans le dataset transforme:
  - attendu: X.shape = (N, 6)
- verifier y.shape:
  - attendu: y.shape = (N, 2)

Si ces formes sont correctes, le format est coherent pour l entrainement et l execution du modele.
