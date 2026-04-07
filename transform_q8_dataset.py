import argparse
import glob
import os
from datetime import datetime

import h5py
import numpy as np


def find_latest_source(default_pattern="line_follower/donnees_suiveur_*.h5"):
    candidates = glob.glob(default_pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_xy_from_source(source_path):
    with h5py.File(source_path, "r") as f:
        left_reflected = np.array(f["entrees/capteur_gauche_norm"], dtype=np.float32)
        right_reflected = np.array(f["entrees/capteur_droit_norm"], dtype=np.float32)
        left_ambiant = np.array(f["entrees/capteur_gauche_ambiant_norm"], dtype=np.float32)
        right_ambiant = np.array(f["entrees/capteur_droit_ambiant_norm"], dtype=np.float32)
        left_delta = np.array(f["entrees/capteur_gauche_delta_norm"], dtype=np.float32)
        right_delta = np.array(f["entrees/capteur_droit_delta_norm"], dtype=np.float32)
        left_motor = np.array(f["sorties/moteur_gauche_norm"], dtype=np.float32)
        right_motor = np.array(f["sorties/moteur_droit_norm"], dtype=np.float32)

    n = len(left_reflected)
    if not (
        len(right_reflected) == n
        and len(left_ambiant) == n
        and len(right_ambiant) == n
        and len(left_delta) == n
        and len(right_delta) == n
        and len(left_motor) == n
        and len(right_motor) == n
    ):
        raise ValueError("Les longueurs des jeux de donnees ne correspondent pas.")

    X = np.column_stack(
        (
            left_reflected,
            right_reflected,
            left_ambiant,
            right_ambiant,
            left_delta,
            right_delta,
        )
    ).astype(np.float32)
    y = np.column_stack((left_motor, right_motor)).astype(np.float32)
    return X, y


def save_transformed_dataset(output_path, X, y, source_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.attrs["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["source_file"] = source_path
        f.attrs["x_description"] = "[reflected_gauche_norm, reflected_droit_norm, ambiant_gauche_norm, ambiant_droit_norm, delta_gauche_norm, delta_droit_norm]"
        f.attrs["y_description"] = "[moteur_gauche_norm, moteur_droit_norm]"


def main():
    parser = argparse.ArgumentParser(
        description="Transforme les donnees HDF5 du suiveur en X/y pour l'entrainement MLP."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Chemin du fichier HDF5 source. Si non fourni, le plus recent line_follower/donnees_suiveur_*.h5 est utilise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier HDF5 de sortie. Defaut: line_follower/line_follow_dataset_<timestamp>.h5",
    )
    args = parser.parse_args()

    source_path = args.input or find_latest_source()
    if source_path is None:
        raise FileNotFoundError(
            "Aucun fichier source trouve. Lance d'abord la collecte pour creer line_follower/donnees_suiveur_*.h5"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"line_follower/line_follow_dataset_{timestamp}.h5"

    X, y = build_xy_from_source(source_path)
    save_transformed_dataset(output_path, X, y, source_path)

    print("Transformation terminee")
    print(f"Source : {source_path}")
    print(f"Sortie : {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
