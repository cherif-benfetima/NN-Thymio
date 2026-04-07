import argparse
import glob
import os
from datetime import datetime

import h5py
import numpy as np


FEATURE_ORDER = [
    "ground_left_norm",
    "ground_right_norm",
    "prox_h0_norm",
    "prox_h1_norm",
    "prox_h2_norm",
    "prox_h3_norm",
    "prox_h4_norm",
]


def find_latest_source(default_pattern="line_follower/donnees_q9_*.h5"):
    candidates = glob.glob(default_pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_xy_from_source(source_path):
    with h5py.File(source_path, "r") as f:
        gl = np.array(f["entrees/ground_left_norm"], dtype=np.float32)
        gr = np.array(f["entrees/ground_right_norm"], dtype=np.float32)
        h0 = np.array(f["entrees/prox_h0_norm"], dtype=np.float32)
        h1 = np.array(f["entrees/prox_h1_norm"], dtype=np.float32)
        h2 = np.array(f["entrees/prox_h2_norm"], dtype=np.float32)
        h3 = np.array(f["entrees/prox_h3_norm"], dtype=np.float32)
        h4 = np.array(f["entrees/prox_h4_norm"], dtype=np.float32)

        ml = np.array(f["sorties/moteur_gauche_norm"], dtype=np.float32)
        mr = np.array(f["sorties/moteur_droit_norm"], dtype=np.float32)

    n = len(gl)
    if not (
        len(gr) == n
        and len(h0) == n
        and len(h1) == n
        and len(h2) == n
        and len(h3) == n
        and len(h4) == n
        and len(ml) == n
        and len(mr) == n
    ):
        raise ValueError("Inconsistent source lengths in HDF5 data.")

    X = np.column_stack((gl, gr, h0, h1, h2, h3, h4)).astype(np.float32)
    y = np.column_stack((ml, mr)).astype(np.float32)
    return X, y


def save_transformed_dataset(output_path, X, y, source_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.attrs["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["source_file"] = source_path
        f.attrs["feature_order"] = ",".join(FEATURE_ORDER)
        f.attrs["x_description"] = str(FEATURE_ORDER)
        f.attrs["y_description"] = "[moteur_gauche_norm,moteur_droit_norm]"


def main():
    parser = argparse.ArgumentParser(
        description="Transforme les donnees Q9 en X/y pour un MLP unique line+obstacles."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Source HDF5. Si absent, prend le dernier line_follower/donnees_q9_*.h5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Sortie HDF5. Defaut: line_follower/line_follow_obstacles_dataset_<timestamp>.h5",
    )
    args = parser.parse_args()

    source_path = args.input or find_latest_source()
    if source_path is None:
        raise FileNotFoundError("No Q9 source found. Run Suiveur_q9_obstacles.py first.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"line_follower/line_follow_obstacles_dataset_{timestamp}.h5"

    X, y = build_xy_from_source(source_path)
    save_transformed_dataset(output_path, X, y, source_path)

    print("Q9 transformation done")
    print(f"Source : {source_path}")
    print(f"Output : {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
