from pathlib import Path
import cv2
import numpy as np
import pickle
from common.dataclasses import ORBFeature


def read_orb_features(
    path_to_pkl: Path,
) -> dict[int, tuple[list[cv2.KeyPoint], np.ndarray]]:
    with open(path_to_pkl, "rb") as f:
        id2orb_features = pickle.load(f)
    return id2orb_features
