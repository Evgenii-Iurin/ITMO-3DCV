import cv2
import pickle
import numpy as np
from pathlib import Path

from common.dataclasses import ORBFeature, ORBFeatures


def extract_orb_features(
    id2imgpath: dict[int, str],
    path_to_pkl: Path | None = None,
    data_dir: str = "./data",
) -> dict[int, list[ORBFeature]]:
    id2orb_features = {}
    for img_id, img_path in id2imgpath.items():
        img = cv2.imread(
            f"{data_dir}/{img_path}", cv2.IMREAD_GRAYSCALE
        )  # pylint: disable=no-member
        orb = cv2.ORB_create()  # pylint: disable=no-member
        kpts, descriptors = orb.detectAndCompute(img, None)
        orb_features = ORBFeatures(
            features=[
                ORBFeature(
                    pt=kpt.pt,
                    size=kpt.size,
                    angle=kpt.angle,
                    response=kpt.response,
                    octave=kpt.octave,
                    class_id=kpt.class_id,
                    descriptor=desc,
                )
                for kpt, desc in zip(kpts, descriptors)
            ]
        )

        id2orb_features[img_id] = orb_features

    if path_to_pkl is not None:
        with open(path_to_pkl, "wb") as f:
            pickle.dump(id2orb_features, f)

    return id2orb_features


def extract_keypoints_and_descriptors(
    orb_features: ORBFeatures,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    features = orb_features.features
    key_points = [
        cv2.KeyPoint(
            x=orb_feature.pt[0],
            y=orb_feature.pt[1],
            size=orb_feature.size,
            angle=orb_feature.angle,
            response=orb_feature.response,
            octave=orb_feature.octave,
            class_id=orb_feature.class_id,
        )
        for orb_feature in features
    ]

    descriptor = np.array(
        [orb_feature.descriptor for orb_feature in features], dtype=np.float32
    )
    return key_points, descriptor
