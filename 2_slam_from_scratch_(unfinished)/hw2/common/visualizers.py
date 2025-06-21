import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from common.constants import PATH_TO_INPUT_FOLDER


def visualize_orb_features(img_path: Path, kpts):
    img = cv2.imread(f"{PATH_TO_INPUT_FOLDER}/{img_path}", cv2.IMREAD_COLOR)
    img = cv2.drawKeypoints(
        img,
        kpts,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


def visualize_matches(
    img1_path: Path,
    img2_path: Path,
    kpts1: list[cv2.KeyPoint],
    kpts2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    noMatches=250,
):
    img1 = cv2.imread(f"{PATH_TO_INPUT_FOLDER}/{img1_path}", cv2.IMREAD_COLOR)
    img2 = cv2.imread(f"{PATH_TO_INPUT_FOLDER}/{img2_path}", cv2.IMREAD_COLOR)

    matchesImage = np.zeros(shape=(1, 1))
    matchesImage = cv2.drawMatchesKnn(
        img1=img1,
        keypoints1=kpts1,
        img2=img2,
        keypoints2=kpts2,
        matches1to2=matches[:noMatches],
        outImg=matchesImage,
        flags=2,
    )

    cv2.imshow("BFMatcherknn", matchesImage)
    cv2.waitKey(0)
