from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ORBFeature:
    pt: cv2.typing.Point2f
    size: float
    angle: float
    response: float
    octave: int
    class_id: int
    descriptor: np.ndarray


@dataclass
class ORBFeatures:
    features: list[ORBFeature]


@dataclass
class FundamentalMatrix:
    matrix: np.ndarray


@dataclass
class Pair2FundamentalMatrix:
    pairs: dict[tuple[int, int], FundamentalMatrix]


@dataclass
class PairMatches:
    good_matches: list[cv2.DMatch]
    pts1: list[cv2.KeyPoint]
    pts2: list[cv2.KeyPoint]
    desc1: list[np.ndarray]
    desc2: list[np.ndarray]


@dataclass
class Pair2Matches:
    pairs: dict[tuple[int, int], PairMatches]


class PairManager:
    def __init__(self):
        self.pair2fundamental_matrix = {}
        self.pair2matches = {}

    def add_fundamental_matrix(
        self, img_id1: int, img_id2: int, matrix: FundamentalMatrix
    ):
        self.pair2fundamental_matrix[(img_id1, img_id2)] = matrix

    def add_matches(self, img_id1: int, img_id2: int, matches: PairMatches):
        self.pair2matches[(img_id1, img_id2)] = matches

    def get_fundamental_matrix_by_id(
        self, img_id1: int, img_id2: int
    ) -> FundamentalMatrix:
        return self.pair2fundamental_matrix.get((img_id1, img_id2))

    def get_matches_by_id(self, img_id1: int, img_id2: int) -> PairMatches:
        return self.pair2matches.get((img_id1, img_id2))

    def get_pairs_by_first_id(self, img_id: int) -> list[int]:
        return [pair for pair in self.pair2matches if pair[0] == img_id]

    def get_kpt2_by_kpt_1_from_pair(self, pair: tuple[int, int], pt1) -> np.ndarray:
        if pt1 in self.pair2matches[pair].pts1:
            return self.pair2matches[pair].pts2
