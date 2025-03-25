import cv2
import numpy as np

from common.dataclasses import ORBFeatures
from common.feature_extractor import extract_keypoints_and_descriptors


class MatchMethods:
    BRUTE_FORCE = "brute_force"
    KNN_MATCHER_WITH_LOWES_RATIO = "knn_matcher_with_lowes_ratio"
    KNN_MATCHER_WITH_FUNDAMENTAL_MATRIX = "knn_matcher_with_fundamental_matrix"


class ORBMatcher:
    def __init__(
        self,
        method: str = MatchMethods.KNN_MATCHER_WITH_FUNDAMENTAL_MATRIX,
    ):
        self.method = method
        self.knownimg2matches = {}
        self.unknownimg2matches = {}

    def match_all_known_images(self, id2orb_features: dict[int, ORBFeatures]):
        """
        Iterativalle matches all images with each other.
        """
        ids = list(id2orb_features.keys())
        for i, img_id_1 in enumerate(ids):
            kpts1, des1 = extract_keypoints_and_descriptors(id2orb_features[img_id_1])
            for j, img_id_2 in enumerate(ids[i + 1 :]):
                print(f"- Pair ({img_id_1}, {img_id_2}):")
                kpts2, des2 = extract_keypoints_and_descriptors(id2orb_features[ids[j]])

                (
                    f_matrix,
                    inlier_points_image_1,
                    inlier_points_image_2,
                    inlier_descriptors_image_1,
                    inlier_descriptors_image_2,
                ) = self.knn_matcher_with_fundamental_matrix(des1, des2, kpts1, kpts2)
                if f_matrix is not None:
                    self.knownimg2matches[(img_id_1, img_id_2)] = {
                        "fundamental_matrix": f_matrix,
                        img_id_1: {
                            "points": inlier_points_image_1,
                            "descriptors": inlier_descriptors_image_1,
                        },
                        img_id_2: {
                            "points": inlier_points_image_2,
                            "descriptors": inlier_descriptors_image_2,
                        },
                    }

        return self.knownimg2matches

    @staticmethod
    def knn_matcher_with_fundamental_matrix(des1, des2, kpts1, kpts2):
        # Convert descriptors to uint8
        des1 = np.uint8(des1)
        des2 = np.uint8(des2)

        # Use BFMatcher with Hamming distance for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        points_image_1 = []
        points_image_2 = []
        descriptors_image_1 = []
        descriptors_image_2 = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                points_image_1.append(kpts1[m.queryIdx].pt)
                points_image_2.append(kpts2[m.trainIdx].pt)
                descriptors_image_1.append(des1[m.queryIdx])
                descriptors_image_2.append(des2[m.trainIdx])

        # Check if we have enough points for fundamental matrix calculation
        if len(points_image_1) < 8:  # Fundamental matrix needs at least 8 points
            print("Not enough matches to calculate fundamental matrix")
            return None, [], [], [], []

        points_image_1 = np.float32(points_image_1)
        points_image_2 = np.float32(points_image_2)

        # Use RANSAC for fundamental matrix
        F, mask = cv2.findFundamentalMat(
            points_image_1, points_image_2, cv2.FM_RANSAC, 3.0, 0.99
        )

        if mask is None:
            print("Fundamental matrix computation failed.")
            return None, [], [], [], []

        inlier_indices = np.where(mask.ravel() == 1)[0]
        inlier_points_image_1 = points_image_1[mask.ravel() == 1]
        inlier_points_image_2 = points_image_2[mask.ravel() == 1]
        inlier_descriptors_image_1 = np.array(descriptors_image_1)[inlier_indices]
        inlier_descriptors_image_2 = np.array(descriptors_image_2)[inlier_indices]

        print(
            f"--- Good matches: {len(good)}\n--- Inlier keypoints {len(inlier_points_image_1)}"
        )

        return (
            F,
            inlier_points_image_1,
            inlier_points_image_2,
            inlier_descriptors_image_1,
            inlier_descriptors_image_2,
        )

    # @staticmethod
    # def get_descriptors_for_each_image(pair_manager: PairManager) -> dict[int, np.ndarray]:
    #     """Convert PairMarches to dict of descriptors. Key: img_id, Value: np.ndarray"""
    #     id2descriptors = {}
    #     for pair, matches in pair_manager.pair2matches.items():
    #         img_id_1, img_id_2 = pair
    #         des1 = matches.desc1
    #         des2 = matches.desc2
    #         id2descriptors[img_id_1] = des1
    #         id2descriptors[img_id_2] = des2
    #     return id2descriptors

    # def match_known_images_with_unknown(self, id2orb_features_1: dict[int, list[ORBFeatures]], id2orb_features_2: dict[int, list[ORBFeatures]]):

    #     # reset pair manager
    #     self.pair_manager = PairManager()

    #     ids_1 = list(id2orb_features_1.keys())
    #     ids_2 = list(id2orb_features_2.keys())
    #     for img_id_1 in ids_1:
    #         kpts1, des1 = extract_keypoints_and_descriptors(id2orb_features_1[img_id_1])
    #         for img_id_2 in ids_2:
    #             kpts2, des2 = extract_keypoints_and_descriptors(id2orb_features_2[img_id_2])

    #             if self.method == MatchMethods.KNN_MATCHER_WITH_FUNDAMENTAL_MATRIX:
    #                 f_matrix, matches = self.knn_matcher_with_fundamental_matrix(
    #                     des1, des2, kpts1, kpts2
    #                 )
    #                 if f_matrix.matrix is not None:
    #                     self.pair_manager.add_fundamental_matrix(img_id_1, img_id_2, f_matrix)
    #                     self.pair_manager.add_matches(img_id_1, img_id_2, matches)

    #     return self.pair_manager

    # @staticmethod
    # def brute_force_matcher(des1: np.ndarray, des2: np.ndarray) -> list[cv2.DMatch]:
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(des1, des2)
    #     matches = sorted(matches, key=lambda x: x.distance)
    #     return matches

    # @staticmethod
    # def knn_matcher_with_lowes_ratio(
    #     desc1: np.ndarray, desc2: np.ndarray, normType=cv2.NORM_HAMMING, k: int = 2
    # ) -> list[cv2.DMatch]:
    #     """
    #     link: https://vzat.github.io/comparing_images/week5.html
    #     """
    #     bf = cv2.BFMatcher(normType=normType)
    #     matches = bf.knnMatch(desc1, desc2, k=2)

    #     filtered_counter = 0
    #     correct_matches_counter = 0

    #     # Use D.Lowe's Ratio Test
    #     if k == 2:
    #         bestMatches = []
    #         for match1, match2 in matches:
    #             if match1.distance < 0.75 * match2.distance:
    #                 bestMatches.append(match1)
    #                 correct_matches_counter += 1
    #             else:
    #                 filtered_counter += 1
    #         matches = bestMatches

    #     return matches
