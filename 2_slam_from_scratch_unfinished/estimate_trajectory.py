import numpy as np
import os
import cv2
from collections import defaultdict

from common.dataset import Dataset
from common.trajectory import Trajectory

# DisjointSet from build_tracks.py
class DisjointSet:
    """Disjoint Set (Union-Find) to efficiently merge tracks."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


# TrackManager from build_tracks.py
class TrackManager:
    def __init__(self):
        self.track_id_counter = 0
        self.tracks = {}

    def calculate_tracks(self, pair2matches):
        """
        Build tracks of 2D points across multiple images using a graph-based approach with Union-Find.
        """
        # Disjoint Set to track point-group relationships
        ds = DisjointSet()
        point_mapping = {}

        # First pass: build disjoint sets (connect points)
        for (img_id1, img_id2), matches in pair2matches.items():
            points1 = matches[img_id1]["points"]
            points2 = matches[img_id2]["points"]

            for pt1, pt2 in zip(points1, points2):
                pt1_tuple = tuple(pt1)
                pt2_tuple = tuple(pt2)

                # Assign a unique ID to each point
                if (img_id1, pt1_tuple) not in point_mapping:
                    point_mapping[(img_id1, pt1_tuple)] = len(point_mapping)
                    ds.parent[point_mapping[(img_id1, pt1_tuple)]] = point_mapping[
                        (img_id1, pt1_tuple)
                    ]

                if (img_id2, pt2_tuple) not in point_mapping:
                    point_mapping[(img_id2, pt2_tuple)] = len(point_mapping)
                    ds.parent[point_mapping[(img_id2, pt2_tuple)]] = point_mapping[
                        (img_id2, pt2_tuple)
                    ]

                # Merge the two points into the same track
                ds.union(
                    point_mapping[(img_id1, pt1_tuple)],
                    point_mapping[(img_id2, pt2_tuple)],
                )

        # Second pass: Group points by connected components
        track_groups = defaultdict(set)
        for (img_id, pt_tuple), pt_id in point_mapping.items():
            root_id = ds.find(pt_id)  # Find the representative track ID
            track_groups[root_id].add((img_id, pt_tuple))

        # Third pass: Build final track dictionary
        self.tracks = {}
        for track_id, connections in track_groups.items():
            track = [(img_id, np.array(pt)) for img_id, pt in connections]
            if len(track) >= 2:
                self.tracks[self.track_id_counter] = track
                self.track_id_counter += 1

        return self.tracks

    def get_filtered_tracks(self, min_track_length: int = 2):
        """
        Filter tracks to keep only those that appear in at least min_track_length images.
        """
        filtered_tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if len(track) >= min_track_length
        }
        print(f"Filtered tracks: {len(filtered_tracks)} out of {len(self.tracks)}")
        return filtered_tracks

    def merge_tracks(self, tracks):
        merged = {}
        for t1_id, track1 in tracks.items():
            points1 = set((img_id, tuple(pt)) for img_id, pt in track1)

            # Find all tracks that share points with this one
            for t2_id, track2 in tracks.items():
                if t1_id == t2_id:
                    continue
                points2 = set((img_id, tuple(pt)) for img_id, pt in track2)
                if points1 & points2:  # If they share points
                    points1.update(points2)  # Merge points

            if len(points1) >= 2:
                merged[len(merged)] = list(points1)
        return merged


# CameraParams class
class CameraParams:
    def __init__(self, path_to_intrinsic_parameter):
        with open(path_to_intrinsic_parameter, "r") as f:
            lines = f.readlines()
            if lines[0].startswith("#"):
                lines = lines[1:]
            tokens = lines[0].strip().split()
            self.fx = float(tokens[0])
            self.fy = float(tokens[1])
            self.cx = float(tokens[2])
            self.cy = float(tokens[3])

    def get_intrinsic_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def get_camera_params_for_cameras(self, id2camera_poses):
        id2camera_params = {}
        for img_id, pose in id2camera_poses.items():
            tx, ty, tz, qx, qy, qz, qw = [float(x) for x in pose]

            # Create rotation matrix from quaternion
            q = np.array([qw, qx, qy, qz])
            q = q / np.linalg.norm(q)  # Normalize quaternion

            # Convert quaternion to rotation matrix
            rot = self._quaternion_to_rotation_matrix(q)

            # Create translation vector
            t = np.array([tx, ty, tz])

            # Create full camera matrix [R|t]
            extrinsic = np.hstack((rot, t.reshape(3, 1)))
            extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1])))

            # Camera matrix = K[R|t]
            camera_matrix = self.get_intrinsic_matrix() @ extrinsic[:3, :]
            id2camera_params[img_id] = camera_matrix

        return id2camera_params

    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )


# Triangulation class
class Triangulation:
    @staticmethod
    def get_A_for_3D_point(point, camera_parameters):
        """Compute matrix A for solving Ax = 0 for 3D point x."""
        x, y = point
        P = camera_parameters

        A = np.array(
            [
                [
                    x * P[2, 0] - P[0, 0],
                    x * P[2, 1] - P[0, 1],
                    x * P[2, 2] - P[0, 2],
                    x * P[2, 3] - P[0, 3],
                ],
                [
                    y * P[2, 0] - P[1, 0],
                    y * P[2, 1] - P[1, 1],
                    y * P[2, 2] - P[1, 2],
                    y * P[2, 3] - P[1, 3],
                ],
            ]
        )

        return A

    @staticmethod
    def resolve_3D_point(A):
        """Compute 3D point from matrix A."""
        _, _, Vh = np.linalg.svd(A)
        point3d_homogeneous = Vh[-1, :]

        # Convert from homogeneous to Euclidean coordinates
        point3d = point3d_homogeneous[:3] / point3d_homogeneous[3]

        # Check condition number
        condition_number = np.linalg.cond(A)
        if condition_number > 1e8:
            return None

        return point3d

    @staticmethod
    def _verify_point_using_reprojection_error(track, point3d, id2camera_params):
        _PXL_THRESHOLD = 15.0

        for img_id, pt in track:
            camera_parameters = id2camera_params[img_id]
            pt_homogeneous = np.append(point3d, 1)
            pt_reprojected = camera_parameters @ pt_homogeneous

            if pt_reprojected[2] <= 0:  # Check if point is in front of camera
                return False

            pt_reprojected = pt_reprojected[:2] / pt_reprojected[2]
            error = np.linalg.norm(pt - pt_reprojected)

            if error > _PXL_THRESHOLD:
                return False

        return True

    @staticmethod
    def triangulate_3D_point_from_tracks(tracks, id2camera_params):
        """
        Triangulate 3D points from multiple views.
        """
        filtered_tracks_counter = 0
        found_points_counter = 0
        trackId2Points3D = {}

        for track_id, track in tracks.items():
            if len(track) < 2:  # Need at least 2 views
                filtered_tracks_counter += 1
                continue

            A = np.array([])
            for img_id, pt in track:
                camera_parameters = id2camera_params[img_id]
                A_mat = Triangulation.get_A_for_3D_point(pt, camera_parameters)
                if A.size == 0:
                    A = A_mat
                else:
                    A = np.vstack((A, A_mat))

            point3d = Triangulation.resolve_3D_point(A)

            if (
                point3d is not None
                and Triangulation._verify_point_using_reprojection_error(
                    track, point3d, id2camera_params
                )
            ):
                trackId2Points3D[track_id] = point3d
                found_points_counter += 1
            else:
                filtered_tracks_counter += 1

        print(f"Triangulation finished. Found {len(trackId2Points3D)} 3D points.")
        print(f"Filtered {filtered_tracks_counter} tracks.")

        return trackId2Points3D


# ORB Matcher from matchers.py
class ORBMatcher:
    def __init__(self):
        pass

    def match_all_known_images(self, id2orb_features):
        """Match all known images with each other."""
        pair2matches = {}
        ids = list(id2orb_features.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                img_id1 = ids[i]
                img_id2 = ids[j]

                kpts1, desc1 = extract_keypoints_and_descriptors(
                    id2orb_features[img_id1]
                )
                kpts2, desc2 = extract_keypoints_and_descriptors(
                    id2orb_features[img_id2]
                )

                (
                    F,
                    pts1,
                    pts2,
                    descs1,
                    descs2,
                ) = self.knn_matcher_with_fundamental_matrix(desc1, desc2, kpts1, kpts2)

                if F is not None:
                    pair2matches[(img_id1, img_id2)] = {
                        img_id1: {"points": pts1, "descriptors": descs1},
                        img_id2: {"points": pts2, "descriptors": descs2},
                        "fundamental_matrix": F,
                    }

        return pair2matches

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
        inlier_points_image_1 = points_image_1[inlier_indices]
        inlier_points_image_2 = points_image_2[inlier_indices]
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


# Feature extraction
def extract_keypoints_and_descriptors(orb_features):
    features = orb_features.features
    keypoints = [cv2.KeyPoint(x=f.pt[0], y=f.pt[1], size=f.size) for f in features]
    descriptors = np.array([f.descriptor for f in features], dtype=np.uint8)
    return keypoints, descriptors


def extract_orb_features(id2img_path, data_dir):
    result = {}
    for img_id, img_path in id2img_path.items():
        full_path = os.path.join(data_dir, img_path[0])
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Create ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Create ORB features
        features = []
        if keypoints is None or descriptors is None:
            print(f"Error extracting features for image {img_id}")
            continue

        for kp, desc in zip(keypoints, descriptors):
            features.append(
                {
                    "pt": kp.pt,
                    "size": kp.size,
                    "angle": kp.angle,
                    "response": kp.response,
                    "descriptor": desc,
                }
            )

        # Store features
        result[img_id] = {"features": features}

    return result


# Estimate Unknown Position
def estimate_unknown_camera_poses(
    id2orb_features_unknown,
    id2orb_features_known,
    triangulated_points,
    filtered_tracks_known,
    matcher,
    camera_matrix,
    pair2matches_known_images,
):
    """
    Estimate camera poses for unknown images using PnP.
    """
    camera_poses = {}

    # Create a faster lookup structure
    print("Building descriptor lookup...")
    point_to_track = {}
    track_to_3d = {}

    # First, create mapping from (img_id, point) to track_id
    for track_id, track in filtered_tracks_known.items():
        if track_id in triangulated_points:
            for img_id, pt in track:
                point_to_track[(img_id, tuple(pt))] = track_id
            track_to_3d[track_id] = triangulated_points[track_id]

    # Now create descriptor to 3D point mapping
    known_desc_to_3d = {}
    for (img_id1, img_id2), matches_data in pair2matches_known_images.items():
        for img_id in [img_id1, img_id2]:
            if img_id in matches_data:
                points = matches_data[img_id]["points"]
                descriptors = matches_data[img_id]["descriptors"]
                for pt, desc in zip(points, descriptors):
                    key = (img_id, tuple(pt))
                    if key in point_to_track:
                        track_id = point_to_track[key]
                        if track_id in track_to_3d:
                            known_desc_to_3d[tuple(desc)] = track_to_3d[track_id]

    # Match features and estimate poses
    print("Estimating camera poses...")
    for unknown_id, unknown_features in id2orb_features_unknown.items():
        points_3d = []
        points_2d = []

        unknown_kpts, unknown_desc = extract_keypoints_and_descriptors(unknown_features)

        # Match with pre-computed descriptors
        for desc_tuple, point3d in known_desc_to_3d.items():
            desc = np.array(desc_tuple)
            # Use Hamming distance for ORB descriptors
            distances = cv2.norm(
                np.uint8(unknown_desc), np.uint8(desc.reshape(1, -1)), cv2.NORM_HAMMING
            )
            best_match_idx = np.argmin(distances)
            if distances[best_match_idx] < 50:  # Threshold for Hamming distance
                points_3d.append(point3d)
                points_2d.append(unknown_kpts[best_match_idx].pt)

        if len(points_3d) < 6:  # Need at least 6 points for robust PnP
            print(f"Not enough correspondences for image {unknown_id}")
            continue

        # Convert to numpy arrays
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            None,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success and inliers is not None and len(inliers) >= 3:
            print(f"Estimated pose for image {unknown_id} with {len(inliers)} inliers")
            # Convert rotation vector to matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Create 4x4 transformation matrix as numpy array
            transform_matrix = np.array(
                [
                    [rmat[0, 0], rmat[0, 1], rmat[0, 2], tvec[0, 0]],
                    [rmat[1, 0], rmat[1, 1], rmat[1, 2], tvec[1, 0]],
                    [rmat[2, 0], rmat[2, 1], rmat[2, 2], tvec[2, 0]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            camera_poses[unknown_id] = transform_matrix
        else:
            print(f"Pose estimation failed for image {unknown_id}")

    return camera_poses


# Now define the main functions from estimate_trajectory.py
def get_trajectories(data_dir):
    rgb_txt = data_dir + "/rgb.txt"
    known_poses_txt = data_dir + "/known_poses.txt"
    intrinsics_txt = data_dir + "/intrinsics.txt"

    dataset = Dataset()
    matcher = ORBMatcher()
    track_manager = TrackManager()
    camera_params = CameraParams(intrinsics_txt)

    id2known_poses = dataset.read_dict_of_lists(known_poses_txt)
    for key in id2known_poses:
        id2known_poses[key] = [float(x) for x in id2known_poses[key]]

    id2img_path_all = dataset.read_dict_of_lists(rgb_txt)
    id2img_path_known = {
        k: id2img_path_all[k] for k in id2img_path_all if k in id2known_poses
    }
    id2img_path_unknown = {
        k: id2img_path_all[k] for k in id2img_path_all if k not in id2known_poses
    }

    id2orb_features_for_known_images = extract_orb_features(
        id2img_path_known, data_dir=data_dir
    )
    pair2matches_known_images = matcher.match_all_known_images(
        id2orb_features_for_known_images
    )

    track_manager.calculate_tracks(pair2matches_known_images)
    filtered_tracks_for_known_images = track_manager.get_filtered_tracks()

    id2camera_params_for_known_images = camera_params.get_camera_params_for_cameras(
        id2known_poses
    )

    print("Start triangulation...")
    triangulation_known_images = Triangulation.triangulate_3D_point_from_tracks(
        filtered_tracks_for_known_images, id2camera_params_for_known_images
    )

    id2orb_features_for_uknown_cameras = extract_orb_features(
        id2img_path_unknown, data_dir=data_dir
    )

    print("Estimating unknown camera poses...")
    camera_poses = estimate_unknown_camera_poses(
        id2orb_features_for_uknown_cameras,
        id2orb_features_for_known_images,
        triangulation_known_images,
        filtered_tracks_for_known_images,
        matcher,
        camera_params.get_intrinsic_matrix(),
        pair2matches_known_images,
    )

    trajectory = {}
    for frame_id, pose in camera_poses.items():
        trajectory[frame_id] = np.array(pose, dtype=np.float64)

    return trajectory


def estimate_trajectory(data_dir, out_dir):
    trajectory = get_trajectories(data_dir)

    fixed_trajectory = {}
    for frame_id, pose in trajectory.items():
        fixed_trajectory[frame_id] = np.array(pose, dtype=np.float64)

    Trajectory.write(Dataset.get_result_poses_file(out_dir), fixed_trajectory)
