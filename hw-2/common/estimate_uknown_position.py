import cv2
import numpy as np
import tqdm
from common.matchers import ORBMatcher
from common.trajectory import Trajectory
from common.feature_extractor import extract_keypoints_and_descriptors


def estimate_unknown_camera_poses(
    id2orb_features_unknown: dict,
    id2orb_features_known: dict,
    triangulated_points: dict,
    filtered_tracks_known: dict,
    matcher: ORBMatcher,
    camera_matrix: np.ndarray,
    pair2matches_known_images: dict,
):
    """
    Estimate camera poses for unknown images using PnP with pre-computed inlier matches.
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
    for (img_id1, img_id2), matches_data in tqdm.tqdm(
        pair2matches_known_images.items()
    ):
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

    # Rest of the code remains the same...
    print("Estimating camera poses...")
    for unknown_id, unknown_features in tqdm.tqdm(id2orb_features_unknown.items()):
        points_3d = []
        points_2d = []

        unknown_kpts, unknown_desc = extract_keypoints_and_descriptors(unknown_features)

        # Match with pre-computed descriptors
        for desc_tuple, point3d in known_desc_to_3d.items():
            desc = np.array(desc_tuple)
            # Use simple descriptor matching (Hamming distance for ORB)
            distances = np.linalg.norm(unknown_desc - desc, axis=1)
            best_match_idx = np.argmin(distances)
            if distances[best_match_idx] < 300:  # Threshold for matching
                points_3d.append(point3d)
                points_2d.append(unknown_kpts[best_match_idx].pt)

        if len(points_3d) < 6:  # Need at least 6 points for PnP
            print(f"Not enough correspondences for image {unknown_id}")
            continue

        # Convert to numpy arrays
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            camera_matrix,  # You'll need to pass the intrinsic matrix
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            print(f"Estimated pose for image {unknown_id}")
            rmat, _ = cv2.Rodrigues(rvec)
            transformation_matrix = np.hstack((rmat, tvec.reshape(3, 1)))
            camera_poses[unknown_id] = np.vstack(
                (transformation_matrix, np.array([0, 0, 0, 1]))
            )
        else:
            print(
                f"Pose estimation failed for image {unknown_id}, using default matrix."
            )
            default_matrix = np.ones((4, 4))
            default_matrix[3, :] = [0, 0, 0, 1]
            camera_poses[unknown_id] = default_matrix

    return camera_poses
