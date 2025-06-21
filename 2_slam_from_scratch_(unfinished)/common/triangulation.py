import numpy as np


class Triangulation:
    @staticmethod
    def get_A_for_3D_point(point: tuple[int, int], camera_parameters: np.ndarray):
        x, y = point
        # Normalize coordinates for better numerical stability
        A = np.array(
            [
                x * camera_parameters[2, :] - camera_parameters[0, :],
                y * camera_parameters[2, :] - camera_parameters[1, :],
            ]
        )
        return A / np.linalg.norm(A)  # Normalize matrix rows

    @staticmethod
    def resolve_3D_point(A: np.ndarray):
        U, S, Vt = np.linalg.svd(A)

        # Check condition number for numerical stability
        condition_number = S[0] / S[-1]
        if condition_number > 1e8:  # Threshold for ill-conditioned system
            return None

        X = Vt[-1]
        if X[3] == 0:  # Check for points at infinity
            return None

        # normalize to X = (X, Y, Z, 1)
        X /= X[3]

        # Check if point is in front of cameras
        if X[2] < 0:
            return None

        return X[:3]

    @staticmethod
    def triangulate_3D_point_from_tracks(
        tracks: dict[int, list[tuple[int, np.ndarray]]], id2camera_params
    ):
        """
        Triangulate 3D points from multiple views.

        Args:
            tracks: Dictionary mapping track_id to list of (img_id, point) pairs
            id2camera_params: Dictionary mapping image_id to camera matrix

        Returns:
            Dictionary mapping track_id to 3D point
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

    @staticmethod
    def _verify_point_using_reprojection_error(track, point3d, id2camera_params):
        _PXL_THRESHOLD = 15.0  # Reduced from 25.0
        _MAX_ERROR_MULTIPLIER = 2.0  # Allow some outliers
        errors = []

        for img_id, pt in track:
            camera_parameters = id2camera_params[img_id]
            pt_reprojected = camera_parameters @ np.append(point3d, 1)
            if pt_reprojected[2] <= 0:  # Check for points behind camera
                return False
            pt_reprojected = pt_reprojected[:2] / pt_reprojected[2]
            error = np.linalg.norm(pt - pt_reprojected)
            errors.append(error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        return (
            avg_error < _PXL_THRESHOLD
            and max_error < _PXL_THRESHOLD * _MAX_ERROR_MULTIPLIER
        )
