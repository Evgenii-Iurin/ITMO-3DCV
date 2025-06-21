import numpy as np
from common.intrinsics import Intrinsics
from common.trajectory import Trajectory


class CameraParams:
    def __init__(self, path_to_intrinsic_parameter: str):
        self.K = Intrinsics.read(path_to_intrinsic_parameter)

    def get_intrinsic_matrix(self):
        return np.array(
            ([self.K.fx, 0, self.K.cx], [0, self.K.fy, self.K.cy], [0, 0, 1])
        )

    def get_camera_params(self, extrinsics: np.ndarray):
        """ "Return 3x4 camera matrix P"""
        K = self.get_intrinsic_matrix()

        P = np.dot(K, extrinsics[:3, :])
        return P

    def get_camera_params_for_cameras(self, id2poses: dict[int, np.ndarray]):
        """Calculate the camera parameters for each camera in the dict

        id2poses: {img_id: [t, t, t, q, q, q, q]}
        """
        id2camera_params = {}
        for img_id, extrinsic in id2poses.items():
            matrix = Trajectory.to_matrix4(extrinsic)
            camera_params = self.get_camera_params(matrix)
            id2camera_params[img_id] = camera_params
        return id2camera_params
