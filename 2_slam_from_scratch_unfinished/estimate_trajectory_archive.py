from common.build_tracks import TrackManager
from common.dataset import Dataset
from common.feature_extractor import extract_orb_features
from common.matchers import ORBMatcher
from common.trajectory import Trajectory
from common.triangulation import Triangulation
from common.camera_params import CameraParams
from common.estimate_uknown_position import estimate_unknown_camera_poses
import numpy as np


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

    # with open("./cache_data/tracks_only_known.pkl", "wb") as f:
    #     pickle.dump(filtered_tracks_for_known_images, f)

    # with open("./cache_data/tracks_only_known.pkl", "rb") as f:
    #     filtered_tracks_for_known_images = pickle.load(f)

    id2camera_params_for_known_images = camera_params.get_camera_params_for_cameras(
        id2known_poses
    )

    print("Start triangulation...")
    triangulation_known_images = Triangulation.triangulate_3D_point_from_tracks(
        filtered_tracks_for_known_images, id2camera_params_for_known_images
    )

    # with open("./cache_data/triangulated_points_only_known.pkl", "wb") as f:
    #     pickle.dump(triangulation_known_images, f)

    # with open("./cache_data/triangulated_points_only_known.pkl", "rb") as f:
    #     triangulation_known_images = pickle.load(f)

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
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
