import numpy as np
from collections import defaultdict
from tqdm import tqdm


class DisjointSet:
    """Disjoint Set (Union-Find) to efficiently merge tracks."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        """Find the root of the set containing x. Uses path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Merge two sets (tracks)."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x  # Merge y's root into x's root


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
        for track_id, connections in tqdm(track_groups.items()):
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
