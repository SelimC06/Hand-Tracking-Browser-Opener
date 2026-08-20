import numpy as np

# MediaPipe hand landmark indices, chained per finger from the wrist.
FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def _to_points(raw_vector):
    """42-length flat [x0,y0,x1,y1,...,x20,y20] -> (21,2) array."""
    arr = np.asarray(raw_vector, dtype=float).reshape(21, 2)
    return arr


def raw_coordinates(raw_vector):
    """(a) Current representation: raw x,y coordinates, unchanged."""
    return np.asarray(raw_vector, dtype=float)


def wrist_relative_scaled(raw_vector):
    """(b) Coordinates relative to the wrist, scaled by hand bounding-box size.

    Translation-invariant (wrist becomes the origin) and scale-invariant
    (divided by the larger of the bbox width/height), so moving the hand
    across the frame or changing camera distance should not shift the vector
    for an identical gesture.
    """
    points = _to_points(raw_vector)
    wrist = points[WRIST_IDX]
    relative = points - wrist
    bbox_w = points[:, 0].max() - points[:, 0].min()
    bbox_h = points[:, 1].max() - points[:, 1].min()
    scale = max(bbox_w, bbox_h)
    if scale <= 1e-9:
        scale = 1.0
    scaled = relative / scale
    return scaled.reshape(-1)


def _angle(u, v):
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm <= 1e-9:
        return 0.0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def inter_landmark_angles(raw_vector):
    """(c) Inter-landmark angles: for each finger, the angle of the base
    segment relative to the palm reference (wrist -> middle MCP), followed by
    the three joint-bend angles along that finger. 5 fingers x 4 angles = 20
    features. Rotation-invariant in addition to translation/scale-invariant,
    since only relative angles between vectors are used.
    """
    points = _to_points(raw_vector)
    wrist = points[WRIST_IDX]
    palm_ref = points[MIDDLE_MCP_IDX] - wrist

    angles = []
    for chain in FINGER_CHAINS.values():
        chain_points = [points[idx] for idx in chain]
        segments = [chain_points[i + 1] - chain_points[i] for i in range(len(chain_points) - 1)]
        angles.append(_angle(segments[0], palm_ref))
        for i in range(1, len(segments)):
            angles.append(_angle(segments[i], segments[i - 1]))
    return np.asarray(angles, dtype=float)


REPRESENTATIONS = {
    "raw_coordinates": raw_coordinates,
    "wrist_relative_scaled": wrist_relative_scaled,
    "inter_landmark_angles": inter_landmark_angles,
}
