import numpy as np

from scipy.spatial.transform import Rotation as R
import numpy as np


def pose_to_matrix(pos, quat):
    """
    pos: (3,)
    quat: (4,)  [x, y, z, w]  (Isaac/PhysX 标准)
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def transform_points(T, pts):
    """
    T: (4,4) transform matrix
    pts: (N,3)
    """
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    pts_t = (T @ pts_h.T).T
    return pts_t[:, :3]
