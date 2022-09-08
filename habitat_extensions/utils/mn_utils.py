from typing import List, Union

import magnum as mn
import numpy as np
from scipy.spatial.transform.rotation import Rotation


def to_Quaternion(x: Union[List[float], np.ndarray]):
    """The quaternion format is (x, y, z, w)."""
    assert len(x) == 4, x
    return mn.Quaternion(mn.Vector3(x[0:3]), x[3])


def to_Matrix4(p, q):
    p = mn.Vector3(p)
    q = to_Quaternion(q)
    return mn.Matrix4.from_(q.to_matrix(), p)


def to_list(x):
    if isinstance(x, mn.Vector3):
        return list(x)
    elif isinstance(x, mn.Quaternion):
        return list(x.vector) + [x.scalar]
    else:
        raise TypeError(type(x))


def mat3_to_quat(x: mn.Matrix3, return_list=False):
    quat = mn.Quaternion.from_matrix(x)
    if return_list:
        quat = to_list(quat)
    return quat


def quat_to_list(x: mn.Quaternion, qformat="xyzw"):
    if qformat == "xyzw":
        return list(x.vector) + [x.scalar]
    elif qformat == "wxyz":
        return [x.scalar] + list(x.vector)
    else:
        raise ValueError(qformat)


def mat4_to_pose(T: mn.Matrix4, qformat="xyzw"):
    pos = T.translation
    quat = mn.Quaternion.from_matrix(T.rotation())
    quat = quat_to_list(quat, qformat=qformat)
    return np.hstack([pos, quat])


def orthogonalize(T: mn.Matrix4):
    T_np = np.array(T, dtype=np.float64)
    T_np[:3, :3] = Rotation.from_matrix(T_np[:3, :3]).as_matrix()
    return mn.Matrix4(T_np)
