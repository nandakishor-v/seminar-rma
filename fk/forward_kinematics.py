import numpy as np

def get_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) == 0:
        return np.eye(3)

    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _joint_angle(joint_name, q_dict):
    for key in (joint_name, f"q_{joint_name}", f"{joint_name}_tilt"):
        if key in q_dict:
            return np.deg2rad(q_dict[key])
    return 0.0


def _joint_location(info, q_dict, joint_name):
    loc = np.asarray(info.get("joint_location", info.get("location", [0.0, 0.0, 0.0])), dtype=float)
    translation = np.asarray([
        q_dict.get(f"{joint_name}_tx", 0.0),
        q_dict.get(f"{joint_name}_ty", 0.0),
        q_dict.get(f"{joint_name}_tz", 0.0),
    ], dtype=float)
    return loc + translation


def get_single_joint_transform(joint_name, q_dict, kintree):
    if joint_name not in kintree or joint_name in {"ground", None}:
        return np.eye(3), np.zeros(3)

    info = kintree[joint_name]
    parent_name = info.get("parent", "ground")
    R_parent, p_parent = get_single_joint_transform(parent_name, q_dict, kintree)

    loc = _joint_location(info, q_dict, joint_name)
    angle_val = _joint_angle(joint_name, q_dict)

    axis = np.asarray(info.get("axis", [0.0, 0.0, 1.0]), dtype=float)
    if np.any(axis < 0):
        axis = np.abs(axis)
        angle_val = -angle_val

    R_local = get_rotation_matrix(axis, angle_val)
    return R_parent @ R_local, p_parent + (R_parent @ loc)


def forward_kinematics(q, key, kintree):
    q_dict = dict(zip(key, q))
    joints = {}
    markers = {}

    for joint_name, joint_info in kintree.items():
        R_g, p_g = get_single_joint_transform(joint_name, q_dict, kintree)
        joints[joint_name] = p_g
        for marker_name, marker_pos in joint_info.get("markers", {}).items():
            markers[marker_name] = p_g + (R_g @ np.asarray(marker_pos, dtype=float))

    return joints, markers


def get_connections(kintree, joints):
    """Draws a line between every child and its parent."""
    connections = []
    for child, info in kintree.items():
        parent = info.get('parent')
        if parent in joints and child in joints:
            connections.append((joints[parent].tolist(), joints[child].tolist()))
    return connections