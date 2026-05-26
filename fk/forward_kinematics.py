import torch
import pandas as pd

def get_rotation_matrix(axis, angle):
    """Generates a 3x3 rotation matrix using PyTorch and Rodrigues' formula."""
    # Use as_tensor to prevent copying existing tensors (Fixes Pytest Warnings)
    axis = torch.as_tensor(axis, dtype=torch.float32)
    angle = torch.as_tensor(angle, dtype=torch.float32)

    norm = torch.linalg.norm(axis)
    if norm == 0:
        return torch.eye(3, dtype=torch.float32)

    axis = axis / norm
    kx, ky, kz = axis[0], axis[1], axis[2]

    # Skew-symmetric matrix
    K = torch.tensor([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ], dtype=torch.float32)

    I = torch.eye(3, dtype=torch.float32)
    
    # Rodrigues' rotation formula using PyTorch
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    return R

def _joint_angle(joint_name, q_dict):
    """Safely extracts the angle and converts degrees to radians using PyTorch."""
    for key in (joint_name, f"q_{joint_name}", f"{joint_name}_tilt"):
        if key in q_dict:
            val = q_dict[key]
            # Convert to float tensor and then to radians
            return torch.deg2rad(torch.tensor(val, dtype=torch.float32))
    return torch.tensor(0.0, dtype=torch.float32)

def _joint_location(info, q_dict, joint_name):
    """Calculates local bone offset using PyTorch tensors."""
    loc = info.get("joint_location", info.get("location", [0.0, 0.0, 0.0]))
    loc = torch.tensor(loc, dtype=torch.float32)

    # Grab dynamic translation
    tx = q_dict.get(f"{joint_name}_tx", 0.0)
    ty = q_dict.get(f"{joint_name}_ty", 0.0)
    tz = q_dict.get(f"{joint_name}_tz", 0.0)

    translation = torch.tensor([tx, ty, tz], dtype=torch.float32)
    return loc + translation

def get_single_joint_transform(joint_name, q_dict, kintree):
    """Recursively calculates global transforms using PyTorch matrix multiplication."""
    if joint_name not in kintree or joint_name in {"ground", None}:
        return torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)

    info = kintree[joint_name]
    parent_name = info.get("parent", "ground")
    
    # Recurse to find parent
    R_parent, p_parent = get_single_joint_transform(parent_name, q_dict, kintree)

    # Get local properties
    loc = _joint_location(info, q_dict, joint_name)
    angle_val = _joint_angle(joint_name, q_dict)

    axis = torch.tensor(info.get("axis", [0.0, 0.0, 1.0]), dtype=torch.float32)
    
    if torch.any(axis < 0):
        axis = torch.abs(axis)
        angle_val = -angle_val

    R_local = get_rotation_matrix(axis, angle_val)

    # PyTorch matrix multiplication: @ or torch.matmul
    p_global = p_parent + torch.matmul(R_parent, loc)
    R_global = torch.matmul(R_parent, R_local)

    return R_global, p_global

def forward_kinematics(q, key, kintree):
    """Main loop. Ensures PyTorch tensors are used throughout."""
    # Ensure q is a flat list/array, especially if it came from a pandas DataFrame
    if isinstance(q, pd.Series):
        q = q.tolist()
        
    q_dict = dict(zip(key, q))
    joints = {}
    markers = {}

    for joint_name, joint_info in kintree.items():
        R_g, p_g = get_single_joint_transform(joint_name, q_dict, kintree)
        
        # Save as PyTorch tensor
        joints[joint_name] = p_g
        
        for marker_name, marker_pos in joint_info.get("markers", {}).items():
            m_pos = torch.tensor(marker_pos, dtype=torch.float32)
            markers[marker_name] = p_g + torch.matmul(R_g, m_pos)

    return joints, markers

def get_connections(kintree, joints):
    """Converts the PyTorch tensors back to Python lists so Matplotlib can plot them."""
    connections = []
    for child, info in kintree.items():
        parent = info.get('parent')
        if parent in joints and child in joints:
            # .tolist() is critical here so Matplotlib doesn't crash on PyTorch objects
            connections.append((joints[parent].tolist(), joints[child].tolist()))
    return connections