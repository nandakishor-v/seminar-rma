import torch
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers (differentiable PyTorch)
# ---------------------------------------------------------------------------

def get_rotation_matrix(axis, angle):
    """3x3 rotation matrix via Rodrigues' formula -- fully differentiable.

    zero = kx * 0 keeps the zero in the computation graph (critical for autograd).
    """
    axis  = torch.as_tensor(axis,  dtype=torch.float32)
    angle = torch.as_tensor(angle, dtype=torch.float32)

    norm = torch.linalg.norm(axis)
    if norm < 1e-10:
        return torch.eye(3, dtype=torch.float32)

    axis = axis / norm
    kx, ky, kz = axis[0], axis[1], axis[2]

    zero = kx * 0  # stays in computation graph!

    row1 = torch.stack([ zero,  -kz,   ky])
    row2 = torch.stack([ kz,    zero, -kx])
    row3 = torch.stack([-ky,    kx,   zero])
    K = torch.stack([row1, row2, row3])

    I = torch.eye(3, dtype=torch.float32)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(q, key, kintree):
    """Compute forward kinematics using PyTorch for differentiability (IK-ready).

    Key design decisions:
    - Body quaternions (orientation field) are IGNORED -- they are a static
      OpenSim->MuJoCo coordinate frame artifact, not joint motion.
    - Joint axes are applied in the parent frame, consistent with the
      OpenSim-derived angle data in angles_clean.csv.
    - CSV column names like 'q_hip_r' are normalized to 'hip_r' to match
      the XML joint names (q_ prefix stripped).

    Args:
        q (ndarray | list | pd.Series): Generalized coordinates of shape (N,).
        key (list): Joint names corresponding to angles in q.
        kintree (dict): Nested kinematic tree from get_model_dictionary().

    Returns:
        joints  (dict): {body_name:   torch.Tensor(3,)} global positions.
        markers (dict): {marker_name: torch.Tensor(3,)} global positions.
    """
    if isinstance(q, pd.Series):
        q = q.tolist()

    # Build angle lookup -- strip 'q_' prefix to match XML joint names
    # e.g. 'q_hip_r' -> 'hip_r', 'pelvis_tx' -> 'pelvis_tx' (unchanged)
    q_dict = {}
    for name, val in zip(key, q):
        normalized = name[2:] if name.startswith('q_') else name
        if isinstance(val, torch.Tensor):
            q_dict[normalized] = val.float()
        else:
            q_dict[normalized] = torch.tensor(float(val), dtype=torch.float32)

    joints  = {}
    markers = {}

    def _traverse(body_dict, R_parent, p_parent):
        for body_name, body_info in body_dict.items():

            # 1. Body origin: parent_pos + parent_R @ joint_location
            joint_location = torch.as_tensor(
                body_info.get('joint_location', np.zeros(3)), dtype=torch.float32)
            p_current = p_parent + R_parent @ joint_location

            # 2. Start rotation from parent -- body quaternion intentionally skipped
            #    (it is a static OpenSim->MuJoCo coordinate frame artifact)
            R_current = R_parent.clone()

            # 3. Apply each joint: positional offset + rotation or translation
            for joint_name, joint_info in body_info['joints'].items():

                # Joint pos offset within the body frame
                joint_pos = torch.as_tensor(
                    joint_info.get('pos', np.zeros(3)), dtype=torch.float32)
                p_current = p_current + R_current @ joint_pos

                angle = q_dict.get(joint_name, torch.tensor(0.0))
                axis  = torch.as_tensor(
                    joint_info.get('axis', np.array([0., 0., 1.])), dtype=torch.float32)
                jtype = joint_info.get('type', 'hinge')

                # Flip angle for negative-dominant axes (OpenSim convention)
                dominant = torch.argmax(torch.abs(axis)).item()
                if axis[dominant] < 0:
                    axis  = -axis
                    angle = -angle

                if jtype == 'hinge':
                    R_current = R_current @ get_rotation_matrix(axis, angle)

                elif jtype == 'slider':
                    norm = torch.linalg.norm(axis)
                    if norm > 1e-10:
                        axis = axis / norm
                    p_current = p_current + R_current @ (axis * angle)

            # 4. Record body global position
            joints[body_name] = p_current

            # 5. Transform markers into global frame
            for marker_name, marker_local in body_info.get('markers', {}).items():
                m = torch.as_tensor(marker_local, dtype=torch.float32)
                markers[marker_name] = p_current + R_current @ m

            # 6. Recurse into children
            _traverse(body_info.get('children', {}), R_current, p_current)

    _traverse(kintree, torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))

    return joints, markers


# ---------------------------------------------------------------------------
# Get connections for skeleton visualization
# ---------------------------------------------------------------------------

def get_connections(kintree, joints):
    """Get joint-to-joint connections for skeleton visualization.

    Returns:
        connections: list of ([x1,y1,z1], [x2,y2,z2]) tuples -- one per bone.
    """
    connections = []

    def _recurse(body_dict):
        for body_name, body_info in body_dict.items():
            parent_name = body_info.get('parent')
            if parent_name in joints and body_name in joints:
                p = joints[parent_name]
                c = joints[body_name]
                connections.append((
                    p.tolist() if isinstance(p, torch.Tensor) else list(p),
                    c.tolist() if isinstance(c, torch.Tensor) else list(c)
                ))
            _recurse(body_info.get('children', {}))

    _recurse(kintree)
    return connections