import numpy as np
import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_experimental_markers(marker_row):
    """Parse experimental marker data into {marker_name: np.array(3,)}.

    Handles two formats:
    1. Flat dict: {'marker_test_x': 1.0, 'marker_test_y': -2.0, 'marker_test_z': 3.0}
    2. pandas Series with 'marker_marker_hip_r_x/y/z' column naming
    """
    if isinstance(marker_row, dict):
        # Try to group _x/_y/_z keys into 3D positions
        marker_map = {}
        keys = list(marker_row.keys())
        for k in keys:
            if not k.endswith('_x'):
                continue
            base = k[:-2]
            ky, kz = f"{base}_y", f"{base}_z"
            if ky in marker_row and kz in marker_row:
                marker_map[base] = np.array([
                    float(marker_row[k]),
                    float(marker_row[ky]),
                    float(marker_row[kz])
                ])
        return marker_map if marker_map else {k: np.asarray(v, dtype=float)
                                               for k, v in marker_row.items()}

    # pandas Series
    if hasattr(marker_row, 'index'):
        index = [str(c) for c in marker_row.index]
        values = marker_row.to_numpy(dtype=float)
        marker_map = {}
        for i in range(len(index) - 2):
            col = index[i]
            if not col.endswith('_x'):
                continue
            base = col[:-2]
            expected = [f"{base}_x", f"{base}_y", f"{base}_z"]
            if index[i:i+3] != expected:
                continue
            marker_map[base] = values[i:i+3]
        return marker_map
    return {}


def _lookup_marker(exp_map, fk_name):
    """Match an FK marker name to an experimental marker name."""
    bare = fk_name.removeprefix('marker_')
    candidates = [
        fk_name,              # 'marker_hip_r'
        f"marker_{fk_name}",  # 'marker_marker_hip_r'
        bare,                 # 'hip_r'
        f"marker_{bare}",     # 'marker_hip_r'
    ]
    for c in candidates:
        if c in exp_map:
            return exp_map[c]
    return None


def _to_torch_pos(pos):
    """Convert FK marker position to a torch tensor, preserving grad graph.

    Handles:
    - torch.Tensor (already in graph)
    - plain list of torch scalars: [q[0], q[1], q[2]]  <- test case
    - numpy array / plain list of floats
    """
    if isinstance(pos, torch.Tensor):
        return pos

    # List that may contain torch scalars (e.g. from _simple_marker_fk)
    if isinstance(pos, (list, tuple)):
        if len(pos) > 0 and isinstance(pos[0], torch.Tensor):
            return torch.stack([p if isinstance(p, torch.Tensor)
                                else torch.tensor(float(p)) for p in pos])
        return torch.tensor(np.asarray(pos, dtype=np.float32))

    return torch.tensor(np.asarray(pos, dtype=np.float32))


# ---------------------------------------------------------------------------
# IK target function
# ---------------------------------------------------------------------------

def ik_target_function(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute the IK loss: MSE between FK markers and experimental markers.

    Args:
        fk_function: forward_kinematics(q, key, kintree) -> (joints, markers)
        q (np.ndarray | torch.Tensor): Joint angles, shape (N,).
        key (list): Joint names for q.
        kintree (dict): Kinematic tree.
        marker_positions: pandas Series or dict of experimental markers.
        bounds (tuple): (lower, upper) np.ndarrays for joint limits.

    Returns:
        torch.Tensor: Scalar loss (differentiable).
    """
    if isinstance(q, np.ndarray):
        q_t = torch.tensor(q.astype(np.float32), requires_grad=True)
    elif isinstance(q, torch.Tensor):
        q_t = q.float()
    else:
        q_t = torch.tensor(list(q), dtype=torch.float32)

    exp_map = _extract_experimental_markers(marker_positions)

    _, fk_markers = fk_function(q_t, key, kintree)

    sq_errors = []
    for fk_name, fk_pos in fk_markers.items():
        exp_pos = _lookup_marker(exp_map, fk_name)
        if exp_pos is None:
            continue

        # Convert FK pos -- handles plain lists of torch scalars (keeps grad graph)
        fk_t  = _to_torch_pos(fk_pos)
        exp_t = torch.tensor(np.asarray(exp_pos, dtype=np.float32))
        sq_errors.append(torch.sum((fk_t - exp_t) ** 2))

    if len(sq_errors) == 0:
        return torch.tensor(0.0)

    loss = torch.mean(torch.stack(sq_errors))

    # Soft joint limit penalty
    if bounds is not None:
        lower = torch.tensor(np.asarray(bounds[0], dtype=np.float32))
        upper = torch.tensor(np.asarray(bounds[1], dtype=np.float32))
        penalty = (torch.relu(lower - q_t) ** 2 + torch.relu(q_t - upper) ** 2).mean()
        loss = loss + 0.01 * penalty

    return loss


# ---------------------------------------------------------------------------
# Gradient via autograd
# ---------------------------------------------------------------------------

def compute_ik_gradient(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute dL/dq via PyTorch autograd.

    Returns:
        np.ndarray: Gradient of shape (N,).
    """
    q_t = torch.tensor(np.asarray(q, dtype=np.float32), requires_grad=True)

    loss = ik_target_function(fk_function, q_t, key, kintree, marker_positions, bounds)
    loss.backward()

    if q_t.grad is None:
        # Graph was broken (e.g. FK returned plain numpy) -- use finite differences
        q_np = np.asarray(q, dtype=float)
        eps  = 1e-5
        grad = np.zeros_like(q_np)
        def _loss_np(q_arr):
            return float(ik_target_function(
                fk_function,
                torch.tensor(q_arr.astype(np.float32)),
                key, kintree, marker_positions, bounds
            ).detach())
        f0 = _loss_np(q_np)
        for i in range(len(q_np)):
            q_plus = q_np.copy(); q_plus[i] += eps
            grad[i] = (_loss_np(q_plus) - f0) / eps
        return grad

    return q_t.grad.detach().numpy().astype(float)


# ---------------------------------------------------------------------------
# Barzilai-Borwein step size (optional)
# ---------------------------------------------------------------------------

def barzilai_borwein_step(xk, xk_minus_1, gk, gk_minus_1):
    s = xk - xk_minus_1
    y = gk - gk_minus_1
    denom = np.dot(s, y)
    if abs(denom) < 1e-12:
        return 1e-3
    return np.dot(s, s) / denom


# ---------------------------------------------------------------------------
# IK solver (LBFGS per frame)
# ---------------------------------------------------------------------------

def ik_solver(fk_function, kintree, marker_data, key, bounds, max_iters=100, tol=1e-6):
    """Solve inverse kinematics for all frames using LBFGS.

    Args:
        fk_function: forward_kinematics callable
        kintree (dict): Kinematic tree
        marker_data (pd.DataFrame): All frames of experimental markers
        key (list): Joint names
        bounds (tuple): (lower, upper) arrays for joint limits
        max_iters (int): LBFGS max iterations per frame
        tol (float): Convergence tolerance

    Returns:
        q_sol (list of np.ndarray): Solved joint angles per frame
        mse_history (list of float): Final loss per frame
    """
    n_frames = len(marker_data)
    n_dof    = len(key)

    q_sol       = []
    mse_history = []

    q_init = np.zeros(n_dof, dtype=np.float32)

    for frame_idx in range(n_frames):
        marker_row = marker_data.iloc[frame_idx]

        q_t = torch.tensor(q_init.copy(), dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [q_t],
            max_iter=max_iters,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn='strong_wolfe'
        )

        frame_losses = []

        def closure():
            optimizer.zero_grad()
            loss = ik_target_function(
                fk_function, q_t, key, kintree, marker_row, bounds
            )
            loss.backward()
            frame_losses.append(loss.item())
            return loss

        optimizer.step(closure)

        q_solved   = q_t.detach().numpy().astype(float)
        final_loss = frame_losses[-1] if frame_losses else float('nan')

        q_sol.append(q_solved)
        mse_history.append(final_loss)

        q_init = q_solved.astype(np.float32)

        if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx+1:>4}/{n_frames}  loss={final_loss:.6f}")

    return q_sol, mse_history