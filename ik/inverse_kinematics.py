import numpy as np
import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_experimental_markers(marker_row):
    """Parse a row of marker_data into {marker_name: np.array(3,)}.

    Handles the 'marker_marker_hip_r_x/y/z' column naming convention.
    """
    if isinstance(marker_row, dict):
        return {k: np.asarray(v, dtype=float) for k, v in marker_row.items()}

    # pandas Series
    if hasattr(marker_row, 'index'):
        index = [str(c) for c in marker_row.index]
        values = marker_row.to_numpy(dtype=float)
        marker_map = {}
        for i in range(len(index) - 2):
            col = index[i]
            if not col.endswith('_x'):
                continue
            base = col[:-2]  # strip '_x'
            expected = [f"{base}_x", f"{base}_y", f"{base}_z"]
            if index[i:i+3] != expected:
                continue
            marker_map[base] = values[i:i+3]
        return marker_map
    return {}


def _lookup_marker(exp_map, fk_name):
    """Match an FK marker name to an experimental marker name.

    Tries several candidate names to handle 'marker_' prefix variations
    and the 'marker_marker_' double-prefix in the CSV.
    """
    bare = fk_name.removeprefix('marker_')
    candidates = [
        fk_name,                    # e.g. 'marker_hip_r'
        f"marker_{fk_name}",        # e.g. 'marker_marker_hip_r'
        bare,                       # e.g. 'hip_r'
        f"marker_{bare}",           # e.g. 'marker_hip_r'
    ]
    for c in candidates:
        if c in exp_map:
            return exp_map[c]
    return None


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
    # Convert q to torch tensor with grad if not already
    if isinstance(q, np.ndarray):
        q_t = torch.tensor(q, dtype=torch.float32, requires_grad=q.dtype == float)
    elif isinstance(q, torch.Tensor):
        q_t = q.float()
    else:
        q_t = torch.tensor(list(q), dtype=torch.float32)

    # Parse experimental markers
    exp_map = _extract_experimental_markers(marker_positions)

    # Run FK -- returns dicts of torch tensors
    _, fk_markers = fk_function(q_t, key, kintree)

    # Accumulate squared errors over matched markers
    sq_errors = []
    for fk_name, fk_pos in fk_markers.items():
        exp_pos = _lookup_marker(exp_map, fk_name)
        if exp_pos is None:
            continue  # skip missing markers
        exp_t = torch.tensor(exp_pos, dtype=torch.float32)
        sq_errors.append(torch.sum((fk_pos - exp_t) ** 2))

    if len(sq_errors) == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = torch.mean(torch.stack(sq_errors))

    # Optional: smooth joint limit penalty (ReLu-based)
    if bounds is not None:
        lower = torch.tensor(bounds[0], dtype=torch.float32)
        upper = torch.tensor(bounds[1], dtype=torch.float32)
        penalty = (torch.relu(lower - q_t) ** 2 + torch.relu(q_t - upper) ** 2).mean()
        loss = loss + 0.01 * penalty  # small weight -- limits are soft

    return loss


# ---------------------------------------------------------------------------
# Gradient via autograd
# ---------------------------------------------------------------------------

def compute_ik_gradient(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute dL/dq via PyTorch autograd.

    Args:
        (same as ik_target_function)

    Returns:
        np.ndarray: Gradient of shape (N,).
    """
    q_t = torch.tensor(np.asarray(q, dtype=np.float32), requires_grad=True)

    loss = ik_target_function(fk_function, q_t, key, kintree, marker_positions, bounds)
    loss.backward()

    return q_t.grad.detach().numpy().astype(float)


# ---------------------------------------------------------------------------
# Barzilai-Borwein step size (optional first-order helper)
# ---------------------------------------------------------------------------

def barzilai_borwein_step(xk, xk_minus_1, gk, gk_minus_1):
    """Compute Barzilai-Borwein step size.

    BB step: alpha = (s^T s) / (s^T y)  where s = xk - xk_minus_1, y = gk - gk_minus_1
    """
    s = xk - xk_minus_1
    y = gk - gk_minus_1
    denom = np.dot(s, y)
    if abs(denom) < 1e-12:
        return 1e-3  # fallback
    return np.dot(s, s) / denom


# ---------------------------------------------------------------------------
# IK solver (LBFGS per frame)
# ---------------------------------------------------------------------------

def ik_solver(fk_function, kintree, marker_data, key, bounds, max_iters=100, tol=1e-6):
    """Solve inverse kinematics for all frames using LBFGS.

    Strategy:
    - Frame 0: initialize q with zeros
    - Frame i>0: initialize with solution from frame i-1 (warm start)
    - Optimizer: LBFGS with strong_wolfe line search (same as Rosenbrock example)

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

    # Initial guess: zeros for frame 0
    q_init = np.zeros(n_dof, dtype=np.float32)

    for frame_idx in range(n_frames):
        marker_row = marker_data.iloc[frame_idx]

        # Wrap q as optimizable tensor
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

        q_solved = q_t.detach().numpy().astype(float)
        final_loss = frame_losses[-1] if frame_losses else float('nan')

        q_sol.append(q_solved)
        mse_history.append(final_loss)

        # Warm start: use current solution for next frame
        q_init = q_solved.astype(np.float32)

        if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx+1:>4}/{n_frames}  loss={final_loss:.6f}")

    return q_sol, mse_history