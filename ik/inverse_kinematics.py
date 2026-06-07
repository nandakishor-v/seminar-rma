import numpy as np
import torch
import pandas as pd


def _extract_experimental_markers(marker_row):
    """Parse experimental marker data into {marker_name: np.array(3,)}.

    Handles two formats:
    1. Flat dict: {'marker_test_x': 1.0, 'marker_test_y': -2.0, 'marker_test_z': 3.0}
    2. pandas Series with 'marker_marker_hip_r_x/y/z' column naming
    """
    if isinstance(marker_row, dict):                       # getting real world data marker position if it's in dict form
        # Try to group _x/_y/_z keys into 3D positions
        marker_map = {}
        keys = list(marker_row.keys())
        for k in keys:
            if not k.endswith('_x'):              # checking if it the marker has a x data to start with or not
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


    if hasattr(marker_row, 'index'):                       # getting real world data marker position if it's in pandas form
        index = [str(c) for c in marker_row.index]
        values = marker_row.to_numpy(dtype=float)
        marker_map = {}
        for i in range(len(index) - 2):
            col = index[i]
            if not col.endswith('_x'):
                continue
            base = col[:-2]
            expected = [f"{base}_x", f"{base}_y", f"{base}_z"]
            if index[i:i+3] != expected:                   # as its a pandas series we check of all x,y,z are in succession columns or not
                continue
            marker_map[base] = values[i:i+3]
        return marker_map
    return {}


def _lookup_marker(exp_map, fk_name):
    """Match an FK marker name to an experimental marker name."""
    bare = fk_name.removeprefix('marker_')                       # compares the obtained forward kinematics marker names to match with real data only then to use
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
    if isinstance(pos, (list, tuple)):                                     # converting all possible outcomes to torch
        if len(pos) > 0 and isinstance(pos[0], torch.Tensor):
            return torch.stack([p if isinstance(p, torch.Tensor)
                                else torch.tensor(float(p)) for p in pos])
        return torch.tensor(np.asarray(pos, dtype=np.float32))

    return torch.tensor(np.asarray(pos, dtype=np.float32))



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

    exp_map = _extract_experimental_markers(marker_positions) # finding the marker position using the helper function

    _, fk_markers = fk_function(q_t, key, kintree)  # getting the marker position from our forward kinematics function

    sq_errors = []
    for fk_name, fk_pos in fk_markers.items():    # looping through all the forward kinematics markers and matching it with our real world data calling lookup function
        exp_pos = _lookup_marker(exp_map, fk_name)
        if exp_pos is None:
            continue

        # Convert FK pos -- handles plain lists of torch scalars (keeps grad graph)
        fk_t  = _to_torch_pos(fk_pos)
        exp_t = torch.tensor(np.asarray(exp_pos, dtype=np.float32))
        sq_errors.append(torch.sum((fk_t - exp_t) ** 2))

    if len(sq_errors) == 0:
        return torch.tensor(0.0)

    loss = torch.mean(torch.stack(sq_errors))  # final loss for that particular frame given by mean of all squared errors of all q we get

    # Soft joint limit penalty
    if bounds is not None:   # bounds set in early subtasks used for relu penalties
        lower = torch.tensor(np.asarray(bounds[0], dtype=np.float32))
        upper = torch.tensor(np.asarray(bounds[1], dtype=np.float32))
        penalty = (torch.relu(lower - q_t) ** 2 + torch.relu(q_t - upper) ** 2).mean()
        loss = loss + 0.01 * penalty

    return loss



def compute_ik_gradient(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute dL/dq via PyTorch autograd.

    Returns:
        np.ndarray: Gradient of shape (N,).
    """
    q_t = torch.tensor(np.asarray(q, dtype=np.float32), requires_grad=True)

    loss = ik_target_function(fk_function, q_t, key, kintree, marker_positions, bounds)
    loss.backward() # gets us gradient using simple torch command

    if q_t.grad is None:
        # Graph was broken (eg if FK returned plain numpy) -- use finite differences
        q_np = np.asarray(q, dtype=float)
        eps  = 1e-5
        grad = np.zeros_like(q_np)
        def _loss_np(q_arr):
            return float(ik_target_function( # just calls the loss function again but the result is now detached as torch and is just a float
                fk_function,
                torch.tensor(q_arr.astype(np.float32)),
                key, kintree, marker_positions, bounds
            ).detach())
        f0 = _loss_np(q_np)
        for i in range(len(q_np)):                     # we calculate gradient for each q in the q array using the f0 baseline we got for that particular frame
            q_plus = q_np.copy(); q_plus[i] += eps # the
            grad[i] = (_loss_np(q_plus) - f0) / eps
        return grad

    return q_t.grad.detach().numpy().astype(float)




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

    for frame_idx in range(n_frames):  # loops from 1st till our last frame in marker_data file
        marker_row = marker_data.iloc[frame_idx]    # assigns each row as marker_row variable used as an argument in loss function

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
            frame_losses.append(loss.item())   # as this function will be called by the optimizer for 100 iterations, each iteration loss will be appened
            return loss

        optimizer.step(closure)

        q_solved   = q_t.detach().numpy().astype(float)
        final_loss = frame_losses[-1] if frame_losses else float('nan') # final loss for each frame is saved as the last appeneded loss by closer function

        q_sol.append(q_solved)
        mse_history.append(final_loss)

        q_init = q_solved.astype(np.float32)    # new initialization done for the next iteration

        if (frame_idx + 1) % 50 == 0 or frame_idx == 0:   # just a good looking print message for user
            print(f"  Frame {frame_idx+1:>4}/{n_frames}  loss={final_loss:.6f}")

    return q_sol, mse_history