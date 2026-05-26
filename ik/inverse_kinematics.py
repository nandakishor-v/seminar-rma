import numpy as np
import torch
import pandas as pd

def barzilai_borwein_step(xk, xk_minus_1, gk, gk_minus_1):
    """Compute one Barzilai-Borwein step size.
    
    This helper is optional for the assignment. Keep it if you want to compare
    first-order methods against LBFGS.
    """
    # s_k = x_k - x_{k-1}
    sk = xk - xk_minus_1
    # y_k = g_k - g_{k-1}
    yk = gk - gk_minus_1
    
    # BB Formula: (s_k^T * s_k) / (s_k^T * y_k)
    numerator = np.dot(sk, sk)
    denominator = np.dot(sk, yk)
    
    # Prevent division by zero if the gradient hasn't changed
    if denominator == 0:
        return 1e-3
        
    return numerator / denominator


def compute_ik_gradient(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute the gradient of the IK objective with respect to q.

    Use your autodiff library of choice here. If you use PyTorch, make sure that
    `numpy.ndarray` inputs are converted before the optimization step.
    """
    # Wrap the numpy array into a PyTorch tensor that tracks gradients
    q_tensor = torch.tensor(q, dtype=torch.float32, requires_grad=True)
    
    # Calculate the loss
    loss = ik_target_function(fk_function, q_tensor, key, kintree, marker_positions, bounds)
    
    # Trigger PyTorch's Autograd to compute dL / dq
    loss.backward()
    
    # Return the gradient as a standard numpy array
    return q_tensor.grad.detach().numpy()


def ik_target_function(fk_function, q, key, kintree, marker_positions, bounds):
    """Compute the IK target function (loss) given joint angles and target positions."""
    
    # 1. Ensure q is a tensor
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        
    # 2. Run forward kinematics for the current q
    _, predicted_markers = fk_function(q, key, kintree)
    
    loss = torch.tensor(0.0, dtype=torch.float32)
    
    # 3. Compare model and experimental marker positions
    for marker_name, p_pos in predicted_markers.items():
        col_x = f"{marker_name}_x"
        col_y = f"{marker_name}_y"
        col_z = f"{marker_name}_z"
        
        # Skip missing markers (NaNs in the CSV)
       # Skip missing markers (NaNs in the CSV)
        if col_x in marker_positions and not pd.isna(marker_positions[col_x]):
            real_pos = torch.tensor([
                marker_positions[col_x],
                marker_positions[col_y],
                marker_positions[col_z]
            ], dtype=torch.float32)
            
            # MAGIC FIX: Safely process dummy test lists vs real tensors
            if isinstance(p_pos, (list, tuple, np.ndarray)):
                # We MUST use torch.stack() here, otherwise PyTorch snaps the gradient chain!
                p_pos_tensor = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in p_pos])
            else:
                p_pos_tensor = p_pos
                
            # Add Squared Error to the loss
            loss = loss + torch.sum((p_pos_tensor - real_pos) ** 2)
            
    # 4. Optionally add a penalty for violating joint limits
    if bounds is not None:
        min_b = torch.tensor(bounds[0], dtype=torch.float32)
        max_b = torch.tensor(bounds[1], dtype=torch.float32)
        
        # Heavy penalty weight
        penalty_weight = 10.0 
        
        # ReLU naturally returns 0 if inside bounds, and a positive number if outside
        lower_penalty = torch.sum(torch.relu(min_b - q) ** 2)
        upper_penalty = torch.sum(torch.relu(q - max_b) ** 2)
        
        loss = loss + penalty_weight * (lower_penalty + upper_penalty)
        
    # 5. Return a scalar loss
    return loss


def ik_solver(fk_function, kintree, marker_data, key, bounds, max_iters=100, tol=1e-6):
    """Inverse Kinematics solver looping through all frames using LBFGS."""
    
    q_sol = []
    mse_history = []
    
    # 1. Choose an initial guess for the first frame (all zeros)
    q_current = np.zeros(len(key), dtype=float)

    # Loop through every single frame in the walking trial
    for idx in range(len(marker_data)):
        frame_data = marker_data.iloc[idx]
        
        # Create a tensor from the PREVIOUS frame's solution (or zeros if it's frame 0)
        q_tensor = torch.tensor(q_current, dtype=torch.float32, requires_grad=True)
        
        # 3. Optimize each frame with PyTorch LBFGS
        optimizer = torch.optim.LBFGS([q_tensor], max_iter=max_iters, tolerance_change=tol)
        
        def closure():
            optimizer.zero_grad()
            loss = ik_target_function(fk_function, q_tensor, key, kintree, frame_data, bounds)
            loss.backward()
            return loss

        # Run the optimization loop for this specific frame
        optimizer.step(closure)
        
        # Extract the solved angles as a numpy array
        q_current = q_tensor.detach().numpy()
        
        # Get the final lowest loss value achieved for this frame
        final_loss = ik_target_function(fk_function, torch.tensor(q_current, dtype=torch.float32), key, kintree, frame_data, bounds).item()
        
        # 4. Store the solution and the loss history
        q_sol.append(q_current)
        mse_history.append(final_loss)
        
        # Optional: Print progress every 100 frames so you know it hasn't crashed
        if idx % 100 == 0:
            print(f"Solved Frame {idx}/{len(marker_data)} | Loss: {final_loss:.4f}")

    # 5. Return the full trial solution
    return np.array(q_sol), mse_history