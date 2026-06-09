import numpy as np


def _axis_angle_to_rot(axis, angle):
    """Rodrigues' rotation formula."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)                              # formula same as we used in first assignment helpful for caluclating products
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm
    dominant = np.argmax(np.abs(axis))
    if axis[dominant] < 0:
        angle = -angle
        axis  = -axis
    K = np.array([[ 0,       -axis[2],  axis[1]],
                  [ axis[2],  0,       -axis[0]],
                  [-axis[1],  axis[0],  0      ]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)



def forward_kinematics_second_order(q, qdot, qddot, key, kintree):
    """Compute second-order forward kinematics for the given generalized coordinates.

    Propagates segment kinematics through the tree and computes global quantities
    for each segment and marker using recursive Newton-Euler equations.

    Recursive equations:
    ─────────────────────
    HINGE joint (axis n, q_j, qdot_j, qddot_j):
        n_w     = R_p @ n_local
        R_c     = R_p @ R_joint(n, q_j)
        pos_c   = pos_p + R_p @ r_joint
        omega_c = omega_p + qdot_j * n_w
        alpha_c = alpha_p + qddot_j * n_w + qdot_j * (omega_p x n_w)
        v_c     = v_p + omega_p x r_w
        a_c     = a_p + alpha_p x r_w + omega_p x (omega_p x r_w)

    SLIDER joint (axis n, q_j, qdot_j, qddot_j):
        n_w     = R_p @ n_local
        d_w     = R_p @ r_joint + q_j * n_w
        pos_c   = pos_p + d_w
        omega_c = omega_p  (unchanged)
        alpha_c = alpha_p  (unchanged)
        v_c     = v_p + qdot_j * n_w + omega_p x d_w
        a_c     = a_p + qddot_j * n_w
                      + alpha_p x d_w
                      + omega_p x (omega_p x d_w)
                      + 2 * qdot_j * (omega_p x n_w)   [Coriolis]

    Args:
        q      (ndarray): Generalized coordinates of shape (N,).
        qdot   (ndarray): Generalized velocities of shape (N,).
        qddot  (ndarray): Generalized accelerations of shape (N,).
        key    (list): Joint names for q / qdot / qddot.
        kintree (dict): Nested kinematic tree from get_model_dictionary().

    Returns:
        segments (dict): {body_name: {position, orientation, linear_velocity,
                          linear_acceleration, angular_velocity, angular_acceleration}}
        markers  (dict): {marker_name: {position, velocity, acceleration}}
    """

    val_map = {}
    for name, qv, dqv, ddqv in zip(key, q, qdot, qddot):                      # getting the first and second order derivatives ( velocity and accelration)
        normalized = name[2:] if name.startswith('q_') else name          # if name starts with q_ we normalise each marker by removing the q_
        val_map[normalized] = (float(qv), float(dqv), float(ddqv))

    segments = {}
    markers  = {}

    def _traverse(body_dict, R_p, pos_p, v_p, a_p, omega_p, alpha_p):
        for body_name, body_info in body_dict.items():

            # 1. Body origin in global frame
            joint_location = np.asarray(                               # starts to loop in our kintree through each body segment
                body_info.get('joint_location', np.zeros(3)), dtype=float)
            pos_c   = pos_p + R_p @ joint_location
            R_c     = R_p.copy()                                   # copy all the r_p as r_c for recursion, as child adds up parents movement as well
            v_c     = v_p.copy()
            a_c     = a_p.copy()
            omega_c = omega_p.copy()
            alpha_c = alpha_p.copy()

            # Body quaternion skipped (OpenSim->MuJoCo artifact, same as FK)

            # 2. Apply joints
            for joint_name, joint_info in body_info['joints'].items():              # loops through joint info and gets value for each joint movement for that body segment

                r_joint = np.asarray(joint_info.get('pos', np.zeros(3)), dtype=float)
                r_w     = R_c @ r_joint   # joint offset in world frame

                qv, dqv, ddqv = val_map.get(joint_name, (0.0, 0.0, 0.0))

                axis  = np.asarray(joint_info.get('axis', [0., 0., 1.]), dtype=float)
                jtype = joint_info.get('type', 'hinge')

                # Handle negative-dominant axis
                norm = np.linalg.norm(axis)
                if norm > 1e-10:
                    axis_n = axis / norm
                else:
                    axis_n = axis
                dominant = np.argmax(np.abs(axis_n))
                if axis_n[dominant] < 0:
                    axis_n = -axis_n
                    qv, dqv, ddqv = -qv, -dqv, -ddqv

                n_w = R_c @ axis_n   # joint axis in world frame

                if jtype == 'hinge':
                    # Position                             #HINGE type joint
                    pos_c = pos_c + r_w

                    # Linear velocity / acceleration (contribution of r_w lever arm)
                    v_c = v_c + np.cross(omega_c, r_w)
                    a_c = a_c + np.cross(alpha_c, r_w) + \
                          np.cross(omega_c, np.cross(omega_c, r_w))

                    # Angular velocity / acceleration update
                    alpha_c = alpha_c + ddqv * n_w + dqv * np.cross(omega_c, n_w)
                    omega_c = omega_c + dqv * n_w

                    # Orientation update
                    R_c = R_c @ _axis_angle_to_rot(axis_n, qv)

                elif jtype == 'slider':
                    # Total offset including displacement
                    d_w   = r_w + qv * n_w                           #Slider type Joint
                    pos_c = pos_c + d_w

                    # Linear velocity / acceleration (with Coriolis)
                    v_c = v_c + dqv * n_w + np.cross(omega_c, d_w)
                    a_c = a_c + ddqv * n_w + \
                          np.cross(alpha_c, d_w) + \
                          np.cross(omega_c, np.cross(omega_c, d_w)) + \
                          2.0 * dqv * np.cross(omega_c, n_w)

                    # omega / alpha unchanged for slider

            # 3. Store segment state
            segments[body_name] = {                               # we store all values of that body segment
                'position':             pos_c.copy(),
                'orientation':          R_c.copy(),
                'linear_velocity':      v_c.copy(),
                'linear_acceleration':  a_c.copy(),
                'angular_velocity':     omega_c.copy(),
                'angular_acceleration': alpha_c.copy(),
            }

            # 4. Compute marker kinematics
            for marker_name, marker_local in body_info.get('markers', {}).items():
                m   = np.asarray(marker_local, dtype=float)                              # loop for marker movement updates as well relative to that body segment
                r_m = R_c @ m
                markers[marker_name] = {
                    'position':     pos_c + r_m,
                    'velocity':     v_c   + np.cross(omega_c, r_m),
                    'acceleration': a_c   + np.cross(alpha_c, r_m) +
                                    np.cross(omega_c, np.cross(omega_c, r_m)),
                }

            # 5. Recurse into children
            _traverse(                                        # recursive calling the function but with children info so it runs entire function for children
                body_info.get('children', {}),                # this time with arguments r_c as they will now be parent (r_p) for the upcoming children body segment
                R_c, pos_c, v_c, a_c, omega_c, alpha_c
            )

    _traverse(                # to start the first body segment(root) we start with zeros and ones
        kintree,
        R_p    = np.eye(3),
        pos_p  = np.zeros(3),
        v_p    = np.zeros(3),
        a_p    = np.zeros(3),
        omega_p= np.zeros(3),
        alpha_p= np.zeros(3),
    )

    return segments, markers




def get_marker_kinematics(segments, kintree):
    """Compute marker kinematics from already-computed segment kinematics.

    Rotates each marker's local offset into the global frame and propagates
    position, velocity, and acceleration from its parent segment.

    Args:
        segments (dict): Output of forward_kinematics_second_order.
        kintree  (dict): Kinematic tree from get_model_dictionary().

    Returns:
        markers (dict): {marker_name: {position, velocity, acceleration}}
    """
    markers = {}

    def _recurse(body_dict):
        for body_name, body_info in body_dict.items():
            if body_name not in segments:
                _recurse(body_info.get('children', {}))
                continue

            seg = segments[body_name]
            R_seg     = seg['orientation']
            pos_seg   = seg['position']
            v_seg     = seg['linear_velocity']
            a_seg     = seg['linear_acceleration']
            omega_seg = seg['angular_velocity']
            alpha_seg = seg['angular_acceleration']

            for marker_name, marker_local in body_info.get('markers', {}).items():
                m   = np.asarray(marker_local, dtype=float)
                r_m = R_seg @ m   # marker offset in world frame

                markers[marker_name] = {
                    'position':     pos_seg + r_m,
                    'velocity':     v_seg   + np.cross(omega_seg, r_m),
                    'acceleration': a_seg   + np.cross(alpha_seg, r_m) +
                                    np.cross(omega_seg, np.cross(omega_seg, r_m)),
                }

            _recurse(body_info.get('children', {}))

    _recurse(kintree)
    return markers
