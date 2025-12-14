import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------------------------------------------------------
# 1. Math & Kinematics Library (Custom Implementation)
# ------------------------------------------------------------

def dh_transform(theta, d, a, alpha):
    """
    Creates a transformation matrix based on DH parameters.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])
    
class PRRRRobot:
    # L_d1_const: Constant vertical offset (d) for Joint 2
    # L_a2: Length of link 2 (a)
    # L_a3: Length of link 3 (a)
    def __init__(self, L_d1_const, L_a2, L_a3, d1_limits):
        self.L_d1_const = L_d1_const 
        self.L_a2 = L_a2
        self.L_a3 = L_a3 
        self.d1_min, self.d1_max = d1_limits # Limits for the prismatic joint q1 (index 0)

    def forward_kinematics_all(self, q):
        """
        Calculates the transform matrices for every joint.
        Configuration: Prismatic (q1), Revolute (q2), Revolute (q3), Revolute (q4)
        Returns a list of 4x4 matrices [T0, T1, T2, T3, T4].
        
        DH Parameters:
        | Joint | Type | Variable | theta | d       | a       | alpha  |
        |:------|:-----|:---------|:------|:--------|:--------|:-------|
        | 1     | P    | q1       | 0     | q1      | 0       | 0      |  (Vertical Lift)
        | 2     | R    | q2       | q2    | L_d1_const | 0       | -pi/2  |  (Base Rotation)
        | 3     | R    | q3       | q3    | 0       | L_a2    | 0      |  (Shoulder)
        | 4     | R    | q4       | q4    | 0       | L_a3    | 0      |  (Elbow/Wrist)
        """
        q1, q2, q3, q4 = q
        
        # Base frame (Identity)
        T0 = np.eye(4)
        
        # Joint 1: Prismatic (theta=0, d=q1, a=0, alpha=0) 
        T0_1 = dh_transform(0, q1, 0, 0)
        T1 = T0 @ T0_1
        
        # Joint 2: Revolute (theta=q2, d=L_d1_const, a=0, alpha=-pi/2)
        T1_2 = dh_transform(q2, self.L_d1_const, 0, -np.pi/2)
        T2 = T1 @ T1_2
        
        # Joint 3: Revolute (theta=q3, d=0, a=L_a2, alpha=0) 
        T2_3 = dh_transform(q3, 0, self.L_a2, 0)
        T3 = T2 @ T2_3
        
        # Joint 4: Revolute (theta=q4, d=0, a=L_a3, alpha=0) 
        T3_4 = dh_transform(q4, 0, self.L_a3, 0)
        T4 = T3 @ T3_4
        
        return [T0, T1, T2, T3, T4]

    def get_end_effector_pos(self, q):
        transforms = self.forward_kinematics_all(q)
        return transforms[-1][:3, 3]

    def jacobian(self, q):
        """
        Calculates the Geometric Jacobian (6x4 matrix).
        Joint types: P, R, R, R
        """
        transforms = self.forward_kinematics_all(q)
        pe = transforms[-1][:3, 3] # Position of end-effector
        
        J = np.zeros((6, 4))
        
        for i in range(4):
            T_prev = transforms[i]
            z_prev = T_prev[:3, 2] # Z-axis vector (of the previous frame)
            p_prev = T_prev[:3, 3] # Origin coordinates (of the previous frame)
            
            if i == 0: # Prismatic Joint (q1, index 0)
                # Jv = z_prev, Jw = 0
                J[:3, i] = z_prev
                J[3:, i] = np.zeros(3)
            else: # Revolute Joints (q2, q3, q4, indices 1, 2, 3)
                # Jv = z_prev x (pe - p_prev), Jw = z_prev
                J[:3, i] = np.cross(z_prev, (pe - p_prev))
                J[3:, i] = z_prev
                
        return J

    def inverse_kinematics(self, target_pos, q0):
        """
        Numerical Inverse Kinematics using Damped Least Squares (DLS).
        """
        q = np.array(q0, dtype=float)
        target_pos = np.array(target_pos)
        
        max_iter = 100
        tolerance = 1e-5
        learning_rate = 0.3
        
        for i in range(max_iter):
            # 1. Forward Kinematics
            current_pos = self.get_end_effector_pos(q)
            
            # 2. Error Calculation (Position only)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                break
            
            # 3. Jacobian (Position part only: top 3 rows)
            J_full = self.jacobian(q)
            J_pos = J_full[:3, :] # 3x4 matrix
            
            # 4. Damped Least Squares (DLS) Pseudo-Inverse for robust solution
            lambda_sq = 0.01  # Damping factor to stabilize near singularities
            J_T = J_pos.T
            # J_plus = J_T @ inv(J_pos @ J_T + lambda_sq * I)
            J_plus = J_T @ np.linalg.inv(J_pos @ J_T + lambda_sq * np.eye(3))
            
            dq = J_plus @ error
            q += learning_rate * dq
            
            # Clamp Prismatic Joint (q1 is index 0)
            q[0] = np.clip(q[0], self.d1_min, self.d1_max)
            
            # Normalize Revolute angles (q2, q3, q4 are indices 1, 2, 3)
            revolute_indices = [1, 2, 3]
            q[revolute_indices] = (q[revolute_indices] + np.pi) % (2 * np.pi) - np.pi
            
        return q

# ------------------------------------------------------------
# 2. Robot Definition
# ------------------------------------------------------------
# L_d1_const: Constant vertical offset (Z-axis offset for J2)
L_d1_const = 0.25 
# L_a2: Length of link 2
L_a2 = 0.30 
# L_a3: Length of link 3
L_a3 = 0.30 
# Prismatic joint limits (q1 is vertical movement)
d_prismatic_min, d_prismatic_max = 0.0, 0.60 

robot = PRRRRobot(L_d1_const, L_a2, L_a3, [d_prismatic_min, d_prismatic_max])

# ------------------------------------------------------------
# 3. Tasks & Trajectory Generation
# ------------------------------------------------------------
# --- FLOOR TARGETS ---
# Note: Since the prismatic joint (q1) starts at 0.0, target Z=0.0 means P4 is close to L_d1_const
# Adjusting target Z to match the new robot base structure (Z=0 plane)
floor_targets = [
    (0.3, -0.4, 0.01),      
    (-0.4, -0.3, 0.01),     
    (-0.1, 0.4, 0.01)       
]
# --- SHELF TARGETS (Higher levels) ---
shelf_targets = [
    (0.4, 0.55, 0.30),     
    (0.4, 0.55, 0.40),     
    (0.4, 0.55, 0.50)      
]

tasks = []
for i in range(3):
    tasks.append({"start": floor_targets[i], "end": shelf_targets[i]})

APPROACH_OFFSET = 0.10 # For safe clearance
DWELL_STEPS = 5

def jtraj(q0, q1, steps):
    """
    Generates a joint space trajectory (Linear interpolation with shortest path).
    """
    q0 = np.array(q0)
    q1 = np.array(q1)
    qs = np.zeros((steps, len(q0)))
    
    diff = q1 - q0
    
    # Adjust revolute joints (indices 1, 2, and 3) for shortest angular path
    revolute_indices = [1, 2, 3]
    for j in revolute_indices:
        if diff[j] > np.pi:
            diff[j] -= 2 * np.pi
        elif diff[j] < -np.pi:
            diff[j] += 2 * np.pi
        
    # Interpolate
    for i, s in enumerate(np.linspace(0, 1, steps)):
        qs[i, :] = q0 + s * diff
        
    return qs

def build_task(q_current, start_xyz, end_xyz):
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz
    
    # Define approach and contact points based on offset (vertical clearance)
    p_approach_start = np.array([sx, sy, sz + APPROACH_OFFSET])
    p_contact_start  = np.array([sx, sy, sz])
    p_approach_end   = np.array([ex, ey, ez + APPROACH_OFFSET]) 
    p_contact_end    = np.array([ex, ey, ez])

    # IK Solutions - ensure continuity by chaining the q0 seed
    q_app_start = robot.inverse_kinematics(p_approach_start, q_current)
    q_con_start = robot.inverse_kinematics(p_contact_start, q_app_start) 
    
    q_app_end   = robot.inverse_kinematics(p_approach_end, q_con_start)
    q_con_end   = robot.inverse_kinematics(p_contact_end, q_app_end)

    traj = []
    
    # Sequence of moves (steps adjusted for smooth execution)
    traj.extend(jtraj(q_current, q_app_start, 50))   # Move to approach point
    traj.extend(jtraj(q_app_start, q_con_start, 30)) # Slide down to contact
    traj.extend([q_con_start] * DWELL_STEPS)         # Grip/Wait
    traj.extend(jtraj(q_con_start, q_app_start, 30)) # Slide up to clearance
    traj.extend(jtraj(q_app_start, q_app_end, 70))   # Move to shelf approach 
    traj.extend(jtraj(q_app_end, q_con_end, 30))     # Slide down to place
    traj.extend([q_con_end] * DWELL_STEPS)           # Release/Wait
    traj.extend(jtraj(q_con_end, q_app_end, 30))     # Slide up to clearance

    return np.array(traj), q_app_end

def build_program(tasks):
    # Home position: q1=d_prismatic_min (fully retracted/low), q2=0 (forward), q3=0, q4=0
    q_current = np.array([d_prismatic_min, 0.0, 0.0, 0.0]) 
    program = []
    for i, t in enumerate(tasks, 1):
        traj, q_current = build_task(q_current, t["start"], t["end"])
        program.extend(traj)
    
    # Final move back to home position
    q_home = np.array([d_prismatic_min, 0.0, 0.0, 0.0])
    program.extend(jtraj(q_current, q_home, 50))
    
    return np.array(program)

frames = build_program(tasks)

# ------------------------------------------------------------
# 4. Visualization Helpers (Hexagons)
# ------------------------------------------------------------
def hex_faces(center, radius=0.05, height=0.05):
    cx, cy, cz = center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    verts_bottom = [[cx + radius*np.cos(a), cy + radius*np.sin(a), cz] for a in angles]
    verts_top = [[x, y, cz + height] for (x, y, _) in verts_bottom]
    faces = []
    faces.append(verts_bottom)  # bottom face
    faces.append(verts_top)     # top face
    for i in range(6):
        j = (i + 1) % 6
        faces.append([verts_bottom[i], verts_bottom[j], verts_top[j], verts_top[i]])
    return faces

def make_hex_collection(center, radius=0.05, height=0.05, color='gray'):
    return Poly3DCollection(hex_faces(center, radius, height), facecolors=color, alpha=0.85, edgecolor='black')

# ------------------------------------------------------------
# 5. Plotting & Animation
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("P R R R Robot | Stable Pick and Place Simulation")

# Environment
reach = L_a2 + L_a3 + L_d1_const + 0.1 # Maximum horizontal reach
ax_limit = reach
Xp, Yp = np.meshgrid(np.linspace(-ax_limit, ax_limit, 2), np.linspace(-ax_limit, ax_limit, 2))
Zp = np.zeros_like(Xp)
ax.plot_surface(Xp, Yp, Zp, alpha=0.2, color='lightblue', edgecolor='none') # Floor plane at Z=0

# --- SHELF VISUALIZATION ---
shelf_x, shelf_y = [0.3, 0.7], [0.45, 0.65] 
shelf_levels = [0.30, 0.40, 0.50]
for z_level in shelf_levels:
    Xs, Ys = np.meshgrid(shelf_x, shelf_y)
    Zs = np.full_like(Xs, z_level)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='saddlebrown', edgecolor='none')

corner_points = [(shelf_x[0], shelf_y[0]), (shelf_x[0], shelf_y[1]),
                 (shelf_x[1], shelf_y[0]), (shelf_x[1], shelf_y[1])]
for (cx, cy) in corner_points:
    ax.plot([cx, cx], [cy, cy], [0, shelf_levels[-1]], color='saddlebrown', linewidth=3)

# Labels for Targets
for i, (x, y, z) in enumerate(floor_targets, 1):
    ax.text(x, y, z + 0.08, f"F{i}", color="red", ha="center")
for i, (x, y, z) in enumerate(shelf_targets, 1):
    ax.text(x, y, z + 0.08, f"S{i}", color="blue", ha="center")

# Objects (Hexagons)
objects = []
hex_radius, hex_height = 0.05, 0.05
for i, (x, y, z) in enumerate(floor_targets, 1):
    poly = make_hex_collection((x, y, z), radius=hex_radius, height=hex_height, color='gray')
    ax.add_collection3d(poly)
    objects.append({
        "poly": poly,
        "radius": hex_radius, "height": hex_height,
        "start": np.array([x, y, z]),
        "end": np.array(shelf_targets[i - 1]),
        "gripped": False, "placed": False
    })

# Robot Visuals
# Link 1 (Prismatic) connects P0 to P1, which is vertical movement along Z axis.
link1_line, = ax.plot([], [], [], 'm-', linewidth=3) # Joint 1 (Prismatic)
link2_line, = ax.plot([], [], [], 'b-', linewidth=3)
link3_line, = ax.plot([], [], [], 'g-', linewidth=3)
link4_line, = ax.plot([], [], [], 'orange', linewidth=3)
joints, = ax.plot([], [], [], 'ko', markersize=6)

ax.set_xlim(-ax_limit, ax_limit)
ax.set_ylim(-ax_limit, ax_limit)
ax.set_zlim(0.0, 1.2)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

grip_tol = 0.05
place_tol = 0.05

def reset_objects():
    """Resets all objects to their original floor positions."""
    for obj in objects:
        center = obj["start"]
        faces = hex_faces(center, obj["radius"], obj["height"])
        obj["poly"].set_verts(faces)
        obj["gripped"] = False
        obj["placed"] = False

def update(frame_idx):
    # Reset objects at the start of the animation loop
    if frame_idx == 0:
        reset_objects()
        
    q = frames[frame_idx]
    
    transforms = robot.forward_kinematics_all(q)
    pts = [T[:3, 3] for T in transforms]
    base, P1, P2, P3, P4 = pts[0], pts[1], pts[2], pts[3], pts[4] # P0 is origin (0,0,0)

    # Update Links
    # Link 1 (Prismatic movement along Z axis)
    link1_line.set_data([base[0], P1[0]], [base[1], P1[1]])
    link1_line.set_3d_properties([base[2], P1[2]])

    # Link 2
    link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    link2_line.set_3d_properties([P1[2], P2[2]])

    # Link 3
    link3_line.set_data([P2[0], P3[0]], [P2[1], P3[1]]) 
    link3_line.set_3d_properties([P2[2], P3[2]])

    # Link 4
    link4_line.set_data([P3[0], P4[0]], [P3[1], P4[1]]) 
    link4_line.set_3d_properties([P3[2], P4[2]])

    # Joints
    joints.set_data([base[0], P1[0], P2[0], P3[0], P4[0]],
                     [base[1], P1[1], P2[1], P3[1], P4[1]])
    joints.set_3d_properties([base[2], P1[2], P2[2], P3[2], P4[2]])

    # Grip/Place Logic (based on task indexing)
    frames_per_task = 250
    num_tasks = len(tasks)
    
    if frame_idx < frames_per_task * num_tasks:
        current_task_idx = frame_idx // frames_per_task
        obj = objects[current_task_idx]
        
        # Phase 1: Picking
        if not obj["gripped"] and not obj["placed"]:
            # Check for contact
            if np.linalg.norm(P4 - obj["start"]) < grip_tol:
                obj["gripped"] = True 

        # Phase 2: Carrying
        if obj["gripped"] and not obj["placed"]:
            # Object moves with the end-effector
            if np.linalg.norm(P4 - obj["end"]) > place_tol:
                center = P4 
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
            # Phase 3: Placing
            else:
                # Object placed at target
                center = obj["end"]
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
                obj["gripped"] = False
                obj["placed"] = True

    return link1_line, link2_line, link3_line, link4_line, joints

ani = FuncAnimation(fig, update, frames=len(frames), interval=20, blit=False, repeat=True)

plt.show()