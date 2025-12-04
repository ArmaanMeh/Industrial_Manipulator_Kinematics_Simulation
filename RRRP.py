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

class RRRPRobot:
    def __init__(self, L1, L2, L3, d4_limits):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.d4_min, self.d4_max = d4_limits

    def forward_kinematics_all(self, q):
        """
        Calculates the transform matrices for every joint.
        Returns a list of 4x4 matrices [T0, T1, T2, T3, T4].
        """
        q1, q2, q3, q4 = q
        
        # Base frame (Identity)
        T0 = np.eye(4)
        
        # Joint 1: Revolute (d=L1, a=0, alpha=pi/2)
        T0_1 = dh_transform(q1, self.L1, 0, np.pi/2)
        T1 = T0 @ T0_1
        
        # Joint 2: Revolute (d=0, a=L2, alpha=0)
        T1_2 = dh_transform(q2, 0, self.L2, 0)
        T2 = T1 @ T1_2
        
        # Joint 3: Revolute (d=0, a=L3, alpha=0)
        T2_3 = dh_transform(q3, 0, self.L3, 0)
        T3 = T2 @ T2_3
        
        # Joint 4: Prismatic (theta=0, d=q4, a=0, alpha=0)
        # Note: For prismatic, 'd' is the variable
        T3_4 = dh_transform(0, q4, 0, 0)
        T4 = T3 @ T3_4
        
        return [T0, T1, T2, T3, T4]

    def get_end_effector_pos(self, q):
        transforms = self.forward_kinematics_all(q)
        return transforms[-1][:3, 3]

    def jacobian(self, q):
        """
        Calculates the Geometric Jacobian (6x4 matrix).
        Rows 0-2: Linear Velocity (Jv)
        Rows 3-5: Angular Velocity (Jw)
        """
        transforms = self.forward_kinematics_all(q)
        pe = transforms[-1][:3, 3] # Position of end-effector
        
        J = np.zeros((6, 4))
        
        # Iterate over joints to fill Jacobian columns
        # z_i is the z-axis of the previous frame (i-1)
        # p_i is the origin of the previous frame (i-1)
        
        for i in range(4):
            T_prev = transforms[i]
            z_prev = T_prev[:3, 2] # Z-axis vector
            p_prev = T_prev[:3, 3] # Origin coordinates
            
            if i < 3: # Revolute Joints (1, 2, 3)
                # Jv = z_prev x (pe - p_prev)
                # Jw = z_prev
                J[:3, i] = np.cross(z_prev, (pe - p_prev))
                J[3:, i] = z_prev
            else: # Prismatic Joint (4)
                # Jv = z_prev
                # Jw = 0
                J[:3, i] = z_prev
                J[3:, i] = np.zeros(3)
                
        return J

    def inverse_kinematics(self, target_pos, q0):
        """
        Numerical Inverse Kinematics using Newton-Raphson.
        """
        q = np.array(q0, dtype=float)
        target_pos = np.array(target_pos)
        
        max_iter = 100
        tolerance = 1e-4
        learning_rate = 0.5
        
        for _ in range(max_iter):
            # 1. Forward Kinematics
            current_pos = self.get_end_effector_pos(q)
            
            # 2. Error Calculation (Position only)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                break
            
            # 3. Jacobian (Position part only: top 3 rows)
            J_full = self.jacobian(q)
            J_pos = J_full[:3, :] # 3x4 matrix
            
            # 4. Pseudo-Inverse update
            dq = np.linalg.pinv(J_pos) @ error
            q += learning_rate * dq
            
            # Clamp Prismatic Joint
            q[3] = np.clip(q[3], self.d4_min, self.d4_max)
            
            # Normalize angles to [-pi, pi] to prevent wind-up
            q[:3] = (q[:3] + np.pi) % (2 * np.pi) - np.pi
            
        return q

# ------------------------------------------------------------
# 2. Robot Definition
# ------------------------------------------------------------
L1 = 0.30
L2 = 0.35
L3 = 0.35
d4_min, d4_max = -0.45, 0.55

robot = RRRPRobot(L1, L2, L3, [d4_min, d4_max])

# ------------------------------------------------------------
# 3. Tasks & Trajectory Generation
# ------------------------------------------------------------
floor_targets = [
    (-0.4, 0.4, 0.0),
    (0.5,  0.5, 0.0),
    (0.0, -0.4, 0.0)
]
shelf_targets = [
    (0.5, 0.55, 0.25),
    (0.5, 0.55, 0.35),
    (0.5, 0.55, 0.45)
]

tasks = []
for i in range(3):
    tasks.append({"start": floor_targets[i], "end": shelf_targets[i]})

APPROACH_OFFSET = 0.15 
DWELL_STEPS = 5

def jtraj(q0, q1, steps):
    """
    Generates a joint space trajectory (Linear interpolation).
    Includes logic to take the shortest angular path (unwrap).
    """
    q0 = np.array(q0)
    q1 = np.array(q1)
    qs = np.zeros((steps, len(q0)))
    
    # 1. Calculate raw difference
    diff = q1 - q0
    
    # 2. Adjust revolute joints (indices 0, 1, 2) for shortest path
    # If moving more than 180 degrees, go the other way
    for j in range(3):
        if diff[j] > np.pi:
            diff[j] -= 2 * np.pi
        elif diff[j] < -np.pi:
            diff[j] += 2 * np.pi
            
    # 3. Interpolate
    for i, s in enumerate(np.linspace(0, 1, steps)):
        qs[i, :] = q0 + s * diff
        
    return qs

def build_task(q_current, start_xyz, end_xyz):
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz
    
    p_approach_start = np.array([sx, sy, sz + APPROACH_OFFSET])
    p_contact_start  = np.array([sx, sy, sz])
    p_approach_end   = np.array([ex, ey, ez + APPROACH_OFFSET]) 
    p_contact_end    = np.array([ex, ey, ez])

    # IK Solutions
    q_app_start = robot.inverse_kinematics(p_approach_start, q_current)
    q_con_start = robot.inverse_kinematics(p_contact_start, q_app_start)
    q_app_end   = robot.inverse_kinematics(p_approach_end, q_app_start)
    q_con_end   = robot.inverse_kinematics(p_contact_end, q_app_end)

    traj = []
    
    # Sequence of moves
    traj.extend(jtraj(q_current, q_app_start, 40))   # Move to object
    traj.extend(jtraj(q_app_start, q_con_start, 20)) # Slide down
    traj.extend([q_con_start] * DWELL_STEPS)         # Grip
    traj.extend(jtraj(q_con_start, q_app_start, 20)) # Slide up
    traj.extend(jtraj(q_app_start, q_app_end, 50))   # Move to shelf
    traj.extend(jtraj(q_app_end, q_con_end, 20))     # Slide in
    traj.extend([q_con_end] * DWELL_STEPS)           # Release
    traj.extend(jtraj(q_con_end, q_app_end, 20))     # Slide out

    return np.array(traj), q_app_end

def build_program(tasks):
    q_current = np.array([0.0, 0.0, 0.0, 0.0]) # Home position
    program = []
    for i, t in enumerate(tasks, 1):
        traj, q_current = build_task(q_current, t["start"], t["end"])
        program.extend(traj)
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
ax.set_title("RRRP Robot | Optimized Trajectory & Reset")

# Environment
reach = L1 + L2 + L3 + d4_max + 0.1
Xp, Yp = np.meshgrid(np.linspace(-reach, reach, 2), np.linspace(-reach, reach, 2))
Zp = np.zeros_like(Xp)
ax.plot_surface(Xp, Yp, Zp, alpha=0.2, color='lightblue', edgecolor='none')

# Shelf
shelf_x, shelf_y = [0.45, 0.65], [0.50, 0.60]
shelf_levels = [0.25, 0.35, 0.45]
for z_level in shelf_levels:
    Xs, Ys = np.meshgrid(shelf_x, shelf_y)
    Zs = np.full_like(Xs, z_level)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='saddlebrown', edgecolor='none')

corner_points = [(shelf_x[0], shelf_y[0]), (shelf_x[0], shelf_y[1]),
                 (shelf_x[1], shelf_y[0]), (shelf_x[1], shelf_y[1])]
for (cx, cy) in corner_points:
    ax.plot([cx, cx], [cy, cy], [0, shelf_levels[-1]], color='saddlebrown', linewidth=3)

# Labels
for i, (x, y, z) in enumerate(floor_targets, 1):
    ax.text(x, y, z + 0.08, f"F{i}", color="red", ha="center")
for i, (x, y, z) in enumerate(shelf_targets, 1):
    ax.text(x, y, z + 0.08, f"S{i}", color="blue", ha="center")

# Objects
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
link1_line, = ax.plot([], [], [], 'b-', linewidth=3)
link2_line, = ax.plot([], [], [], 'g-', linewidth=3)
link3_line, = ax.plot([], [], [], 'orange', linewidth=3)
prism_line, = ax.plot([], [], [], 'm-', linewidth=3)
joints, = ax.plot([], [], [], 'ko', markersize=6)

ax.set_xlim(-reach, reach)
ax.set_ylim(-reach, reach)
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
    base, P1, P2, P3, P4 = pts[0], pts[1], pts[2], pts[3], pts[4]

    # Update Links
    link1_line.set_data([base[0], P1[0]], [base[1], P1[1]])
    link1_line.set_3d_properties([base[2], P1[2]])

    link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    link2_line.set_3d_properties([P1[2], P2[2]])

    link3_line.set_data([P2[0], P3[0]], [P2[1], P3[1]])
    link3_line.set_3d_properties([P2[2], P3[2]])

    prism_line.set_data([P3[0], P4[0]], [P3[1], P4[1]])
    prism_line.set_3d_properties([P3[2], P4[2]])

    joints.set_data([base[0], P1[0], P2[0], P3[0], P4[0]],
                    [base[1], P1[1], P2[1], P3[1], P4[1]])
    joints.set_3d_properties([base[2], P1[2], P2[2], P3[2], P4[2]])

    # Grip/Place Logic
    for obj in objects:
        if not obj["gripped"] and not obj["placed"]:
            if np.linalg.norm(P4 - obj["start"]) < grip_tol:
                obj["gripped"] = True

        if obj["gripped"] and not obj["placed"]:
            if np.linalg.norm(P4 - obj["end"]) > place_tol:
                center = P4 
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
            else:
                center = obj["end"]
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
                obj["gripped"] = False
                obj["placed"] = True
            
    return link1_line, link2_line, link3_line, prism_line, joints

ani = FuncAnimation(fig, update, frames=len(frames), interval=20, blit=False, repeat=True)

plt.show()