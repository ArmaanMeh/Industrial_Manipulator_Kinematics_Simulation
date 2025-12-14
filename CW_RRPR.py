import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""CW_RRPR.py

Single-file modular implementation of an RRPR pick-and-place simulation.

Sections are split into: kinematics, trajectory, visualization and runner.
Each section includes a short "Logic:" note explaining reasoning.
"""

# 1. Math & Kinematics Library (Custom Implementation)

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
    # Logic: Encapsulates the Denavit–Hartenberg transform — used to compose link transforms.
    
class RRPRRobot:
    # L1, L2, L3 are link lengths. d4_limits apply to the prismatic joint q3.
    def __init__(self, L1, L2, L3, d4_limits):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3 # L3 is now the length of the final revolute link
        self.d3_min, self.d3_max = d4_limits # Limits for the prismatic joint q3

    def forward_kinematics_all(self, q):
        """
        Calculates the transform matrices for every joint.
        Configuration: Revolute (q1), Revolute (q2), Prismatic (q3), Revolute (q4)
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
        
        # Joint 3: Prismatic (theta=0, d=q3, a=0, alpha=pi/2) <-- Prismatic Joint
        T2_3 = dh_transform(0, q3, 0, np.pi/2)
        T3 = T2 @ T2_3
        
        # Joint 4: Revolute (theta=q4, d=0, a=L3, alpha=0) <-- Revolute Joint (final link)
        T3_4 = dh_transform(q4, 0, self.L3, 0)
        T4 = T3 @ T3_4
        
        return [T0, T1, T2, T3, T4]
    # Logic: Compute joint-to-joint transforms for FK and Jacobian computation.

    def get_end_effector_pos(self, q):
        transforms = self.forward_kinematics_all(q)
        return transforms[-1][:3, 3]
    # Logic: Convenience wrapper returning only the 3D end-effector position.

    def jacobian(self, q):
        """
        Calculates the Geometric Jacobian (6x4 matrix).
        Joint types: R, R, P, R
        """
        transforms = self.forward_kinematics_all(q)
        pe = transforms[-1][:3, 3] # Position of end-effector
        
        J = np.zeros((6, 4))
        
        for i in range(4):
            T_prev = transforms[i]
            z_prev = T_prev[:3, 2] # Z-axis vector (of the previous frame)
            p_prev = T_prev[:3, 3] # Origin coordinates (of the previous frame)
            
            if i == 2: # Prismatic Joint (q3, index 2)
                # Jv = z_prev
                # Jw = 0
                J[:3, i] = z_prev
                J[3:, i] = np.zeros(3)
            else: # Revolute Joints (q1, q2, q4, indices 0, 1, 3)
                # Jv = z_prev x (pe - p_prev)
                # Jw = z_prev
                J[:3, i] = np.cross(z_prev, (pe - p_prev))
                J[3:, i] = z_prev
                
        return J
    # Logic: Geometric Jacobian (6x4) mapping joint rates to end-effector twist.

    def inverse_kinematics(self, target_pos, q0):
        
        q = np.array(q0, dtype=float)
        target_pos = np.array(target_pos)
        
        max_iter = 100
        tolerance = 1e-5
        learning_rate = 0.3
        
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
            
            # Clamp Prismatic Joint (q3 is index 2)
            q[2] = np.clip(q[2], self.d3_min, self.d3_max)
            
            # Normalize Revolute angles (q1, q2, q4 are indices 0, 1, 3)
            revolute_indices = [0, 1, 3]
            q[revolute_indices] = (q[revolute_indices] + np.pi) % (2 * np.pi) - np.pi
            
        return q
    # Logic: Simple pseudo-inverse based IK solver; clamps prismatic and normalizes revolute joints.

# 2. Robot Definition

def create_robot():
    """Create and return a configured RRPRRobot instance.

    Logic: Centralizes robot configuration so callers can create instances consistently.
    """
    L1 = 0.25
    L2 = 0.25
    L3 = 0.25
    d_prismatic_min, d_prismatic_max = 0, 0.45
    return RRPRRobot(L1, L2, L3, [d_prismatic_min, d_prismatic_max])


robot = create_robot()


# 3. Tasks & Trajectory Generation

# --- NEW FLOOR TARGETS ---
floor_targets = [
    (0.3, -0.4, 0.0),    
    (-0.4, -0.3, 0.0),   
    (-0.1, 0.4, 0.0)     
]
# --- NEW SHELF TARGETS (Higher levels) ---
shelf_targets = [
    (0.4, 0.55, 0.30),
    (0.4, 0.55, 0.40),
    (0.4, 0.55, 0.50)
]

tasks = []
for i in range(3):
    tasks.append({"start": floor_targets[i], "end": shelf_targets[i]})

def create_tasks():
    """Return tasks and timing parameters.

    Logic: Keeps task definitions and timing parameters together for clarity.
    """
    tasks = []
    for i in range(3):
        tasks.append({"start": floor_targets[i], "end": shelf_targets[i]})
    approach_offset = 0
    dwell_steps = 5
    return tasks, approach_offset, dwell_steps


tasks, APPROACH_OFFSET, DWELL_STEPS = create_tasks()

def jtraj(q0, q1, steps):
    """
    Generates a joint space trajectory (Linear interpolation).
    Includes logic to take the shortest angular path for Revolute joints.
    """
    q0 = np.array(q0)
    q1 = np.array(q1)
    qs = np.zeros((steps, len(q0)))
    
    # 1. Calculate raw difference
    diff = q1 - q0
    
    # 2. Adjust revolute joints (indices 0, 1, and 3) for shortest path
    revolute_indices = [0, 1, 3]
    for j in revolute_indices:
        if diff[j] > np.pi:
            diff[j] -= 2 * np.pi
        elif diff[j] < -np.pi:
            diff[j] += 2 * np.pi
       
    # 4. Interpolate
    for i, s in enumerate(np.linspace(0, 1, steps)):
        qs[i, :] = q0 + s * diff
        
    return qs

def build_task(robot, q_current, start_xyz, end_xyz, obj_index, approach_offset, dwell_steps):
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz

    # Define approach and contact points based on offset
    p_approach_start = np.array([sx, sy, sz + approach_offset])
    p_contact_start = np.array([sx, sy, sz])
    p_approach_end = np.array([ex, ey, ez + approach_offset])
    p_contact_end = np.array([ex, ey, ez])

    # IK Solutions
    q_app_start = robot.inverse_kinematics(p_approach_start, q_current)
    q_con_start = robot.inverse_kinematics(p_contact_start, q_app_start)
    # Force IK seed to be the contact configuration at start
    q_app_end = robot.inverse_kinematics(p_approach_end, q_con_start)
    q_con_end = robot.inverse_kinematics(p_contact_end, q_app_end)

    traj = []
    actions = []
    indices = []

    # Sequence of moves
    traj.extend(jtraj(q_current, q_app_start, 40))
    actions.extend(["move"] * 40); indices.extend([obj_index] * 40)

    traj.extend(jtraj(q_app_start, q_con_start, 20))
    actions.extend(["move"] * 20); indices.extend([obj_index] * 20)

    traj.extend([q_con_start] * dwell_steps)
    actions.extend(["grip"] * dwell_steps); indices.extend([obj_index] * dwell_steps)

    traj.extend(jtraj(q_con_start, q_app_start, 20))
    actions.extend(["move"] * 20); indices.extend([obj_index] * 20)

    traj.extend(jtraj(q_app_start, q_app_end, 50))
    actions.extend(["move"] * 50); indices.extend([obj_index] * 50)

    traj.extend(jtraj(q_app_end, q_con_end, 20))
    actions.extend(["move"] * 20); indices.extend([obj_index] * 20)
        
    traj.extend([q_con_end] * dwell_steps)
    actions.extend(["place"] * dwell_steps); indices.extend([obj_index] * dwell_steps)

    traj.extend(jtraj(q_con_end, q_app_end, 20))
    actions.extend(["move"] * 20); indices.extend([obj_index] * 20)

    # Logic: Task builder produces approach/contact/grip/transfer/place segments deterministically.
    return np.array(traj), q_app_end, actions, indices

def build_program(tasks):
    q_current = np.array([0.0, 0.0, 0.0, 0.0]) # Home position
    program = []
    all_actions = []
    all_indices = []
    for i, t in enumerate(tasks, 1):
        traj, q_current, task_actions, task_indices = build_task(robot, q_current, t["start"], t["end"], i-1, APPROACH_OFFSET, DWELL_STEPS)
        program.extend(traj)
        all_actions.extend(task_actions)
        all_indices.extend(task_indices)
    # Logic: Chains each task's trajectory into a single program with aligned action/indices arrays.
    return np.array(program), all_actions, all_indices

frames, actions, indices = build_program(tasks)


# 4. Visualization Helpers (Hexagons)

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


# 5. Plotting & Animation

def create_scene():
    """Create Matplotlib figure, axes, objects and graphic handles.

    Returns: (fig, ax, objects, lines, joints, ax_limit)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("R R P R Robot |Targets Simulation")

    # Environment
    reach = robot.L1 + robot.L2 + robot.L3 + robot.d3_max + 0.1
    ax_limit = reach
    Xp, Yp = np.meshgrid(np.linspace(-ax_limit, ax_limit, 2), np.linspace(-ax_limit, ax_limit, 2))
    Zp = np.zeros_like(Xp)
    ax.plot_surface(Xp, Yp, Zp, alpha=0.2, color='lightblue', edgecolor='none')

    # Shelf visualization
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

    # Robot visuals
    link1_line, = ax.plot([], [], [], 'b-', linewidth=3)
    link2_line, = ax.plot([], [], [], 'g-', linewidth=3)
    prism_line, = ax.plot([], [], [], 'm-', linewidth=3) # Joint 3 (Prismatic)
    link4_line, = ax.plot([], [], [], 'orange', linewidth=3) # Joint 4 (Revolute)
    joints, = ax.plot([], [], [], 'ko', markersize=6)

    ax.set_xlim(-ax_limit, ax_limit)
    ax.set_ylim(-ax_limit, ax_limit)
    ax.set_zlim(0.0, 1.2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Logic: Build scene objects and return handles so animation can be run separately.
    return fig, ax, objects, (link1_line, link2_line, prism_line, link4_line), joints, ax_limit


def run_animation(frames, actions, indices, interval=20):
    fig, ax, objects, (link1_line, link2_line, prism_line, link4_line), joints, ax_limit = create_scene()

    grip_tol = 0.01
    place_tol = 0.01

    def reset_objects():
        for obj in objects:
            center = obj["start"]
            faces = hex_faces(center, obj["radius"], obj["height"])
            obj["poly"].set_verts(faces)
            obj["gripped"] = False
            obj["placed"] = False

    def update(frame_idx):
        if frame_idx == 0:
            reset_objects()

        q = frames[frame_idx]
        action = actions[frame_idx]
        obj_index = indices[frame_idx]
        obj = objects[obj_index]

        transforms = robot.forward_kinematics_all(q)
        pts = [T[:3, 3] for T in transforms]
        base, P1, P2, P3, P4 = pts[0], pts[1], pts[2], pts[3], pts[4]

        # Update robot links
        link1_line.set_data([base[0], P1[0]], [base[1], P1[1]])
        link1_line.set_3d_properties([base[2], P1[2]])

        link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
        link2_line.set_3d_properties([P1[2], P2[2]])

        prism_line.set_data([P2[0], P3[0]], [P2[1], P3[1]])
        prism_line.set_3d_properties([P2[2], P3[2]])

        link4_line.set_data([P3[0], P4[0]], [P3[1], P4[1]])
        link4_line.set_3d_properties([P3[2], P4[2]])

        joints.set_data([base[0], P1[0], P2[0], P3[0], P4[0]],
                        [base[1], P1[1], P2[1], P3[1], P4[1]])
        joints.set_3d_properties([base[2], P1[2], P2[2], P3[2], P4[2]])

        # Grip/Place Logic with action flags
        if action == "grip" and not obj["gripped"] and not obj["placed"]:
            obj["gripped"] = True

        if obj["gripped"] and not obj["placed"]:
            if action == "move":
                center = P4
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)

        if action == "place" and obj["gripped"]:
            obj["gripped"] = False
            obj["placed"] = True
            center = obj["end"]
            faces = hex_faces(center, obj["radius"], obj["height"])
            obj["poly"].set_verts(faces)

        return link1_line, link2_line, prism_line, link4_line, joints

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False, repeat=True)
    plt.show()


if __name__ == "__main__":
    # Build program then run animation when executed directly
    frames, actions, indices = build_program(tasks)
    run_animation(frames, actions, indices)
