# Industrial_Manipulator_Kinematics_Simulation
RRPR Robot Arm Simulation
📖 Introduction
This project simulates a 4‑DOF RRPR robot arm (Revolute–Revolute–Prismatic–Revolute) performing pick‑and‑place tasks. It demonstrates forward and inverse kinematics, Jacobian computation, trajectory planning, and animated visualization in Python using Matplotlib.

Objects are picked from floor targets (F1–F3) and placed onto shelf targets (S1–S3) at different heights.

⚙️ Requirements
Python 3.8+

NumPy

Matplotlib

Install dependencies:

bash
pip install numpy matplotlib
🧩 Code Walkthrough
1. Math & Kinematics Library
Defines the Denavit–Hartenberg transform and the RRPRRobot class for kinematics.

python
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])
Logic: Builds a homogeneous transformation matrix for each joint using DH parameters.

python
class RRPRRobot:
    def __init__(self, L1, L2, L3, d4_limits):
        self.L1, self.L2, self.L3 = L1, L2, L3
        self.d3_min, self.d3_max = d4_limits
Logic: Stores robot link lengths and prismatic joint limits.

forward_kinematics_all(q) → Computes transforms for all joints.

jacobian(q) → Builds the 6×4 Jacobian (linear + angular velocity contributions).

inverse_kinematics(target_pos, q0) → Iteratively solves IK using Jacobian pseudo‑inverse.

2. Robot Definition
python
L1 = 0.25; L2 = 0.25; L3 = 0.25
d_prismatic_min, d_prismatic_max = 0, 0.45
robot = RRPRRobot(L1, L2, L3, [d_prismatic_min, d_prismatic_max])
Logic: Defines robot dimensions and creates an instance.

3. Tasks & Trajectory Generation
Defines floor and shelf targets, then builds trajectories with action flags.

python
floor_targets = [(0.3,-0.4,0.0), (-0.4,-0.3,0.0), (-0.1,0.4,0.0)]
shelf_targets = [(0.4,0.55,0.30), (0.4,0.55,0.40), (0.4,0.55,0.50)]
Trajectory interpolation
python
def jtraj(q0, q1, steps):
    diff = q1 - q0
    # Adjust revolute joints for shortest path
    for j in [0,1,3]:
        if diff[j] > np.pi: diff[j] -= 2*np.pi
        elif diff[j] < -np.pi: diff[j] += 2*np.pi
    return [q0 + s*diff for s in np.linspace(0,1,steps)]
Logic: Generates smooth joint‑space paths, correcting for angular wraparound.

Task builder
python
def build_task(q_current, start_xyz, end_xyz, obj_index):
    # Approach, contact, grip, lift, traverse, place, release
    traj, actions, indices = [], [], []
    # Each segment adds frames + action labels + object index
    ...
    return np.array(traj), q_app_end, actions, indices
Logic: Encodes the full pick‑and‑place sequence with grip/place dwell phases.

Program builder
python
def build_program(tasks):
    program, all_actions, all_indices = [], [], []
    q_current = np.array([0,0,0,0])
    for i,t in enumerate(tasks):
        traj, q_current, task_actions, task_indices = build_task(q_current, t["start"], t["end"], i)
        program.extend(traj)
        all_actions.extend(task_actions)
        all_indices.extend(task_indices)
    return np.array(program), all_actions, all_indices
Logic: Chains multiple tasks into one continuous trajectory.

4. Visualization Helpers
python
def hex_faces(center, radius=0.05, height=0.05):
    # Builds hexagonal prism faces for objects
Logic: Creates 3D hexagon geometry for floor/shelf objects.

5. Plotting & Animation
Initializes Matplotlib 3D plot, draws environment, labels targets, and creates robot link lines.

6. Update Function (Animation Loop)
python
def update(frame_idx):
    q = frames[frame_idx]
    action = actions[frame_idx]
    obj_index = indices[frame_idx]
    obj = objects[obj_index]

    # Update robot links
    ...

    # Grip/Place Logic
    if action == "grip" and not obj["gripped"]:
        obj["gripped"] = True
    if obj["gripped"] and not obj["placed"] and action == "move":
        obj["poly"].set_verts(hex_faces(P4, obj["radius"], obj["height"]))
    if action == "place" and obj["gripped"]:
        obj["gripped"] = False; obj["placed"] = True
        obj["poly"].set_verts(hex_faces(obj["end"], obj["radius"], obj["height"]))
Logic:

Attaches object during grip dwell.

Moves object with end‑effector during move.

Detaches object at shelf target during place dwell.

🎬 Running the Simulation
bash
python rrpr_robot.py
The robot will:

Pick objects from F1–F3.

Place them on S1–S3 shelves.

Animate the process with deterministic grip/release.

🔎 Logic Summary
DH Transform: Encodes joint geometry.

FK: Computes end‑effector position.

Jacobian: Relates joint velocities to end‑effector motion.

IK: Iteratively solves for joint angles to reach targets.

Trajectory: Breaks tasks into approach, grip, move, release phases.

Visualization: Animates robot and objects in 3D.

Action Flags: Ensure deterministic grip/release, avoiding tolerance errors.
