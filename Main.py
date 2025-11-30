import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import spatialmath as sm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------------------------------------------------------
# Robot definition (RRRP with DHRobot)
# ------------------------------------------------------------
L1 = 0.30
L2 = 0.35
L3 = 0.35
d4_min, d4_max = -0.45, 0.55

links = [
    RevoluteDH(d=L1, a=0.0, alpha=np.pi/2),
    RevoluteDH(d=0.0, a=L2, alpha=0.0),
    RevoluteDH(d=0.0, a=L3, alpha=0.0),
    PrismaticDH(theta=0.0, a=0.0, alpha=0.0, qlim=[d4_min, d4_max]),
]
rrrp = DHRobot(links, name='RRRP')

# ------------------------------------------------------------
# Tasks: floor (z=0) -> shelf (z>0)
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

APPROACH_OFFSET = 0.0
DWELL_STEPS = 5
POS_MASK = [1, 1, 1, 0, 0, 0]

# ------------------------------------------------------------
# Build approach/contact poses
# ------------------------------------------------------------
def make_poses(x, y, z, slide_dir):
    T_contact = sm.SE3(x, y, z)
    T_approach = sm.SE3(x, y, z) * sm.SE3.Tz(slide_dir * APPROACH_OFFSET)
    return T_approach, T_contact

def jtraj(q0, q1, steps):
    qs = np.zeros((steps, len(q0)))
    for i, s in enumerate(np.linspace(0, 1, steps)):
        qs[i, :] = (1 - s) * q0 + s * q1
    return qs

def clamp_prismatic(q):
    q = np.array(q)
    q[-1] = np.clip(q[-1], d4_min, d4_max)
    return q

# ------------------------------------------------------------
# Build trajectory for one task
# ------------------------------------------------------------
def build_task(q_current, start_xyz, end_xyz):
    Ta_start, Tc_start = make_poses(*start_xyz, slide_dir=-1)
    Ta_end, Tc_end = make_poses(*end_xyz, slide_dir=+1)

    sol_a_start = rrrp.ikine_LM(Ta_start, q0=q_current, mask=POS_MASK)
    sol_c_start = rrrp.ikine_LM(Tc_start, q0=sol_a_start.q, mask=POS_MASK)
    qA_start, qC_start = sol_a_start.q, sol_c_start.q
    qC_slide_start = qA_start.copy(); qC_slide_start[-1] = qC_start[-1]

    sol_a_end = rrrp.ikine_LM(Ta_end, q0=qC_slide_start, mask=POS_MASK)
    sol_c_end = rrrp.ikine_LM(Tc_end, q0=sol_a_end.q, mask=POS_MASK)
    qA_end, qC_end = sol_a_end.q, sol_c_end.q
    qC_slide_end = qA_end.copy(); qC_slide_end[-1] = qC_end[-1]

    traj = []
    traj.extend(jtraj(q_current, qA_start, 40))
    for s in np.linspace(0, 1, 20):
        q = qA_start.copy()
        q[-1] = (1 - s) * qA_start[-1] + s * qC_slide_start[-1]
        traj.append(clamp_prismatic(q))
    traj.extend([qC_slide_start.copy()] * DWELL_STEPS)

    traj.extend(jtraj(qC_slide_start, qA_end, 40))
    for s in np.linspace(0, 1, 20):
        q = qA_end.copy()
        q[-1] = (1 - s) * qA_end[-1] + s * qC_slide_end[-1]
        traj.append(clamp_prismatic(q))
    traj.extend([qC_slide_end.copy()] * DWELL_STEPS)

    return np.array(traj), qC_slide_end

# ------------------------------------------------------------
# Build full program
# ------------------------------------------------------------
def build_program(tasks):
    q_current = np.array([0.0, 0.0, 0.0, 0.0])
    program = []
    for i, t in enumerate(tasks, 1):
        traj, q_current = build_task(q_current, t["start"], t["end"])
        program.extend(traj)
    return np.array(program)

frames = build_program(tasks)

# ------------------------------------------------------------
# Hexagonal object helpers
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
# Visualization 
# ------------------------------------------------------------
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("RRRP | Roboticstoolbox IK")

# Floor plane
reach = L1 + L2 + L3 + d4_max + 0.5
Xp, Yp = np.meshgrid(np.linspace(-reach, reach, 2), np.linspace(-reach, reach, 2))
Zp = np.zeros_like(Xp)
ax.plot_surface(Xp, Yp, Zp, alpha=0.2, color='lightblue', edgecolor='none')

# Shelf planes (aligned with shelf target Z levels)
shelf_x = [0.45, 0.65]   # span around your shelf target X
shelf_y = [0.50, 0.60]   # span around your shelf target Y
shelf_levels = [0.25, 0.35, 0.45]
for z_level in shelf_levels:
    Xs, Ys = np.meshgrid(shelf_x, shelf_y)
    Zs = np.full_like(Xs, z_level)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='saddlebrown', edgecolor='none')
# Add vertical support lines at the four corners
corner_points = [(shelf_x[0], shelf_y[0]),
                 (shelf_x[0], shelf_y[1]),
                 (shelf_x[1], shelf_y[0]),
                 (shelf_x[1], shelf_y[1])]
for (cx, cy) in corner_points:
    ax.plot([cx, cx], [cy, cy], [shelf_levels[0], shelf_levels[-1]],
            color='saddlebrown', linewidth=3)
# Label the shelf
ax.text(np.mean(shelf_x), np.mean(shelf_y), 0.50, "Shelf", color="saddlebrown",
        fontsize=12, ha="center", va="bottom")

# 3D spherical work envelope (visual only)
R = L2 + L3 + d4_max   # max radial reach
u = np.linspace(0, 2*np.pi, 50)   # azimuth
v = np.linspace(0, np.pi/2, 50)   # polar (restrict to upper hemisphere if Z>=0)
x_sphere = R * np.outer(np.cos(u), np.sin(v))
y_sphere = R * np.outer(np.sin(u), np.sin(v))
z_sphere = R * np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3, linewidth=0.5)
ax.text(0, 0, R, "Work Envelope", color="black", fontsize=12, ha="center", va="bottom")

# Plot floor and shelf points (kept as-is)
for i, (x, y, z) in enumerate(floor_targets, 1):
    ax.scatter(x, y, z, c="red", s=100, marker="o", edgecolors="black")
    ax.text(x, y, z + 0.02, f"F{i}", color="red", ha="center")
for i, (x, y, z) in enumerate(shelf_targets, 1):
    ax.scatter(x, y, z, c="blue", s=100, marker="^", edgecolors="black")
    ax.text(x, y, z + 0.02, f"S{i}", color="blue", ha="center")

# Objects (hexagonal valves) at floor targets, to be gripped and placed
objects = []
hex_radius = 0.05
hex_height = 0.05
for i, (x, y, z) in enumerate(floor_targets, 1):
    poly = make_hex_collection((x, y, z), radius=hex_radius, height=hex_height, color='gray')
    ax.add_collection3d(poly)
    objects.append({
        "poly": poly,
        "radius": hex_radius,
        "height": hex_height,
        "start": np.array([x, y, z]),
        "end": np.array(shelf_targets[i - 1]),
        "gripped": False,
        "placed": False
    })

# Robot graphics
link1_line, = ax.plot([], [], [], 'b-', linewidth=3)
link2_line, = ax.plot([], [], [], 'g-', linewidth=3)
link3_line, = ax.plot([], [], [], 'orange', linewidth=3)
prism_line, = ax.plot([], [], [], 'm-', linewidth=3)
joints, = ax.plot([], [], [], 'ko', markersize=6)

ax.set_xlim(-reach, reach)
ax.set_ylim(-reach, reach)
ax.set_zlim(0.0, reach)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

# Grip logic parameters
grip_tol = 0.01   # distance threshold to "touch" and grip at floor
place_tol = 0.01  # distance threshold to place at shelf

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
    # Use roboticstoolbox FK to get all joint frames
    T_all = rrrp.fkine_all(q)
    pts = [T.t for T in T_all]
    base, P1, P2, P3, P4 = pts[0], pts[1], pts[2], pts[3], pts[4]

    # Link 1
    link1_line.set_data([base[0], P1[0]], [base[1], P1[1]])
    link1_line.set_3d_properties([base[2], P1[2]])

    # Link 2
    link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    link2_line.set_3d_properties([P1[2], P2[2]])

    # Link 3
    link3_line.set_data([P2[0], P3[0]], [P2[1], P3[1]])
    link3_line.set_3d_properties([P2[2], P3[2]])

    # Prismatic segment
    prism_line.set_data([P3[0], P4[0]], [P3[1], P4[1]])
    prism_line.set_3d_properties([P3[2], P4[2]])

    # Joints as black dots
    joints.set_data([base[0], P1[0], P2[0], P3[0], P4[0]],
                    [base[1], P1[1], P2[1], P3[1], P4[1]])
    joints.set_3d_properties([base[2], P1[2], P2[2], P3[2], P4[2]])

    # Objects grip and place logic
    for obj in objects:
        if not obj["gripped"] and not obj["placed"]:
            # Grip when end-effector is close to floor object
            if np.linalg.norm(P4 - obj["start"]) < grip_tol:
                obj["gripped"] = True

        if obj["gripped"] and not obj["placed"]:
            # Move with end-effector until close to shelf target
            if np.linalg.norm(P4 - obj["end"]) > place_tol:
                center = P4  # attach object center to P4
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
            else:
                # Place at shelf target and release
                center = obj["end"]
                faces = hex_faces(center, obj["radius"], obj["height"])
                obj["poly"].set_verts(faces)
                obj["gripped"] = False
                obj["placed"] = True
            
    return link1_line, link2_line, link3_line, prism_line, joints, *(o["poly"] for o in objects)

ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False, repeat=True)

plt.show()
