import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 4.0, 5.5           # shoulder and elbow link lengths
d_min, d_max = 0.0, 6.0     # telescopic forearm extension limits

# Dynamic radial constraint: maximum reach in XY plane
max_radial = L1 + L2 + d_max   # safe upper bound

# -----------------------------
# Targets (floor + shelves)
# -----------------------------
floor_targets = [
    (3.0, 0.0, 0.5),        # F1
    (-1.5, 2.598, 0.5),     # F2
    (-1.5, -2.598, 0.5)     # F3
]

shelf_targets = [
    (3.0, 0.0, 3.0),        # S1
    (-1.5, 2.598, 4.0),     # S2
    (-1.5, -2.598, 5.0)     # S3
]

# -----------------------------
# Inverse kinematics
# -----------------------------
def ik_telescopic(x, y, z):
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    dist = np.sqrt(r_xy**2 + z**2)

    cos_theta3 = (dist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    if abs(cos_theta3) > 1.0:
        raise ValueError("Target out of reach")
    theta3 = np.arccos(cos_theta3)

    k1 = L1 + L2 * np.cos(theta3)
    k2 = L2 * np.sin(theta3)
    theta2 = np.arctan2(z, r_xy) - np.arctan2(k2, k1)

    reach_rr = L1 * np.cos(theta2) + L2 * np.cos(theta2 + theta3)
    d = dist - reach_rr

    if d < d_min or d > d_max:
        raise ValueError("Telescopic extension out of range")

    return theta1, theta2, theta3, d

# -----------------------------
# Forward kinematics
# -----------------------------
def fk_telescopic(theta1, theta2, theta3, d):
    base = np.array([0.0, 0.0, 0.0])

    j2 = np.array([
        L1 * np.cos(theta1) * np.cos(theta2),
        L1 * np.sin(theta1) * np.cos(theta2),
        L1 * np.sin(theta2)
    ])

    j3 = j2 + np.array([
        L2 * np.cos(theta1) * np.cos(theta2 + theta3),
        L2 * np.sin(theta1) * np.cos(theta2 + theta3),
        L2 * np.sin(theta2 + theta3)
    ])

    ee = j3 + np.array([
        d * np.cos(theta1) * np.cos(theta2 + theta3),
        d * np.sin(theta1) * np.cos(theta2 + theta3),
        d * np.sin(theta2 + theta3)
    ])

    return base, j2, j3, ee

# -----------------------------
# Pose sequence: F1->S1->F2->S2->F3->S3
# -----------------------------
poses = []
for f, s in zip(floor_targets, shelf_targets):
    poses.append(ik_telescopic(*f))
    poses.append(ik_telescopic(*s))

# -----------------------------
# Plot setup
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-max_radial-2, max_radial+2)
ax.set_ylim(-max_radial-2, max_radial+2)
ax.set_zlim(0, 8)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("3R + Telescopic Forearm | Floor to Shelf Stacking")

# Floor grid
xg = np.linspace(-max_radial, max_radial, 20)
yg = np.linspace(-max_radial, max_radial, 20)
Xg, Yg = np.meshgrid(xg, yg)
Zg = np.zeros_like(Xg)
ax.plot_wireframe(Xg, Yg, Zg, color='gray', linewidth=0.3, alpha=0.4)

# Reachable circle in XY plane
theta = np.linspace(0, 2*np.pi, 200)
xc = max_radial * np.cos(theta)
yc = max_radial * np.sin(theta)
zc = np.zeros_like(theta) + 0.01
ax.plot(xc, yc, zc, color='gray', linestyle='--', linewidth=1)

# Targets
for i, (x, y, z) in enumerate(floor_targets, start=1):
    ax.scatter(x, y, z, c='red', marker='o', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"F{i}", color='red', fontsize=9, ha='center')

for i, (x, y, z) in enumerate(shelf_targets, start=1):
    ax.scatter(x, y, z, c='blue', marker='^', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"S{i}", color='blue', fontsize=9, ha='center')

# Robot graphics
link1, = ax.plot([], [], [], 'b', linewidth=4, label='Link 1')
link2, = ax.plot([], [], [], 'g', linewidth=4, label='Link 2')
telescopic, = ax.plot([], [], [], 'm', linewidth=4, label='Telescopic Forearm')
joints, = ax.plot([], [], [], 'ro', markersize=8, label='Joints')
ax.legend()

# -----------------------------
# Interpolation
# -----------------------------
frames_per_segment = 30
all_frames = []
for i in range(len(poses) - 1):
    start = np.array(poses[i])
    end = np.array(poses[i + 1])
    for t in np.linspace(0.0, 1.0, frames_per_segment):
        all_frames.append((1.0 - t) * start + t * end)

# -----------------------------
# Animation update
# -----------------------------
def update(frame):
    theta1, theta2, theta3, d = frame
    base, j2, j3, ee = fk_telescopic(theta1, theta2, theta3, d)

    link1.set_data([base[0], j2[0]], [base[1], j2[1]])
    link1.set_3d_properties([base[2], j2[2]])

    link2.set_data([j2[0], j3[0]], [j2[1], j3[1]])
    link2.set_3d_properties([j2[2], j3[2]])

    telescopic.set_data([j3[0], ee[0]], [j3[1], ee[1]])
    telescopic.set_3d_properties([j3[2], ee[2]])

    joints.set_data([base[0], j2[0], j3[0], ee[0]],
                    [base[1], j2[1], j3[1], ee[1]])
    joints.set_3d_properties([base[2], j2[2], j3[2], ee[2]])

    return link1, link2, telescopic, joints

ani = FuncAnimation(fig, update, frames=all_frames, interval=100, blit=False)
plt.show()
