import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Robot Parameters
# -----------------------------
L1, L2 = 4.0, 5.5
d_min, d_max = 0.0, 8.0

# -----------------------------
# Target Points
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
# Inverse Kinematics
# -----------------------------
def ik_prismatic(x, y, z):
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    dist = np.sqrt(r_xy**2 + z**2)

    cos_theta3 = (dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(cos_theta3) > 1:
        raise ValueError("Target out of reach")
    theta3 = np.arccos(cos_theta3)

    k1 = L1 + L2 * np.cos(theta3)
    k2 = L2 * np.sin(theta3)
    theta2 = np.arctan2(z, r_xy) - np.arctan2(k2, k1)

    reach = L1 * np.cos(theta2) + L2 * np.cos(theta2 + theta3)
    d = dist - reach
    if d < d_min or d > d_max:
        raise ValueError("Prismatic extension out of range")

    return theta1, theta2, theta3, d

# -----------------------------
# Forward Kinematics
# -----------------------------
def fk_prismatic(theta1, theta2, theta3, d):
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
# Compute IK for all poses
# -----------------------------
poses = []
for f, s in zip(floor_targets, shelf_targets):
    poses.append(ik_prismatic(*f))
    poses.append(ik_prismatic(*s))

# -----------------------------
# Animation Setup
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_zlim(0, 8)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("3R + Prismatic Wrist | Floor to Shelf Stacking")

# Plot targets
for i, (x, y, z) in enumerate(floor_targets, start=1):
    ax.scatter(x, y, z, c='red', marker='o', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"F{i}", color='red', fontsize=9, ha='center')

for i, (x, y, z) in enumerate(shelf_targets, start=1):
    ax.scatter(x, y, z, c='blue', marker='^', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"S{i}", color='blue', fontsize=9, ha='center')

link1, = ax.plot([], [], [], 'b', linewidth=4, label='Link 1')
link2, = ax.plot([], [], [], 'g', linewidth=4, label='Link 2')
prism, = ax.plot([], [], [], 'm', linewidth=4, label='Prismatic Wrist')
joints, = ax.plot([], [], [], 'ro', markersize=8, label='Joints')
ax.legend()

# Interpolation
frames_per_segment = 30
all_frames = []
for i in range(len(poses) - 1):
    start = np.array(poses[i])
    end = np.array(poses[i + 1])
    for t in np.linspace(0, 1, frames_per_segment):
        all_frames.append((1 - t) * start + t * end)

# -----------------------------
# Update Function
# -----------------------------
def update(frame):
    theta1, theta2, theta3, d = frame
    base, j2, j3, ee = fk_prismatic(theta1, theta2, theta3, d)

    link1.set_data([base[0], j2[0]], [base[1], j2[1]])
    link1.set_3d_properties([base[2], j2[2]])

    link2.set_data([j2[0], j3[0]], [j2[1], j3[1]])
    link2.set_3d_properties([j2[2], j3[2]])

    prism.set_data([j3[0], ee[0]], [j3[1], ee[1]])
    prism.set_3d_properties([j3[2], ee[2]])

    joints.set_data([base[0], j2[0], j3[0], ee[0]],
                    [base[1], j2[1], j3[1], ee[1]])
    joints.set_3d_properties([base[2], j2[2], j3[2], ee[2]])

    return link1, link2, prism, joints

ani = FuncAnimation(fig, update, frames=all_frames, interval=100, blit=False)
plt.show()
