import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # ensure interactive backend for animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Robot parameters
# -----------------------------
L1, L2 = 4.0, 3.5            # link lengths
d_min, d_max = 0.0, 7.0      # prismatic limits
max_radial = L1 + L2 + d_max # workspace radius for plotting

# -----------------------------
# Targets (floor + shelves)
# -----------------------------
floor_targets = [
    (3.0, 0.0, 0.5),
    (-1.5, 2.598, 0.5),
    (-1.5, -2.598, 0.5)
]
shelf_targets = [
    (3.0, 0.0, 3.0),
    (-1.5, 2.598, 4.0),
    (-1.5, -2.598, 5.0)
]

# Build motion sequence: F1 → S1 → F2 → S2 → F3 → S3
sequence = []
for f, s in zip(floor_targets, shelf_targets):
    sequence.append(f)
    sequence.append(s)

# -----------------------------
# Geometric forward kinematics (RRRP)
# -----------------------------
def fk_rrrp(theta1, theta2, theta3, d):
    """
    Base at origin. Yaw by theta1 sets the azimuth in XY.
    Elevation is theta2 + theta3. The prismatic d extends along the tool axis.
    Returns base, j2, j3, ee positions.
    """
    # Direction unit vectors
    c1, s1 = np.cos(theta1), np.sin(theta1)
    elev2 = theta2
    elev3 = theta2 + theta3

    # Link direction unit vectors in 3D
    # Link 1 along (theta1, theta2)
    u1 = np.array([c1 * np.cos(elev2), s1 * np.cos(elev2), np.sin(elev2)])
    # Link 2 along (theta1, theta2+theta3)
    u2 = np.array([c1 * np.cos(elev3), s1 * np.cos(elev3), np.sin(elev3)])
    # Tool axis is along u2
    ut = u2.copy()

    base = np.array([0.0, 0.0, 0.0])
    j2 = L1 * u1
    j3 = j2 + L2 * u2
    ee = j3 + d * ut
    return base, j2, j3, ee

# -----------------------------
# IK strategy to ensure J4 (prismatic) touches the target
# -----------------------------
def ik_rrrp_pointed_by_tool(x, y, z):
    """
    Orient tool axis toward the target ray, then choose elbow bend so that
    J4 extension d (>=0) reaches the target exactly:
      1) theta1 = atan2(y, x) (azimuth)
      2) alpha = atan2(z, sqrt(x^2 + y^2)) (target elevation)
      3) Choose theta3 to set how much of the rigid chain lies along the tool axis:
         reach_axis = L1*cos(theta3) + L2
         We need d = dist - reach_axis in [d_min, d_max] and d >= 0.
         Also cos(theta3) must be in [-1, 1] ⇒ reach_axis ∈ [L2 - L1, L2 + L1].
      4) theta2 = alpha - theta3 (so tool axis elevation = alpha).
    """
    r_xy = np.hypot(x, y)
    dist = np.hypot(r_xy, z)

    # Azimuth and elevation of the target direction
    theta1 = np.arctan2(y, x)
    alpha = np.arctan2(z, r_xy)

    # Desired axial reach along tool axis to keep d positive and within limits:
    # Aim for mid-range d to avoid edge cases, then clamp to feasible band.
    d_desired = 0.5 * d_max
    reach_axis_target = dist - d_desired

    # Feasible axial reach band from cos(theta3) constraint
    reach_min = L2 - L1
    reach_max = L2 + L1

    # Clamp reach to feasible band
    reach_axis = np.clip(reach_axis_target, reach_min, reach_max)

    # Compute theta3 from reach_axis = L1*cos(theta3) + L2
    cos_t3 = (reach_axis - L2) / L1
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    theta3 = np.arccos(cos_t3)  # elbow-down; use -arccos for elbow-up if needed

    # Now set shoulder so that tool elevation is alpha
    theta2 = alpha - theta3

    # Compute actual d required to hit target exactly
    # With tool axis aligned to alpha, the axial projection of j3 along ut is:
    # reach_axis = L1*cos(theta3) + L2
    d = dist - (L1 * np.cos(theta3) + L2)
    d = np.clip(d, d_min, d_max)

    return theta1, theta2, theta3, d

# -----------------------------
# Solve poses for the sequence
# -----------------------------
poses = []
for (x, y, z) in sequence:
    theta1, theta2, theta3, d = ik_rrrp_pointed_by_tool(x, y, z)
    poses.append((theta1, theta2, theta3, d))

# -----------------------------
# Plot setup
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-max_radial-2, max_radial+2)
ax.set_ylim(-max_radial-2, max_radial+2)
ax.set_zlim(0, max(8.0, max([p[2] for p in floor_targets + shelf_targets]) + 2.0))
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("Articulated RRRP | J4 touches targets (Floor → Shelf sequence)")

# Targets
for i, (x, y, z) in enumerate(floor_targets, start=1):
    ax.scatter(x, y, z, c='red', marker='o', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"F{i}", color='red', fontsize=9, ha='center')
for i, (x, y, z) in enumerate(shelf_targets, start=1):
    ax.scatter(x, y, z, c='blue', marker='^', s=120, edgecolors='black')
    ax.text(x, y, z + 0.2, f"S{i}", color='blue', fontsize=9, ha='center')

# Robot graphics
link1, = ax.plot([], [], [], 'g', linewidth=4, label='Link 1')
link2, = ax.plot([], [], [], 'b', linewidth=4, label='Link 2')
prism, = ax.plot([], [], [], 'k', linewidth=4, label='Prismatic J4')
joints, = ax.plot([], [], [], 'ro', markersize=8, label='Joints')
ax.legend()

# -----------------------------
# Build animation frames (joint-space interpolation)
# -----------------------------
frames_per_segment = 40
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
    base, j2, j3, ee = fk_rrrp(theta1, theta2, theta3, d)

    # Link 1
    link1.set_data([base[0], j2[0]], [base[1], j2[1]])
    link1.set_3d_properties([base[2], j2[2]])

    # Link 2
    link2.set_data([j2[0], j3[0]], [j2[1], j3[1]])
    link2.set_3d_properties([j2[2], j3[2]])

    # Prismatic (visualize from j3 to ee)
    prism.set_data([j3[0], ee[0]], [j3[1], ee[1]])
    prism.set_3d_properties([j3[2], ee[2]])

    # Joints
    joints.set_data([base[0], j2[0], j3[0], ee[0]],
                    [base[1], j2[1], j3[1], ee[1]])
    joints.set_3d_properties([base[2], j2[2], j3[2], ee[2]])

    return link1, link2, prism, joints

ani = FuncAnimation(fig, update, frames=all_frames, interval=200, blit=False, repeat=True)
plt.show()
