import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Robot parameters
L1 = 3.0
L2 = 5.0
L3 = 4.0
d4_min, d4_max = 0.0, 6.0

# -----------------------------
# Custom inverse kinematics (elbow-up)
# -----------------------------
def ik_rrrp(x, y, z):
    theta1 = np.arctan2(y, x)
    r = np.hypot(x, y)
    dz = z - L1
    dist = np.hypot(r, dz)

    # Elbow angle
    cos_theta3 = (L2**2 + L3**2 - dist**2) / (2*L2*L3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)

    # Shoulder angle (elbow-up branch)
    beta = np.arctan2(dz, r)
    gamma = np.arccos((L2**2 + dist**2 - L3**2) / (2*L2*dist))
    theta2 = beta + gamma   # elbow-up

    # Prismatic extension
    d4 = dist - (L2*np.cos(theta2) + L3*np.cos(theta2+theta3))
    d4 = np.clip(d4, d4_min, d4_max)

    return theta1, theta2, theta3, d4

# -----------------------------
# Forward kinematics
# -----------------------------
def fk_rrrp(theta1, theta2, theta3, d4):
    base = np.array([0,0,0])
    P1 = np.array([0,0,L1])

    x2 = P1[0] + L2*np.cos(theta1)*np.cos(theta2)
    y2 = P1[1] + L2*np.sin(theta1)*np.cos(theta2)
    z2 = P1[2] + L2*np.sin(theta2)
    P2 = np.array([x2,y2,z2])

    x3 = P2[0] + L3*np.cos(theta1)*np.cos(theta2+theta3)
    y3 = P2[1] + L3*np.sin(theta1)*np.cos(theta2+theta3)
    z3 = P2[2] + L3*np.sin(theta2+theta3)
    P3 = np.array([x3,y3,z3])

    x4 = P3[0] + d4*np.cos(theta1)*np.cos(theta2+theta3)
    y4 = P3[1] + d4*np.sin(theta1)*np.cos(theta2+theta3)
    z4 = P3[2] + d4*np.sin(theta2+theta3)
    P4 = np.array([x4,y4,z4])

    return base, P1, P2, P3, P4

# -----------------------------
# Targets
# -----------------------------
floor_targets = [(3.0,0.0,0.5), (-1.5,2.598,0.5), (-1.5,-2.598,0.5)]
shelf_targets = [(3.0,0.0,3.0), (-1.5,2.598,4.0), (-1.5,-2.598,5.0)]

sequence = []
labels = []
for i in range(len(floor_targets)):
    sequence.append(floor_targets[i]); labels.append(f"F{i+1}")
    sequence.append(shelf_targets[i]); labels.append(f"S{i+1}")

solutions = [ik_rrrp(x,y,z) for (x,y,z) in sequence]

# -----------------------------
# Interpolated trajectory
# -----------------------------
frames = []
for i in range(len(solutions)-1):
    start = np.array(solutions[i])
    end = np.array(solutions[i+1])
    for t in np.linspace(0,1,40):
        frames.append((1-t)*start + t*end)

# -----------------------------
# Animation
# -----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("RRRP Robot | Floor (red) → Shelf (blue) sequence")

# Plot floor targets (red circles)
for i,(x,y,z) in enumerate(floor_targets, start=1):
    ax.scatter(x,y,z,c="red",s=120,marker="o",edgecolors="black")
    ax.text(x,y,z+0.2,f"F{i}",color="red",fontsize=9,ha="center")

# Plot shelf targets (blue triangles)
for i,(x,y,z) in enumerate(shelf_targets, start=1):
    ax.scatter(x,y,z,c="blue",s=120,marker="^",edgecolors="black")
    ax.text(x,y,z+0.2,f"S{i}",color="blue",fontsize=9,ha="center")

link1_line, = ax.plot([], [], [], 'b-', linewidth=3)
link2_line, = ax.plot([], [], [], 'g-', linewidth=3)
link3_line, = ax.plot([], [], [], 'orange', linewidth=3)
prism_line, = ax.plot([], [], [], 'm-', linewidth=3)
joints, = ax.plot([], [], [], 'ro', markersize=6)

reach = L1+L2+L3+d4_max
ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach); ax.set_zlim(0, reach)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

def update(frame):
    theta1,theta2,theta3,d4 = frame
    base,P1,P2,P3,P4 = fk_rrrp(theta1,theta2,theta3,d4)

    link1_line.set_data([base[0],P1[0]],[base[1],P1[1]])
    link1_line.set_3d_properties([base[2],P1[2]])

    link2_line.set_data([P1[0],P2[0]],[P1[1],P2[1]])
    link2_line.set_3d_properties([P1[2],P2[2]])

    link3_line.set_data([P2[0],P3[0]],[P2[1],P3[1]])
    link3_line.set_3d_properties([P2[2],P3[2]])

    prism_line.set_data([P3[0],P4[0]],[P3[1],P4[1]])
    prism_line.set_3d_properties([P3[2],P4[2]])

    joints.set_data([base[0],P1[0],P2[0],P3[0],P4[0]],
                    [base[1],P1[1],P2[1],P3[1],P4[1]])
    joints.set_3d_properties([base[2],P1[2],P2[2],P3[2],P4[2]])

    return link1_line, link2_line, link3_line, prism_line, joints

ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False, repeat=True)
plt.show()
