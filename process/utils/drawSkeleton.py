import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy.io as sio

# Load skeleton data (format: frames x 32 x 3)
skeleton_data = np.load("dataset/env1/subjects/subject26/origal/1/aligned/aligned_skeleton_segment01.npy")  # Path to your .npy file
radar_data = sio.loadmat("dataset/env1/subjects/subject26/origal/1/aligned_radar_segment01.mat")
pc = radar_data["radar_data"]  # 888 frames, each with 12 attributes
num_frames = skeleton_data.shape[0]  # Total number of frames

# Skeleton bone connections (torso, limbs)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 26),  # Torso
    (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),  # Left arm
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),  # Right arm
    (0, 18), (18, 19), (19, 20), (20, 21),  # Left leg
    (0, 22), (22, 23), (23, 24), (24, 25)  # Right leg
]

frame_index = 23  # Select a frame for visualization
joints = skeleton_data[frame_index]  # Shape: (32, 3)

# Plot 3D skeleton
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw skeleton connections
for bone in BONES:
    joint1, joint2 = bone
    xs = [joints[joint1, 0], joints[joint2, 0]]
    ys = [joints[joint1, 1], joints[joint2, 1]]
    zs = [joints[joint1, 2], joints[joint2, 2]]
    ax.plot(xs, ys, zs, 'bo-', markersize=5, alpha=0.8)

# Extract radar data for the selected frame
radar_frame_data = pc[0, frame_index]  # Radar data (888, 12, N, 3)
x = (radar_frame_data[7].flatten()) * 1000  # X coordinates
z = (radar_frame_data[8].flatten()) * 1000  # Z coordinates
y = -(radar_frame_data[3].flatten()) * 1000  # Y coordinates
print(y)
ax.scatter(x, y, z, c='r', marker='o', s=10, alpha=0.5)  # Red points

# Set axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-2000, 2000)  # X-axis range
ax.set_ylim(-2000, 2000)  # Y-axis range
ax.set_zlim(0, 5000)      # Z-axis range
ax.set_title("3D Skeleton Visualization")
ax.view_init(elev=10, azim=45)  # Adjust viewing angle

plt.show()

# Animation setup
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Set axis limits
# x_min, x_max = np.min(skeleton_data[:, :, 0]), np.max(skeleton_data[:, :, 0])
# y_min, y_max = np.min(skeleton_data[:, :, 1]), np.max(skeleton_data[:, :, 1])
# z_min, z_max = np.min(skeleton_data[:, :, 2]), np.max(skeleton_data[:, :, 2])
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# ax.set_zlim(z_max, z_min)  # Invert Z-axis for better camera view

# # Set viewing angle
# ax.view_init(elev=-90, azim=-90)

# # Update function
# def update(frame_idx):
#     ax.cla()  # Clear previous frame
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_max, z_min)
#     ax.view_init(elev=-90, azim=-90)

#     joints = skeleton_data[frame_idx]  # Get skeleton data for the current frame

#     # Draw skeleton connections
#     for bone in BONES:
#         joint1, joint2 = bone
#         xs = [joints[joint1, 0], joints[joint2, 0]]
#         ys = [joints[joint1, 1], joints[joint2, 1]]
#         zs = [joints[joint1, 2], joints[joint2, 2]]
#         ax.plot(xs, ys, zs, 'bo-', markersize=5, alpha=0.8)

#     ax.set_title(f"Frame {frame_idx}/{num_frames}")

# # Create animation
# ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

# # Save as GIF or MP4
# ani.save("skeleton_animation.gif", writer="pillow", fps=20)  # GIF
# # ani.save("skeleton_animation.mp4", writer="ffmpeg", fps=20)  # MP4

# plt.show()
