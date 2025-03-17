# import scipy.io as sio
# import numpy as np
# from scipy.signal import butter, filtfilt
# import matplotlib.pyplot as plt



# # Load structured data from the .mat file
# mat_data = sio.loadmat("C:/PHD/Matlab/Radar/matlab_radar/data26/env1/9.mat")

# # Ensure 'pc' exists
# if 'pc' not in mat_data:
#     raise ValueError("The file does not contain the expected variable 'pc'.")

# pc = mat_data['pc'][0]  # Access the struct array

# num_frames = len(pc)  # Number of frames
# fs = 18  # Radar sampling frequency (Hz)
# fc = 2   # Cutoff frequency for low-pass filtering (Hz)

# # Design Butterworth filter
# Wn = fc / (fs / 2)  # Normalized cutoff frequency
# b, a = butter(4, Wn, btype='low')  # 4th order low-pass filter

# motion_energy = np.zeros(num_frames)  # Initialize motion energy array

# # Compute raw motion energy (sum of absolute Doppler values per frame)
# for i in range(num_frames):
#     if 'dopp' in pc[i].dtype.names and pc[i]['dopp'].size > 0:
#         dopp_values = pc[i]['dopp'].flatten()  # Extract Doppler values and flatten
#         motion_energy[i] = np.sum(np.abs(dopp_values))  # Sum of absolute Doppler values

# # Apply low-pass filtering to motion energy
# filtered_motion_energy = filtfilt(b, a, motion_energy)

# # Load Kinect skeleton data
# skeleton_frames = np.load("dataset/env1/subjects/subject26/origal/1/body_skeleton.npy")

# # Compute frame-to-frame differences, combining x, y, and z coordinates
# velocity = np.linalg.norm(np.diff(skeleton_frames, axis=0), axis=2)  # Shape (N-1, M)

# # Compute overall motion energy for each frame
# kinect_motion_energy = np.sum(velocity, axis=1)  # Shape (N-1,)

# # Insert 0 at the first frame to maintain frame count consistency
# kinect_motion_energy = np.insert(kinect_motion_energy, 0, 0)

# # Create a single figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Plot Kinect motion energy
# ax1.plot(kinect_motion_energy, color='b', label='Kinect Motion Energy')
# ax1.set_title('Kinect Motion Energy')
# ax1.legend()
# ax1.grid(True)

# # Plot Radar motion energy
# ax2.plot(filtered_motion_energy, 'b', linewidth=1.5, label='Filtered Motion Energy')
# ax2.plot(motion_energy, 'r', linewidth=1, label='Raw Motion Energy')
# ax2.set_title('Radar Motion Energy')
# ax2.legend()
# ax2.grid(True)

# # Common labels
# plt.xlabel('Frame Index')
# plt.ylabel('Motion Energy')
# plt.tight_layout()
# plt.show()

# print('Motion energy calculation, filtering, and visualization completed.')

import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load structured data from the .mat file
mat_data = sio.loadmat("C:/PHD/Matlab/Radar/matlab_radar/data26/env1/9.mat")

# Ensure 'pc' exists
if 'pc' not in mat_data:
    raise ValueError("The file does not contain the expected variable 'pc'.")

pc = mat_data['pc'][0]  # Access the struct array

num_frames = len(pc)  # Number of frames
fs = 18  # Radar sampling frequency (Hz)
fc = 2   # Cutoff frequency for low-pass filtering (Hz)

# Design Butterworth filter
Wn = fc / (fs / 2)  # Normalized cutoff frequency
b, a = butter(4, Wn, btype='low')  # 4th order low-pass filter

motion_energy = np.zeros(num_frames)  # Initialize motion energy array

# Compute raw motion energy (sum of absolute Doppler values per frame)
for i in range(num_frames):
    if 'dopp' in pc[i].dtype.names and pc[i]['dopp'].size > 0:
        dopp_values = pc[i]['dopp'].flatten()  # Extract Doppler values and flatten
        motion_energy[i] = np.sum(np.abs(dopp_values))  # Sum of absolute Doppler values

# Apply low-pass filtering to motion energy
filtered_motion_energy = filtfilt(b, a, motion_energy)

# Load Kinect skeleton data
skeleton_frames = np.load("dataset/env1/subject26/origal/1/body_skeleton.npy")

# Compute frame-to-frame differences, combining x, y, and z coordinates
velocity = np.linalg.norm(np.diff(skeleton_frames, axis=0), axis=2)  # Shape (N-1, M)

# Compute overall motion energy for each frame
kinect_motion_energy = np.sum(velocity, axis=1)  # Shape (N-1,)

# Insert 0 at the first frame to maintain frame count consistency
kinect_motion_energy = np.insert(kinect_motion_energy, 0, 0)

# Plot Kinect motion energy
plt.figure(figsize=(10, 4))
plt.plot(kinect_motion_energy, color='#1f77b4', label='Kinect Motion Energy', linewidth=2)
plt.title('Kinect Motion Energy')
plt.xlabel('Frame Index')
plt.ylabel('Motion Energy')
plt.legend()
plt.grid(True)
plt.savefig('kinect_motion_energy.png', dpi=300)  # Save high-resolution image
plt.show()

# Plot Radar motion energy
plt.figure(figsize=(10, 4))
plt.plot(filtered_motion_energy, '#d62728', label='Filtered Motion Energy',linewidth=2)
plt.title('Radar Motion Energy')
plt.xlabel('Frame Index')
plt.ylabel('Motion Energy')
plt.legend()
plt.grid(True)
plt.savefig('radar_motion_energy.png', dpi=300)  # Save high-resolution image
plt.show()

print('Motion energy calculation, filtering, and visualization completed.')

# import h5py

# def display_point_attribute_names(file_path):
#     """
#     显示 HDF5 文件中点的属性名称。

#     参数:
#         file_path (str): HDF5 文件的路径。
#     """
#     with h5py.File(file_path, "r") as h5f:
#         # 检查是否存在点级属性名称
#         if "point_attributes" in h5f.attrs:
#             point_attributes = h5f.attrs["point_attributes"]
#             print("点的属性名称:", point_attributes)
#         else:
#             print("未找到点的属性名称。")

# # # 示例用法
# file_path = "dataset/env1/subjects/subject02/aligned/action01/aligned_radar_segment01.h5"
# display_point_attribute_names(file_path)