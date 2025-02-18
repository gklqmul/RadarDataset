import numpy as np

# 读取 Body 坐标数据
body_data = np.load("elif/9.npy")
print("Body data shape:", body_data.shape)  # (num_frames, 32, 3)

# 读取 Color/RGB 坐标数据
color_data = np.load("elfi-splited/skeleton_npy8/color_skeleton.npy")
print("Color data shape:", color_data.shape)  # (num_frames, 32, 3)
