import numpy as np

# 读取 Body 坐标数据
body_data = np.load("C:/PHD/pyKinectAzure/data/operation/dataset/env1/subjects/subject26/origal/1/skeleton_segments/timestamp1.npy")
print("Body data shape:", body_data.shape, body_data[1])  # (num_frames, 32, 3)

# 读取 Color/RGB 坐标数据
# color_data = np.load("data/operation/dataset/env1/subjects/subject26/origal/1/body_skeleton.npy")
# print("Color data shape:", color_data.shape)  # (num_frames, 32, 3)
