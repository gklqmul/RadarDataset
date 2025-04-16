# import numpy as np

# # 读取 Body 坐标数据
# body_data = np.load("C:/PHD/pyKinectAzure/data/operation/dataset/env1/subjects/subject26/origal/1/skeleton_segments/timestamp1.npy")
# print("Body data shape:", body_data.shape, body_data[1])  # (num_frames, 32, 3)

# # 读取 Color/RGB 坐标数据
# # color_data = np.load("data/operation/dataset/env1/subjects/subject26/origal/1/body_skeleton.npy")
# # print("Color data shape:", color_data.shape)  # (num_frames, 32, 3)

import h5py

# 1. 打开HDF5文件
file_path = 'dataset/env1/subjects/subject01/aligned/action01/aligned_radar_segment01.h5' 
with h5py.File(file_path, 'r') as h5_file:
    
    # 2. 获取文件中的所有顶层对象(数据集/组)名称
    print("文件中的顶层对象:")
    for name in h5_file.keys():
        print(f"- {name}")
        
        # 3. 获取每个对象的属性
        obj = h5_file[name]
        if obj.attrs:  # 如果对象有属性
            print(f"  属性列表:")
            for attr_name in obj.attrs.keys():
                print(f"  - {attr_name}")
        else:
            print("  无属性")
            
    # 4. 获取文件本身的全局属性(如果有)
    if h5_file.attrs:
        print("\n文件全局属性:")
        for attr_name in h5_file.attrs.keys():
            print(f"- {attr_name}")
    else:
        print("\n文件无全局属性")

    