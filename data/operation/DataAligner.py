import os
import numpy as np
import scipy.io

# 定义文件夹路径
skeleton_folder = r"C:/PHD/pyKinectAzure/data/operation/dataset/env1/subjects/subject26/origal/1/skeleton_segments"
radar_file_path = r"C:/PHD/Matlab/Radar/matlab_radar/elif/9.mat"  # 只有一个雷达文件

# 获取所有 actionX.npy 和 timestampX.npy，确保按顺序匹配
action_files = sorted([f for f in os.listdir(skeleton_folder) if f.startswith("action") and f.endswith(".npy")])
timestamp_files = sorted([f for f in os.listdir(skeleton_folder) if f.startswith("timestamp") and f.endswith(".npy")])

def load_radar_timestamps(mat_file_path):
    """加载 .mat 文件中的雷达时间戳，并转换为浮点数"""
    mat_data = scipy.io.loadmat(mat_file_path)
    
    if 'pc' in mat_data:  # 假设 MATLAB 变量名为 radar_data
        radar_struct = mat_data['pc']  # 1×818 结构体数组
    else:
        raise ValueError(f"❌ '{mat_file_path}' 文件中找不到 'pc' 变量！")

    num_frames = radar_struct.shape[1]  # 获取帧数 (1×818)
    
    timestamps = np.zeros(num_frames)

    # 解析时间戳
    for i in range(num_frames):
        timestamps[i] = float(radar_struct[0, i]['timestamp'][0])  # 将字符串转换为浮点数

    return timestamps, radar_struct

def filter_radar_by_kinect(radar_timestamps, kinect_timestamps):
    """保留雷达和 Kinect 同时采集的部分"""
    kinect_start = np.min(kinect_timestamps)  # Kinect 采集起始时间
    kinect_end = np.max(kinect_timestamps)    # Kinect 采集终止时间

    # 找到雷达数据的有效索引范围
    start_idx = np.searchsorted(radar_timestamps, kinect_start, side="left")
    end_idx = np.searchsorted(radar_timestamps, kinect_end, side="right") - 1

    # 如果 Kinect 时间范围与雷达数据没有重叠，返回空
    if start_idx > end_idx:
        return None, None  # 没有有效区间

    return start_idx, end_idx


def match_kinect_to_radar(kinect_timestamps, radar_timestamps, kinect_data):
    """使 Kinect 帧数与 Radar 帧数对齐，删除多余 Kinect 帧"""
    kinect_timestamps = np.array(kinect_timestamps)
    radar_timestamps = np.array(radar_timestamps)

    # 找到每个雷达帧最接近的 Kinect 帧
    matched_indices = np.searchsorted(kinect_timestamps, radar_timestamps, side="left")
    
    # 处理边界情况
    for i in range(len(matched_indices)):
        if matched_indices[i] == len(kinect_timestamps):  
            matched_indices[i] -= 1
        elif matched_indices[i] > 0:
            left = kinect_timestamps[matched_indices[i] - 1]
            right = kinect_timestamps[matched_indices[i]]
            if abs(left - radar_timestamps[i]) < abs(right - radar_timestamps[i]):
                matched_indices[i] -= 1

    # 根据匹配的索引提取 Kinect 数据
    matched_kinect_data = kinect_data[matched_indices]  # (radar_frame_count, 32, 3)
    return matched_kinect_data

def process_files(action_path, timestamp_path, radar_timestamps, radar_struct, segment_id):
    """按 timestampX.npy 处理 Kinect 和 Radar 数据"""
    timestamp_data = np.load(timestamp_path)  # 读取 timestampX.npy
    kinect_data = np.load(action_path)  # 读取 actionX.npy

    # 过滤雷达数据，仅保留 Kinect 采集时间范围内的部分
    start_idx, end_idx = filter_radar_by_kinect(radar_timestamps, timestamp_data)
    
    if start_idx is None or end_idx is None:
        print(f"⚠️ 由于 Kinect 和雷达的时间戳范围不重叠，跳过 {timestamp_path} 和 {action_path} 的处理。")
        return  # 如果没有有效数据，跳过处理

    radar_timestamps_segment = radar_timestamps[start_idx:end_idx+1]
    radar_data_segment = radar_struct[:, start_idx:end_idx+1]  # 选择符合时间段的雷达帧

    # 筛选 Kinect 数据，仅保留同时采集的部分
    valid_kinect_indices = (timestamp_data >= radar_timestamps_segment[0]) & (timestamp_data <= radar_timestamps_segment[-1])
    kinect_timestamps = timestamp_data[valid_kinect_indices]  # 筛选 Kinect 时间戳
    kinect_data = kinect_data[valid_kinect_indices]  # 筛选 Kinect 数据

    # 使 Kinect 帧与雷达帧对齐
    aligned_kinect_data = match_kinect_to_radar(kinect_timestamps, radar_timestamps_segment, kinect_data)

    # 保存对齐后的 Kinect 数据
    aligned_kinect_path = os.path.join(skeleton_folder, f"aligned_action{segment_id:02d}.npy")
    np.save(aligned_kinect_path, aligned_kinect_data)
    print(f"✅ 已保存对齐 Kinect 数据: {aligned_kinect_path}")

    # 保存分割后的雷达数据
    radar_save_path = os.path.join(os.path.dirname(radar_file_path), f"filtered_9_segment{segment_id:02d}.mat")
    scipy.io.savemat(radar_save_path, {'radar_data': radar_data_segment})
    print(f"✅ 已保存分割后的雷达数据: {radar_save_path}")


# 读取雷达时间戳
radar_timestamps, radar_struct = load_radar_timestamps(radar_file_path)

# 处理所有 actionX.npy 和 timestampX.npy
for segment_id, (action_file, timestamp_file) in enumerate(zip(action_files, timestamp_files), start=1):
    action_path = os.path.join(skeleton_folder, action_file)
    timestamp_path = os.path.join(skeleton_folder, timestamp_file)

    process_files(action_path, timestamp_path, radar_timestamps, radar_struct, segment_id)

print("🎯 所有数据已对齐并分割！")
