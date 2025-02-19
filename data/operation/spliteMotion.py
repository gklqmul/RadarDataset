import os
import numpy as np
from ActionSegmenter import ActionSegmenter  # 确保你有这个类
from utils import save_action_segments, save_skeleton_segments, plot_motion_energy, save_time_segments  # 相关的工具函数

# 设定根目录，按照规则查找多个 `1/body_skeleton.npy`, `2/body_skeleton.npy` ... `6/body_skeleton.npy`
BASE_DIR = "dataset/env2/subjects/subject26/origal"

def find_skeleton_files(base_dir, num_folders=6):
    """
    在 `base_dir` 目录中查找 `1/body_skeleton.npy` ~ `6/body_skeleton.npy` 并返回完整路径列表。
    """
    skeleton_files = []
    
    # 遍历 `1` 到 `6` 这些文件夹
    for i in range(7, num_folders + 7):  
        folder_path = os.path.join(base_dir, str(i))  # 生成 `1/`, `2/`, ..., `6/`
        skeleton_file = os.path.join(folder_path, "body_skeleton.npy")
        
        if os.path.exists(skeleton_file):
            skeleton_files.append((i, folder_path, skeleton_file))  # 存储 (文件夹索引, 文件夹路径, 文件路径)
        else:
            print(f"未找到文件: {skeleton_file}")

    return skeleton_files

def process_all_npy(base_dir, num_folders=6):
    """
    处理文件夹内所有 .npy 文件，并获取对应的骨骼点数据
    """
    # 获取所有 .npy 文件
    timestamp_file = []

    # 遍历 `1` 到 `6` 这些文件夹
    for i in range(7, num_folders + 7):  
        folder_path = os.path.join(base_dir, str(i))  # 生成 `1/`, `2/`, ..., `6/`
        timestamp_file = os.path.join(folder_path, i, ".npy")
        
        if os.path.exists(timestamp_file):
            timestamp_file.append((i, folder_path, timestamp_file))  # 存储 (文件夹索引, 文件夹路径, 文件路径)
        else:
            print(f"未找到文件: {timestamp_file}")

    return timestamp_file
    

def process_skeleton_file(folder_index, folder_path, skeleton_file):
    """
    处理单个 `.npy` 文件，进行动作分割并将结果存储到对应文件夹中。
    """
    print(f"正在处理: {skeleton_file}")

    # 读取骨骼数据
    skeleton_frames = np.load(skeleton_file, allow_pickle=True)
    
    # 创建动作分割器
    segmenter = ActionSegmenter(frame_Count=len(skeleton_frames), min_segment_length=60, long_stationary_threshold=40, smooth_sigma=1.0)

    # 进行动作分割
    motion_energy, action_segments = segmenter.segment_actions(skeleton_frames)

    # 检测长静止段
    is_stationary = motion_energy < segmenter.threshold_T
    long_stationary_segments = segmenter._find_long_stationary_segments(is_stationary)

    # **在当前文件夹 `1/`, `2/` ... `6/` 下面存储结果**
    action_segments_path = os.path.join(folder_path, "action_segments.txt")
    save_action_segments(action_segments, filename=action_segments_path)
    save_skeleton_segments(skeleton_frames, action_segments, folder_path)

    # **在当前文件夹 `1/`, `2/` ... `6/` 下面存储 `motion_energy.png`**
    energy_plot_path = os.path.join(folder_path, "motion_energy.png")
    plot_motion_energy(motion_energy, action_segments, segmenter.threshold_T, long_stationary_segments, save_path=energy_plot_path)

   
    print(f"处理完成: {skeleton_file}，结果已保存到 {folder_path}/\n")

if __name__ == "__main__":
    # 获取所有待处理的 `.npy` 文件
    skeleton_files = find_skeleton_files(BASE_DIR)

    if not skeleton_files:
        print("没有找到符合规则的 body_skeleton.npy 文件！")
    else:
        print(f"找到 {len(skeleton_files)} 个文件，开始处理...")
        # 逐个处理找到的 `.npy` 文件
        for folder_index, folder_path, skeleton_file in skeleton_files:
            process_skeleton_file(folder_index, folder_path, skeleton_file)
