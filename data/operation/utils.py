import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import pickle

def save_action_segments(segments, filename="action_segments.txt"):
    """
    保存动作分割点到文件
    """
    with open(filename, "w") as f:
        for start, end in segments:
            f.write(f"{start},{end}\n")
    print(f"动作分割信息已保存至 {filename}")


def save_skeleton_segments(skeleton_frames, action_segments, folder_path="dataset"):

     # **创建 skeleton_segments 目录** 先不需要人工核对之后才需要
    skeleton_segments_dir = os.path.join(folder_path, "skeleton_segments")
    os.makedirs(skeleton_segments_dir, exist_ok=True)

    # **存储动作分割的骨骼点，文件命名为 `action01.npy`, `action02.npy`...**
    for idx, (start, end) in enumerate(action_segments, start=1):
        segment_data = skeleton_frames[start:end + 1]
        segment_filename = os.path.join(skeleton_segments_dir, f"action{idx:02d}.npy")  # action01.npy, action02.npy...
        np.save(segment_filename, segment_data)
        
def save_time_segments(timestamps, action_segments, folder_path="dataset"):
    skeleton_segments_dir = os.path.join(folder_path, "skeleton_segments")
    os.makedirs(skeleton_segments_dir, exist_ok=True)

    # **存储动作分割的时间点，文件命名为 `timestamp01.npy`, `timestamp02.npy`...**
    for idx, (start, end) in enumerate(action_segments, start=1):
        segment_data = timestamps[start:end + 1]
        time_filename = os.path.join(skeleton_segments_dir, f"timestamp{idx:02d}.npy")  # action01.npy, action02.npy...
        np.save(time_filename, segment_data)

def plot_motion_energy(motion_energy, action_segments, threshold_T, long_stationary_segments, save_path="plots/motion_energy_plot.png"):
    """
    绘制运动能量曲线并标注动作分割线，并将图像保存到指定路径。
    
    :param motion_energy: 每一帧的运动能量
    :param action_segments: 分割后的动作段列表
    :param threshold_T: 静止状态的阈值
    :param long_stationary_segments: 长静止段的起始和结束帧索引列表
    :param save_path: 保存图片的路径（默认为 "plots/motion_energy_plot.png"）
    """
    plt.figure(figsize=(12, 6))

    # 绘制运动能量曲线
    plt.plot(motion_energy, label="Motion Energy", color="blue")

    # 绘制静止状态阈值线
    plt.axhline(y=threshold_T, color="red", linestyle="--", label="Threshold (T)")

    # 标注动作分割线
    for segment in action_segments:
        start_frame = segment[0]
        end_frame = segment[-1]
        plt.axvline(x=start_frame, color="green", linestyle="--", alpha=0.5)
        plt.axvline(x=end_frame, color="green", linestyle="--", alpha=0.5)
        plt.fill_betweenx(
            y=[0, max(motion_energy)],
            x1=start_frame,
            x2=end_frame,
            color="green",
            alpha=0.1,
            label="Action Segment" if segment == action_segments[0] else "",
        )

    # 标注长静止段
    for start, end in long_stationary_segments:
        plt.axvspan(start, end, color="gray", alpha=0.3, label="Long Stationary Segment" if start == long_stationary_segments[0][0] else "")

    # 添加图例和标签
    plt.xlabel("Frame Index")
    plt.ylabel("Motion Energy")
    plt.title("Motion Energy and Action Segmentation")
    plt.legend()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path, dpi=300)
    print(f"运动能量图已保存至: {save_path}")


def transform_large_point_set(large_points_file):
    with open('rigid_transform.pkl', 'rb') as f:
        R, t = pickle.load(f)

    large_points = pd.read_excel(large_points_file).to_numpy()

    large_points[:, 0] *= -1
    large_points[:, 2] *= -1
    large_points[:, 2] += 9

    return large_points @ R.T + t



