

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def get_aligned_mat_files(base_dir):
    """
    Recursively finds all .mat files in the `aligned` folders under the specified base directory.

    Args:
        base_dir (str): The base directory path, e.g., "dataset/env1/subjects/subject02/origal".

    Returns:
        list: A list of paths to all found .mat files.
    """
    # Convert base_dir to a Path object
    base_dir = Path(base_dir)

    # Find all .mat files in the `aligned` folders
    mat_files = list(base_dir.glob("**/aligned/*.mat"))

    # Convert Path objects to string paths
    mat_files = [str(file) for file in mat_files]

    return mat_files


def find_files(base_dir, extension):
    """
    Search for files with a specific extension in a given directory.

    Args:
        base_dir (str): The root directory to search.
        extension (str): The file extension to filter (e.g., '.npy', '.mkv').

    Returns:
        list: A list of full file paths.
    """
    found_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    found_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return found_files

def find_files_fromsubfolder(base_dir, file_name_template, num_folders=6, start_index=1):
    """
    Find files with a configurable name pattern in a series of indexed folders.

    Args:
        base_dir (str): The root directory containing the indexed subfolders.
        file_name_template (str): The file name pattern, with `{index}` as a placeholder for folder index.
        num_folders (int): The number of indexed folders to search.
        start_index (int): The starting index for the folders.

    Returns:
        list: A list of tuples (folder index, folder path, file path).
    """

    ''' Example usage
    if __name__ == "__main__":
        base_dir = "dataset/env2/subjects/subject26/origal"
        file_name_template = "{index}/body_skeleton.npy"  # Configurable pattern
        files = find_files(base_dir, file_name_template)

        if not files:
            print("No files found!")
        else:
            print(f"Found {len(files)} files:")
            for folder_index, folder_path, file_path in files:
                print(f"{file_path}")
    '''

    found_files = []
    # Iterate over the indexed folders
    for i in range(start_index, num_folders + start_index):  
        folder_path = os.path.join(base_dir, str(i))
        file_name = file_name_template.format(index=i)
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            found_files.append(( folder_path, file_path))  # Store (folder path, file path)
        else:
            print(f"File not found: {file_path}")

    return found_files

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



