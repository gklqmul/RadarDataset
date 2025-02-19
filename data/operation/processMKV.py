import os
import numpy as np
import os
from KinectSkeletonProcessor import KinectSkeletonProcessor  
from data.operation import ActionSegmenter
from data.operation.utils import plot_motion_energy, save_action_segments, save_skeleton_segments

# 设置存放 MKV 文件的文件夹
DATA_FOLDER = "C:\PHD\elif"  # 替换成你的文件夹路径

def process_all_videos(folder_path):
    """
    处理文件夹内所有 .mkv 文件，并获取对应的骨骼点数据
    """
    # 获取所有 .mkv 文件
    mkv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mkv")])

    # 确保有 .mkv 文件
    if not mkv_files:
        print("没有找到任何 .mkv 文件！")
        return

    print(f"找到 {len(mkv_files)} 个 .mkv 文件，开始处理...")
    
    # 依次处理每个 .mkv 文件
    for index, mkv_file in enumerate(mkv_files, start=1):
        video_path = os.path.join(folder_path, mkv_file)
        
        # 创建保存目录（subject26/origal/index）
        save_dir = os.path.join("dataset\env1\subjects\subject26", "origal", str(index))
        os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

        print(f"正在处理: {video_path} 保存目录: {save_dir}")
        
        # 创建处理器并传递 save_dir
        processor = KinectSkeletonProcessor(video_path, save_dir)  # 将 save_dir 传递给类
        processor.process_video()

        print(f"处理完成: {mkv_file}\n")

if __name__ == "__main__":

    process_all_videos(DATA_FOLDER)


