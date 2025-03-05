# get skeleton from mkv file
# split skeleton into different motions find timestamps
# align radar data with skeleton data based on timestamps
# split kinect and radar data into segments
# transform each to radar coordinate


from process.class_files.data_aligner import DataAligner
from process.utils.ProcessAllMKV import process_all_mkv
from process.utils.SaveMotionSplit import process_skeleton_file
from process.utils.FindAndSave import find_files, find_files_fromsubfolder, get_aligned_mat_files
from process.utils.reformatradardata import convert_mat_to_hdf5
import os
import shutil

if __name__ == "__main__":
    # mkv_folder = "D:/kinect/data05"
    # mkv_flies = find_files(mkv_folder, ".mkv")
    # if not mkv_flies:
    #     print("No valid `.mkv` files found!")
    # else:
    #     process_all_mkv(mkv_flies, subjectNum="subject05")

    # remember env2 以及subject26
    # skeletonpointfolder = "dataset/env1/subjects/subject05/origal"
    # # Get all `.npy` files to process
    # skeleton_files = find_files_fromsubfolder(skeletonpointfolder, "body_skeleton.npy")
    # if not skeleton_files:
    #     print("No valid `body_skeleton.npy` files found!")
    # else:
    #     print(f"Found {len(skeleton_files)} files, starting processing...")
    #     # Process each `.npy` file
    #     for folder_path, skeleton_file in skeleton_files:
    #         process_skeleton_file(folder_path, skeleton_file)
    
    # align data
    # radar_file_path = find_files('C:/PHD/Matlab/Radar/matlab_radar/data05/pc/env1', ".mat")
    # if not radar_file_path or not skeleton_files:
    #     print("No valid `.mat` or `action_segments.txt` files found!")
    # else:
    #     for val1, val2 in zip(skeleton_files, radar_file_path):
    #         aligner = DataAligner(val1[0], val2, val1[0])
    #         aligner.align_and_segment_data()
    #     print(f"Data alignment and segmentation complete for {len(skeleton_files)} files.")
    # print("data align done!")

    radar_files = get_aligned_mat_files("dataset/env1/subjects/subject03/origal")
    convert_mat_to_hdf5(radar_files)

    # 定义文件夹路径
    # aligned_folder = r'dataset\env1\subjects\subject02\aligned'

    # # 获取文件列表
    # files = os.listdir(aligned_folder)

    # # 按编号分组整理文件
    # for file in files:
    #     if 'segment' in file:
    #         # 提取动作编号，例如 segment01
    #         segment_id = file.split('_')[-1].split('.')[0]
    #         action_folder = os.path.join(aligned_folder, f'action{segment_id[-2:]}')

    #         # 创建动作文件夹（如果不存在）
    #         if not os.path.exists(action_folder):
    #             os.makedirs(action_folder)

    #         # 移动文件到对应动作文件夹
    #         src_path = os.path.join(aligned_folder, file)
    #         dest_path = os.path.join(action_folder, file)
    #         shutil.move(src_path, dest_path)

    # print("Files organized successfully!")




