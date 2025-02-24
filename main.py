# get skeleton from mkv file
# split skeleton into different motions find timestamps
# align radar data with skeleton data based on timestamps
# split kinect and radar data into segments
# transform each to radar coordinate


from process.class_files.data_aligner import DataAligner
from process.utils.ProcessAllMKV import process_all_mkv
from process.utils.SaveMotionSplit import process_skeleton_file
from process.utils.FindAndSave import find_files, find_files_fromsubfolder


if __name__ == "__main__":
    # mkv_folder = "C:/PHD/elif"
    # mkv_flies = find_files(mkv_folder, ".mkv")
    # if not mkv_flies:
    #     print("No valid `.mkv` files found!")
    # else:
    #     process_all_mkv(mkv_flies)

    # remember env2 以及subject26
    skeletonpointfolder = "dataset/env2/subjects/subject26/origal"
    # # Get all `.npy` files to process
    skeleton_files = find_files_fromsubfolder(skeletonpointfolder, "body_skeleton.npy")
    # if not skeleton_files:
    #     print("No valid `body_skeleton.npy` files found!")
    # else:
    #     print(f"Found {len(skeleton_files)} files, starting processing...")
    #     # Process each `.npy` file
    #     for folder_path, skeleton_file in skeleton_files:
    #         process_skeleton_file(folder_path, skeleton_file)
    
    # align data
    radar_file_path = find_files('C:/PHD/Matlab/Radar/matlab_radar/elif/env2', ".mat")
    if not radar_file_path or not skeleton_files:
        print("No valid `.mat` or `action_segments.txt` files found!")
    else:
        for val1, val2 in zip(skeleton_files, radar_file_path):
            aligner = DataAligner(val1[0], val2, val1[0])
            aligner.align_and_segment_data()
        print(f"Data alignment and segmentation complete for {len(skeleton_files)} files.")
    print("data align done!")





