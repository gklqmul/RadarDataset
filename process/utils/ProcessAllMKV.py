import os
from process.class_files import kinect_skeleton_processor

def process_all_mkv(mkv_files, subjectNum="subject26"):
    
    # Process each .mkv file one by one
    for index, mkv_file in enumerate(mkv_files, start=1):
        if index > 6:
            save_dir = os.path.join("dataset", "env2", "subjects", subjectNum, "origal", str(index-6))
        # Create save directory (subject26/origal/index)
        else:
            save_dir = os.path.join("dataset", "env1", "subjects", "subject26", "origal", str(index))
        
        os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist

        print(f"Processing: {mkv_file} Save directory: {save_dir}")
        
        # Create processor and pass save_dir
        processor = kinect_skeleton_processor(mkv_file, save_dir)  # Pass save_dir to the class
        processor.process_video()

        print(f"Processing complete: {mkv_file}\n")


