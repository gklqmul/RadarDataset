import h5py
import scipy.io as sio
import numpy as np

# Load the .mat file
mat_data = sio.loadmat("C:/PHD/Matlab/Radar/matlab_radar/elif/filtered_9_segment01.mat")  # Replace with your actual .mat file path
pc = mat_data["radar_data"]  # 888 frames, each with 12 attributes
timestamps = pc["timestamp"].flatten()  # Extract timestamps

# Retrieve attribute names and exclude the first two
all_attribute_names = np.array(pc[0, 0].dtype.names, dtype="S")  # Convert to byte strings
frame_attributes = all_attribute_names[:2]  # First two attributes (frame-level)
point_attributes = all_attribute_names[2:]  # Remaining attributes (point-level)

# Create the .h5 file
with h5py.File("output901.h5", "w") as h5f:
    frames_group = h5f.create_group("frames")  # Create a group for frames

    for i in range(pc.shape[1]):  # Iterate through 98 frames
        frame_data = pc[0, i]  # Extract frame i (struct)

        # Extract frame-level attributes (first two attributes)
        frame_metadata = {attr.decode(): frame_data[attr.decode()].item() for attr in frame_attributes}

        # Determine the number of points in this frame
        num_points = len(frame_data[point_attributes[0].decode()].flatten())  # Use any point attribute to check

        # Create an array of shape (num_points, remaining attributes)
        frame_array = np.zeros((num_points, len(point_attributes)))
        for j, attr in enumerate(point_attributes):  # Iterate over point attributes
            frame_array[:, j] = frame_data[attr.decode()].flatten()  # Store in columns
        
        # Store the frame data
        frame_ds = frames_group.create_dataset(f"frame_{i:03d}", data=frame_array)
        
        # Add metadata attributes
        timestamp_str = str(timestamps[i])
        frame_ds.attrs["timestamp"] = timestamp_str.encode()  # Store timestamp
        frame_ds.attrs["attribute_names"] = point_attributes  # Store only point-related attribute names
        for key, value in frame_metadata.items():
            frame_ds.attrs[key] = value  # Store frame-level attributes
