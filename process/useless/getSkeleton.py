import cv2
import numpy as np
import pykinect_azure as pykinect
import os

def initialize_kinect(video_filename):
    """Initialize Kinect playback and body tracker."""
    pykinect.initialize_libraries(track_body=True)
    device = pykinect.start_playback(video_filename)
    playback_calibration = device.get_calibration()
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)
    return device, body_tracker

def get_skeleton_data(body_frame, device, capture, depth_image, transformed_points_map, points_map):
    """Extract 3D skeleton data from Kinect frame."""
    skeletons = {"body": [], "color": [], "depth": [], "depth_converted": []}

    if(body_frame.get_num_bodies() == 0):
        return None
    body_id = 0
 
    color_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_COLOR).numpy()
    depth_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_DEPTH).numpy()
    skeleton_3d = body_frame.get_body(body_id).numpy()  # (32, 4)

    body_joints, color_joints, depth_joints, depth_converted_joints = [], [], [], []
    for joint in range(pykinect.K4ABT_JOINT_COUNT):
        color_joint_2d = color_skeleton_2d[joint, :]
        depth_joint_2d = depth_skeleton_2d[joint, :]

        dh, dw = depth_image.shape    # Convert depth 2D to 3D

        dx, dy = int(depth_joint_2d[0]), int(depth_joint_2d[1])
        # limit the x, y coordinates to the image size
        dx = np.clip(dx, 0, dw - 1)
        dy = np.clip(dy, 0, dh - 1)
        depth = depth_image[dy, dx]
        depth_joint_float2 = pykinect.k4a_float2_t(depth_joint_2d)
        depth_joint_float3 = device.calibration.convert_2d_to_3d(
          depth_joint_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH
        )
        depth_converted_joint_3d = [depth_joint_float3.xyz.x, depth_joint_float3.xyz.y, depth_joint_float3.xyz.z]
        
        # Extract 3D coordinates from different spaces
        h, w, _ = transformed_points_map.shape
        x, y = int(color_joint_2d[0]), int(color_joint_2d[1])

        # limit the x, y coordinates to the image size
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        color_joint_3d = transformed_points_map[y, x, :]
        depth_joint_3d = points_map[dy, dx, :]
        body_joint_3d = skeleton_3d[joint, :3]

        # Add the joint coordinates to the respective lists
        body_joints.append(body_joint_3d)
        color_joints.append(color_joint_3d)
        depth_joints.append(depth_joint_3d)
        depth_converted_joints.append(depth_converted_joint_3d)

    skeletons["body"] = body_joints
    skeletons["color"] = color_joints
    skeletons["depth"] = depth_joints
    skeletons["depth_converted"] = depth_converted_joints

    return skeletons

def interpolate_missing_joints(current_joints, latest_joints):
    """Fill missing joints using the latest available data for all four coordinate systems."""
    interpolated_joints = current_joints.copy()  # 复制当前帧数据

    for key in ['body', 'color', 'depth', 'depth_converted']:
        for i in range(32):  # 遍历 32 个骨骼点
            if np.all(current_joints[key][i] == 0):  
                # 若当前帧的该骨骼点数据全为 0，则用 latest_joints 填充
                interpolated_joints[key][i] = latest_joints[key][i].copy()
            else:  
                # 若当前帧的该骨骼点数据非 0，则更新 latest_joints
                latest_joints[key][i] = current_joints[key][i].copy()

    return interpolated_joints


def save_skeleton_npy(all_skeletons):
    """Save all frames' skeleton data to NPY files for each coordinate system."""
    # Convert to numpy array of shape (num_frames, 32, 3)
    body_data = np.array(all_skeletons["body"])
    color_data = np.array(all_skeletons["color"])
    depth_data = np.array(all_skeletons["depth"])
    depth_converted_data = np.array(all_skeletons["depth_converted"])

    # Save each coordinate system data to separate NPY files
    np.save(os.path.join(SAVE_DIR, "body_skeleton.npy"), body_data)
    np.save(os.path.join(SAVE_DIR, "color_skeleton.npy"), color_data)
    np.save(os.path.join(SAVE_DIR, "depth_skeleton.npy"), depth_data)
    np.save(os.path.join(SAVE_DIR, "depth_converted_skeleton.npy"), depth_converted_data)

if __name__ == "__main__":
    # set the save directory
    latest_joints = {
    'body': np.zeros((32, 3)),
    'color': np.zeros((32, 3)),
    'depth': np.zeros((32, 3)),
    'depth_converted': np.zeros((32, 3))
}

    SAVE_DIR = "skeleton_npy1"
    os.makedirs(SAVE_DIR, exist_ok=True)
    video_filename = "elif/9.mkv"
    device, body_tracker = initialize_kinect(video_filename)

    all_skeletons = {"body": [], "color": [], "depth": [], "depth_converted": []}

    while True:
        ret, capture = device.update()
        if not ret or capture is None:
            print("Invalid capture.")
            break

        body_frame = body_tracker.update(capture=capture)
        ret_color, color_image = capture.get_color_image()
        ret_depth, depth_image = capture.get_depth_image()

        if not ret_color or not ret_depth:
            continue

        ret_transformed_color, transformed_color_image = capture.get_transformed_color_image()
        ret_point, points = capture.get_pointcloud()
        ret_transformed_point, transformed_points = capture.get_transformed_pointcloud()

        if not ret_transformed_color or not ret_transformed_point or not ret_point:
            continue

        points_map = points.reshape((transformed_color_image.shape[0], transformed_color_image.shape[1], 3))
        transformed_points_map = transformed_points.reshape((color_image.shape[0], color_image.shape[1], 3))

        skeletons = get_skeleton_data(body_frame, device, capture, depth_image, transformed_points_map, points_map)

        skeletons = interpolate_missing_joints(skeletons, latest_joints)

        # Append skeletons for this frame
        for key in all_skeletons:
            all_skeletons[key].append(skeletons[key])


        if cv2.waitKey(1) == ord('q'):
            break

    # Save all frames into separate NPY files for each coordinate system
    save_skeleton_npy(all_skeletons)
    print("All frames saved as NPY files for each coordinate system.")
