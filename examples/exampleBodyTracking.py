import cv2
import numpy as np
import pykinect_azure as pykinect
import time

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

	# Start body tracker
    tracker_config = pykinect.k4abt_tracker_configuration_t()
    tracker_config.sensor_orientation = pykinect.K4ABT_SENSOR_ORIENTATION_DEFAULT
    tracker_config.tracker_processing_mode = pykinect.K4ABT_TRACKER_PROCESSING_MODE_GPU
    tracker_config.gpu_device_id = 0
    bodyTracker = pykinect.start_body_tracker(tracker_config)

    cv2.namedWindow('Depth image with skeleton', cv2.WINDOW_NORMAL)
    all_joint_data = []

    # Record the start time
    start_time = time.time()

    while True:
        # Get capture
        capture = device.update()

        # Get body frame
        body_frame = bodyTracker.update()

        # Get the color depth image from the capture
        ret_depth, depth_color_image = capture.get_colored_depth_image()

		# Get the colored body segmentation
        ret_color, body_image_color = body_frame.get_segmentation_image()
        if not ret_depth or not ret_color:
            continue
			
		# Combine both images
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

		# Draw the skeletons
        combined_image = body_frame.draw_bodies(combined_image)

        # Get bodies
        bodies = body_frame.get_bodies()

        # Walk through each body
        for body in bodies:
            joints = body.joints
            for joint in joints:
                # Get joint position
                x, y, z = joint.position.x, joint.position.y, joint.position.z
                joint_id = joint.id
                print(f"Joint: {joint_id}, Position: ({x}, {y}, {z})")

                # Save joint data
                joint_data = [joint_id, x, y, z]
                all_joint_data.append(joint_data)

        # Draw body index
        combined_image = body_frame.draw_bodies(combined_image)
        # Overlay body segmentation on depth image
        cv2.imshow('Depth image with skeleton', combined_image)

        # Check if 20 seconds have passed
        if time.time() - start_time > 20:
            break

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

    # Save the joint data to a NumPy file
    np.save('joint_positions.npy', np.array(all_joint_data))

    cv2.destroyAllWindows()