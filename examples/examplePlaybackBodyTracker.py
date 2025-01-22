import cv2
import time
import pykinect_azure as pykinect

if __name__ == "__main__":

	video_filename = "output.mkv"

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Start playback
	playback = pykinect.start_playback(video_filename)

	playback_config = playback.get_record_configuration()
	# print(playback_config)

	playback_calibration = playback.get_calibration()

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

	cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)

	all_joint_data = []

    # Record the start time
	start_time = time.time()
	print('start time kinect', start_time)

	while True:

		# Get camera capture
		ret, capture = playback.update()

		if not ret or capture is None:
			print("Invalid capture.")
			break

		# Get color image
		ret_color, color_image = capture.get_transformed_color_image()

		# Get the colored depth
		ret_depth, depth_color_image = capture.get_colored_depth_image()


		
		if not ret_color or not ret_depth:
			continue

		body_frame = bodyTracker.update(capture=capture)	
		# Get the colored body segmentation
		ret_seg, body_image_color = body_frame.get_segmentation_image()
		
		# Combine both images
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
		combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

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
				# Draw the skeletons
		combined_image = body_frame.draw_bodies(combined_image)
		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)
		        # Check if 20 seconds have passed

		if time.time() - start_time > 20:
			break

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break