import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Modify camera configuration
	# device_config = pykinect.default_configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED #change the depth mode to 2x2 binned
	#print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	# Start body tracker
	#tracker_config = pykinect.default_tracker_configuration
	tracker_config = pykinect.k4abt_tracker_configuration_t()
	tracker_config.sensor_orientation = pykinect.K4ABT_SENSOR_ORIENTATION_DEFAULT
	tracker_config.tracker_processing_mode = pykinect.K4ABT_TRACKER_PROCESSING_MODE_GPU
	tracker_config.gpu_device_id = 0
	bodyTracker = pykinect.start_body_tracker(tracker_config)

	cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)
	while True:

		# Get capture
		capture = device.update()

		# Get body tracker frame
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
	
		# get body
		bodies = body_frame.get_bodies()

		# walk through each body
		for body in bodies:
			joints = body.joints
			for joint in joints:
				# get joint position
				x, y, z = joint.position.x, joint.position.y, joint.position.z
				print(f"Joint: {joint.id}, Position: ({x}, {y}, {z})")

		# draw body index
		combined_image = body_frame.draw_bodies(combined_image)
		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break