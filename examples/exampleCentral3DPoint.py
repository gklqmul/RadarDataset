import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t

def capture_center_point_3d(device, color_image, transformed_depth_image, center_x, center_y):
    # get centre pixel depth
    rgb_depth = transformed_depth_image[center_y, center_x]

    # create a k4a_float2_t object with the center pixel coordinates
    pixels = k4a_float2_t((center_x, center_y))

    # 2D to 3D conversion
    pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)

    return pos3d_color, pos3d_depth

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Transformed Color Image', cv2.WINDOW_NORMAL)
    while True:
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        # Convert color image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

        # Define the range for white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which is likely to be the whiteboard
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw the bounding box and center point on the color image
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

            # Capture the center point 3D coordinates
            pos3d_color, pos3d_depth = capture_center_point_3d(device, color_image, transformed_depth_image, center_x, center_y)
            print(f"Center Point 3D (Color): {pos3d_color}, Center Point 3D (Depth): {pos3d_depth}")

        # Display the color image with bounding box and center point
        cv2.imshow('Transformed Color Image', color_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()