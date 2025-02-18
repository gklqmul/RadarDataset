import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import time

def capture_vertex_points_3d(device, transformed_depth_image, vertices):
    vertex_points_3d = []
    for vertex in vertices:
        x, y = vertex[0]
        rgb_depth = transformed_depth_image[y, x]
        pixels = k4a_float2_t((x, y))
        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
        vertex_points_3d.append((pos3d_color, pos3d_depth))
    return vertex_points_3d

def calculate_central_point(vertices_3d):
    # 提取 y 坐标和 x 坐标
    y_coords = [vertex[0].xyz.y for vertex in vertices_3d if vertex[0] is not None]
    x_coords = [vertex[0].xyz.x for vertex in vertices_3d if vertex[0] is not None]
    depths = [vertex[0].xyz.z for vertex in vertices_3d if vertex[0] is not None]

    if len(y_coords) < 2 or len(x_coords) < 2 or len(depths) < 1:
        return None, None, None

    # 找到 y 轴上最接近的两个点
    sorted_y_indices = np.argsort(y_coords)
    nearest_y_indices = sorted_y_indices[:2]

     # 计算中心点的 Y 坐标
    central_y = (y_coords[nearest_y_indices[0]] + y_coords[nearest_y_indices[1]]) / 2

    # 找到 x 轴上中间的那个点
    sorted_x_indices = np.argsort(x_coords)
    central_x = x_coords[sorted_x_indices[1]]  # 中间的那个点

    # 使用三个点中最深的点的深度作为中心点的深度
    central_depth = max(depths)

    return central_x, central_y, central_depth

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
#     pykinect.initialize_libraries()

#     # Modify camera configuration
#     device_config = pykinect.default_configuration
#     device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
#     device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
#     device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
#     # print(device_config)

#    # Start device
#     timestamp = int(time.time())  # 获取当前时间戳（秒）
#     video_filename = f"{timestamp}.mkv"  # 生成文件名

#     device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)
    video_filename = "1739292462.mkv"

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Start playback
    playback = pykinect.start_playback(video_filename)

    playback_config = playback.get_record_configuration()
    print(playback_config)

    cv2.namedWindow('color Image',cv2.WINDOW_NORMAL)
    

    # Record the start time
    start_time = time.time()

    while True:
        # Get capture
        ret, capture = playback.update()

        if not ret:
            break
        
        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        # Get current timestamp
        timestamp = time.time()

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2GRAY)

        # Apply Gaussian blur to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Use Canny edge detector to detect edges
        edges = cv2.Canny(blurred_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        closest_triangle = None
        min_depth = float('inf')

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has 3 vertices (triangle)
            if len(approx) == 3:
                # Calculate the area of the triangle
                area = cv2.contourArea(contour)
                if area < 1000:  # Ignore small triangles
                    continue
 
                
                # Capture the vertices 3D coordinates
                vertices_3d = capture_vertex_points_3d(playback, transformed_depth_image, approx)

                # Check if this triangle is the closest one
                depth = min([vertex[1].xyz.z for vertex in vertices_3d])  # Use the minimum depth of the vertices
                if depth < min_depth:
                    min_depth = depth
                    closest_triangle = (approx, vertices_3d)

        if closest_triangle:
            approx, vertices_3d = closest_triangle

            # Draw the bounding box and vertices on the color image
            cv2.drawContours(color_image, [approx], -1, (0, 255, 0), 2)
            for vertex in approx:
                x, y = vertex[0]
                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
            
            # Calculate the central point
            central_x, central_y, central_depth = calculate_central_point(vertices_3d)

            # Print the 3D coordinates of the vertices
            frame_data = [timestamp]
            for i, (pos3d_color, pos3d_depth) in enumerate(vertices_3d):
                print(f"Vertex {i} 3D (Color): ({pos3d_color.xyz.x}, {pos3d_color.xyz.y}, {pos3d_color.xyz.z}), Vertex {i} 3D (Depth): ({pos3d_depth.xyz.x}, {pos3d_depth.xyz.y}, {pos3d_depth.xyz.z})")
                frame_data.extend([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z, pos3d_depth.xyz.x, pos3d_depth.xyz.y, pos3d_depth.xyz.z])
            print(f"Central Point 3D: ({central_x}, {central_y}, {central_depth})")


        # Display the color image with bounding box and vertices
        cv2.imshow('Transformed Color Image', color_image)

        # Check if 20 seconds have passed
        # if time.time() - start_time > 20:
        #     break

        # # Press q key to stop
        # if cv2.waitKey(1) == ord('q'):
        #     break

    cv2.destroyAllWindows()