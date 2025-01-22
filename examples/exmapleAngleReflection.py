import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import time

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
   
# Start device
device = pykinect.start_device(config=device_config)

def get_rgb_and_depth_frames():
    # Get the color image from the capture
    capture = device.update()
    ret_color, color_image = capture.get_color_image()

    # Get the colored depth
    ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    if not ret_color or not ret_depth:
        return None, None

    return color_image, transformed_depth_image

def find_corner_reflector_center(rgb_image, depth_image):
  
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            area = cv2.contourArea(contour)
            if area < 1000:  # Ignore small triangles
                continue

            # 提取顶点坐标
            points = approx.reshape(-1, 2)
            

            # 按 y 坐标排序以找到最近的两个点
            points = sorted(points, key=lambda p: p[1])
            closest_points = points[1:]  # 最近的两个点

            # 计算中点坐标
            x1, y1 = closest_points[0]
            x2, y2 = closest_points[1]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 获取深度值
            depths = [depth_image[y, x] for x, y in points]
            center_depth = max(depths)  # 最大深度值

            # 在 RGB 图像上绘制中点
            cv2.circle(rgb_image, (center_x, center_y), 5, (0, 255, 0), -1)  # 中点

            # 返回中点坐标及深度值
            return (center_x, center_y, center_depth)

    return None

def main():
    # Record the start time
    start_time = time.time()
    while True:
        # 获取 RGB 和深度图像
        rgb_image, depth_image = get_rgb_and_depth_frames()
        if rgb_image is None or depth_image is None:
            continue

        # 检测角反射器中点
        result = find_corner_reflector_center(rgb_image, depth_image)
        if result is not None:
            center_x, center_y, depth = result
            print(f"Corner reflector center: X={center_x}, Y={center_y}, Depth={depth} mm")

        # 显示 RGB 图像
        cv2.imshow("RGB Image", rgb_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Check if 20 seconds have passed
        if time.time() - start_time > 20:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
