import numpy as np

def calculate_central_point(vertices_3d):
    # extract x, y, and depth values from vertices
    y_coords = [vertex[1] for vertex in vertices_3d]
    depths = [vertex[5] for vertex in vertices_3d]
    x_coords = [vertex[0] for vertex in vertices_3d]

    # find the two nearest points based on y coordinate
    sorted_indices = np.argsort(y_coords)
    nearest_indices = sorted_indices[:2]

    # calculate the central x coordinate as the average of the two nearest points
    central_x = (x_coords[nearest_indices[0]] + x_coords[nearest_indices[1]]) / 2

    # use the y coordinate of the nearest point as the central y coordinate
    central_y = y_coords[nearest_indices[0]]

    # use the maximum depth of the two points as the central depth
    central_depth = max(depths)

    return central_x, central_y, central_depth


data = np.load('triangle_vertices_data.npy')

sum_x = 0
sum_y = 0
sum_z = 0
count = 0 

for frame_data in data:
    timestamp = frame_data[0]
    vertices_3d = frame_data[1:].reshape(-1, 6)  # (x, y, z, x_depth, y_depth, z_depth)

    # calculate central point
    central_x, central_y, central_depth = calculate_central_point(vertices_3d)
    sum_x += central_x
    sum_y += central_y
    sum_z += central_depth
    count += 1
    print(f"Timestamp: {timestamp}")
    for i, vertex in enumerate(vertices_3d):
        print(f"Vertex {i} 3D (Color): (X: {vertex[0]}, Y: {vertex[1]}, Z: {vertex[2]}), Vertex {i} 3D (Depth): (X: {vertex[3]}, Y: {vertex[4]}, Z: {vertex[5]})")
    print(f"Central Point 3D: (X: {central_x}, Y: {central_y}, Depth: {central_depth})")
    print()

avg_x = sum_x / count
avg_y = sum_y / count
avg_z = sum_z / count
print(f"Average Central Point 3D: (X: {avg_x}, Y: {avg_y}, Depth: {avg_z})")