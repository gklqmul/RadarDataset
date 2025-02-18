import numpy as np
import pandas as pd

def compute_rigid_transform(P, Q):
    """
    Compute the rigid transformation (rotation matrix R and translation vector t) such that Q ≈ R * P + t
    :param P: (N, 3) numpy array, representing the source point set
    :param Q: (N, 3) numpy array, representing the target point set
    :return: Rotation matrix R (3x3) and translation vector t (3,)
    """
    # Compute centroids
    p_centroid = np.mean(P, axis=0)
    q_centroid = np.mean(Q, axis=0)

    # Center the points
    P_centered = P - p_centroid
    Q_centered = Q - q_centroid

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix R
    R = Vt.T @ U.T

    # Ensure R is orthogonal, if det(R) < 0, correct it
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector t
    t = q_centroid - R @ p_centroid

    return R, t

def transform_points(P, R, t):
    """
    Transform the 3D point set P using rotation matrix R and translation vector t
    :param P: (N, 3) numpy array, representing the source point set
    :param R: (3, 3) rotation matrix
    :param t: (3,) translation vector
    :return: Transformed point set Q
    """
    return P @ R.T + t  # Perform matrix operation to transform points

# Read the Excel file
file_path = "central point coordinate.xlsx"  # Path to your Excel file
df = pd.read_excel(file_path)

# Take the absolute values
df = df.abs()

# Extract 3D coordinates (assuming the Excel file has 6 columns)
P = df.iloc[:, 0:3].to_numpy()  # First set of points (9,3)
Q = df.iloc[:, 3:6].to_numpy()  # Second set of points (9,3)
print("P:", P)

# Compute the rotation matrix R and translation vector t
R, t = compute_rigid_transform(P, Q)

print("Computed rotation matrix R:")
print(R)
print("\nComputed translation vector t:")
print(t)

# Transform P using the computed R and t to match Q
P_transformed = transform_points(P, R, t)

# Compute the error (Euclidean distance)
errors = np.linalg.norm(P_transformed - Q, axis=1)

print("\nTransformation errors (Euclidean distance) for each point:")
print(errors)
print("\nAverage error:", np.mean(errors))
