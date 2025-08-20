###PANORAMA GENERATION

import numpy as np 
from load_data import load_dataset
import matplotlib.pyplot as plt

# Load data
camd, imud, vicd = load_dataset("9")
cam_images = camd['cam']        # shape: (240, 320, 3, K)
# print(np.shape(cam_images))
cam_timestamps = camd['ts'][0]  # shape: (K,)
print(np.shape(cam_timestamps))
vicon_rots = vicd['rots']       # shape: (3, 3, T)
print(np.shape(vicon_rots))
vicon_timestamps = vicd['ts'][0]
print(np.shape(vicon_timestamps))

# Match each image to its closest-in-the-past VICON rotation
vicon_indices = np.searchsorted(vicon_timestamps, cam_timestamps, side='right') - 1
vicon_indices = np.clip(vicon_indices, 0, len(vicon_timestamps) - 1)
print(np.shape(vicon_indices))

# Camera intrinsics
H, W = 240, 320
fov_phi = np.deg2rad(45)      # vertical FOV
fov_lambda = np.deg2rad(60)   # horizontal FOV


# Pixel indices
i = np.arange(H)  # vertical (row indices)
j = np.arange(W)  # horizontal (column indices)

# Create meshgrid
jj, ii = np.meshgrid(j, i)


lambda_centered = (jj - W / 2) / W * fov_lambda    # shape: (H, W)
phi_centered = (H / 2 - ii) / H * fov_phi          # shape: (H, W)

# These are the spherical coordinates (longitude and latitude) per pixel
# λ ∈ [−30°, +30°] ; ϕ ∈ [−22.5°, +22.5°] (in radians)
print(np.shape(lambda_centered))
print(np.shape(phi_centered))

# Assuming lambda_centered and phi_centered are (H, W) arrays in radians

# Compute Cartesian coordinates on the unit sphere
x = np.cos(phi_centered) * np.sin(lambda_centered)
y = np.sin(phi_centered)
z = np.cos(phi_centered) * np.cos(lambda_centered)

# Optionally stack them together
cart_cam = np.stack((x, y, z), axis=-1)  # shape: (H, W, 3)

# R_cam_to_imu = np.eye(3)
R_cam_to_imu = np.array([[0,  0, 1],
                         [-1, 0, 0],
                         [0, -1, 0]])


K = len(vicon_indices)  # Number of images

# Preallocate output
rotated_carts_world = np.empty((K, H, W, 3))  # shape: (K, 240, 320, 3)

for k in range(K):
    idx = vicon_indices[k]
    R_imu_to_world = vicon_rots[:, :, idx]     # shape (3, 3)
    R_world_cam = R_imu_to_world @ R_cam_to_imu

    # Rotate all unit vectors in camera frame to world frame
    rotated_carts_world[k] = cart_cam @ R_world_cam.T  # shape: (240, 320, 3)


print(np.shape(rotated_carts_world)) # rotated_carts_world.shape = (K, 240, 320, 3)

# Flatten all rotated unit vectors from all frames
all_rotated = rotated_carts_world.reshape(-1, 3)

# Compute λ and ϕ in radians
λ_all = np.arctan2(all_rotated[:, 0], all_rotated[:, 2])                      # longitude
ϕ_all = np.arcsin(np.clip(all_rotated[:, 1], -1.0, 1.0))                      # latitude

# Print the range of coverage
print("Longitude λ range (radians):", np.min(λ_all), np.max(λ_all))
print("Latitude ϕ range (radians):", np.min(ϕ_all), np.max(ϕ_all))
print("Longitude λ range (degrees):", np.rad2deg(np.min(λ_all)), np.rad2deg(np.max(λ_all)))
print("Latitude ϕ range (degrees):", np.rad2deg(np.min(ϕ_all)), np.rad2deg(np.max(ϕ_all)))

# Panorama parameters
panorama_height = 800
panorama_width = 1600
panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)  # RGB panorama

# Function to map world vector to panorama coordinates
def project_to_panorama(vec):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]

    # Spherical coordinates
    λ = np.arctan2(x, y)              # longitude ∈ [−π, π]
    ϕ = np.arcsin(np.clip(z, -1, 1))  # latitude ∈ [−π/2, π/2]

    # Panorama pixel coordinates
    pan_x = ((λ + np.pi) / (2 * np.pi)) * panorama_width
    pan_y = ((ϕ + (np.pi / 2)) / np.pi) * panorama_height

    # Convert to integers
    pan_x = np.clip(pan_x.astype(int), 0, panorama_width - 1)
    pan_y = np.clip(pan_y.astype(int), 0, panorama_height - 1)

    return pan_y, pan_x  # row, col indices


for k in range(K):
    image = cam_images[..., k]  # shape: (240, 320, 3)
    rotated_vecs = rotated_carts_world[k]  # shape: (240, 320, 3)

    pan_y, pan_x = project_to_panorama(rotated_vecs)

    # Write to panorama
    panorama[pan_y, pan_x] = image

plt.figure(figsize=(15, 6))
plt.imshow(panorama)
plt.title("Panoramic Stitching from Rotated World Vectors")
plt.axis('off')
plt.show()

