# ECE276A PROJECT 1 (AUTHOR - TANYA ARORA)

import numpy as np
from load_data import load_dataset
from transforms3d.euler import mat2euler, quat2euler
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad

camd, imud, vicd = load_dataset("1")
imu_acc_bias = [-511.0083521,  -501.06916958,  605.8823015]
imu_gyro_bias = [373.56143827, 375.39467607, 369.61733664]

# print(np.shape(imud))
# print(camd.keys())
# print(vicd.keys())
# print(len((camd['ts'])[0]))
# print(len((vicd['ts'])[0]))
# cam_images = camd['cam']       
# cam_timestamps = camd['ts']
# vicon_rots = vicd['rots']      
# vicon_timestamps = vicd['ts']

# Extract raw data
timestamps = imud[0]
print("Dim Timestamps", np.shape(timestamps))
acc_raw = imud[1:4]   # ax, ay, az
print("acc_raw shape", np.shape(acc_raw))
gyro_raw = imud[4:7]  # wx, wy, wz
print("gyro_raw shape", np.shape(gyro_raw))
# ==== Constants ====
Vref = 3300  # mV
sensitivity_a = 300  # mV/g   (Typical Value)
sensitivity_w = 3.33  # mV/(deg/s)  (4x amplified)
scale_factor_a = Vref / 1023 / sensitivity_a   # g units
scale_factor_w = Vref / 1023 / sensitivity_w * (np.pi / 180)  # rad/s

# Accelerometer in g
acc_calibrated = (acc_raw.T - imu_acc_bias) * scale_factor_a  + np.array([0, 0, 1]) # shape: (N, 3)
# print("acc_cali shape", np.shape(acc_calibrated))
# Gyroscope in rad/s
gyro_calibrated = (gyro_raw.T - imu_gyro_bias) * scale_factor_w  # shape: (N, 3)
# print("gyro_cali shape", np.shape(gyro_calibrated))
imu_calibrated = np.vstack((
    timestamps,                         # (1, N)
    acc_calibrated.T,                  # (3, N)
    gyro_calibrated.T                  # (3, N)
))  # shape: (7, N)

print(np.shape(imu_calibrated))

########Quaternion Calculation

def quat_exp(vec):
    """Computes quaternion exponential of [0, v]"""
    theta = np.linalg.norm(vec)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
    axis = vec / theta
    return np.concatenate([[np.cos(theta)], np.sin(theta) * axis])


def quat_mult(q1, q2):
    """Quaternion multiplication: q1 ◦ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def motion_model(timestamps, gyro_calibrated):
    # Identity quaternion as initial orientation
    q = np.array([1.0, 0.0, 0.0, 0.0])
    quaternion_trajectory = [np.concatenate([[timestamps[0]], q])]

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        omega = gyro_calibrated[i]
        delta_q = quat_exp(0.5 * dt * omega)
        q = quat_mult(q, delta_q)
        q = q / np.linalg.norm(q)
        quaternion_trajectory.append(np.concatenate([[timestamps[i]], q]))

    return np.array(quaternion_trajectory)

quaternion_with_timestamps = motion_model(timestamps, gyro_calibrated)    

# print("quaternion_with_timestamps_shape", np.shape(quaternion_with_timestamps))

def convert_quaternion_to_rpy(quaternion_with_timestamps):
    """
        np.ndarray: Array of shape (N, 3), each row is [roll, pitch, yaw]
    """
    return np.array([
        quat2euler(q, axes='sxyz') for q in quaternion_with_timestamps[:, 1:]
    ])

def convert_vicon_rotation_to_rpy(vicon_rots):
    """
      np.ndarray: Array of shape (N, 3), each row is [roll, pitch, yaw]
    """
    return np.array([
        mat2euler(vicon_rots[:, :, i], axes='sxyz') for i in range(vicon_rots.shape[2])
    ])

# --- IMU RPY with Timestamps ---
imu_rpy = convert_quaternion_to_rpy(quaternion_with_timestamps)
timestamps_column = quaternion_with_timestamps[:, 0].reshape(-1, 1)
imu_rpy_with_timestamps = np.concatenate([timestamps_column, imu_rpy], axis=1)
# print(np.shape(imu_rpy_with_timestamps))  # (N, 4) 

# --- VICON RPY with Timestamps ---
vicon_rots = vicd['rots']
vicon_times = vicd['ts'].reshape(-1, 1)
vicon_rpy = convert_vicon_rotation_to_rpy(vicon_rots)
vicon_rpy_with_timestamps = np.concatenate([vicon_times, vicon_rpy], axis=1)
# print(np.shape(vicon_rpy_with_timestamps))  # (N, 4)

# # Plot Roll
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 1], label='IMU Roll', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 1], label='VICON Roll', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Roll (rad)")
# plt.title("Roll: IMU vs VICON")
# plt.legend()
# plt.grid()

# # Plot Pitch
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 2], label='IMU Pitch', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 2], label='VICON Pitch', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Pitch (rad)")
# plt.title("Pitch: IMU vs VICON")
# plt.legend()
# plt.grid()

# # Plot Yaw
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 3], label='IMU Yaw', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 3], label='VICON Yaw', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Yaw (rad)")
# plt.title("Yaw: IMU vs VICON")
# plt.legend()
# plt.grid()

# plt.show()


#######Acceleration - Observation Model 

def quat_conj(q):
    """Returns the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_rotate(q, v):
    """
    Rotate a vector v (3D) using quaternion q.
    Equivalent to: q^-1 ◦ [0,v] ◦ q
    """
    v_quat = np.concatenate([[0.0], v])       # [0, vx, vy, vz]
    q_conj = quat_conj(q)
    return quat_mult(quat_mult(q_conj, v_quat), q)[1:]  # Only return vector part

def observation_model(quaternion_with_timestamps, g_vec_world_g=np.array([0, 0, 1.0])):
    """
    Parameters:
        quaternion_with_timestamps (np.ndarray): Array of shape (N, 5), where each row is [timestamp, qw, qx, qy, qz]
        g_vec_world_g (np.ndarray): Gravity vector in world frame, default is [0, 0, 1.0] (in g units)

    Returns:
        np.ndarray: Predicted accelerations in IMU frame, shape (N, 3)
    """
    return np.array([
        quat_rotate(q, g_vec_world_g) for q in quaternion_with_timestamps[:, 1:]
    ])

predicted_acc_g = observation_model(quaternion_with_timestamps)
# print(np.shape(predicted_acc_g))


# # Compare accelerations from IMU vs quaternion-derived gravity vector

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.plot(timestamps, acc_calibrated[:, 0], label='IMU ax', linestyle='-')
# plt.plot(timestamps, predicted_acc_g[:, 0], label='Predicted ax', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Acceleration X (g)")
# plt.title("Acceleration X: IMU vs Predicted")
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.plot(timestamps, acc_calibrated[:, 1], label='IMU ay', linestyle='-')
# plt.plot(timestamps, predicted_acc_g[:, 1], label='Predicted ay', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Acceleration Y (g)")
# plt.title("Acceleration Y: IMU vs Predicted")
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 3)
# plt.plot(timestamps, acc_calibrated[:, 2], label='IMU az', linestyle='-')
# plt.plot(timestamps, predicted_acc_g[:, 2], label='Predicted az', linestyle='--')
# plt.xlabel("Timestamp")
# plt.ylabel("Acceleration Z (g)")
# plt.title("Acceleration Z: IMU vs Predicted")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


def quat_log(q):
    """Logarithm map of unit quaternion to axis-angle vector (R^3)."""
    v = q[1:]
    norm_v = np.linalg.norm(v)
    norm_q = np.linalg.norm(q)
    if norm_v < 1e-8:
        return np.zeros(3)
    theta = 2 * np.arccos(np.clip(q[0], -1.0, 1.0))
    return theta * (v / norm_v)

