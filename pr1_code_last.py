
import numpy as np
from transforms3d.euler import mat2euler, quat2euler
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad
from pr1_code import acc_calibrated, gyro_calibrated, timestamps, quaternion_with_timestamps, vicon_rpy_with_timestamps, imu_rpy_with_timestamps
from jax import vmap
jax.config.update("jax_enable_x64", True)

timestamps = jnp.array(timestamps).reshape(-1, 1)#[:6,:]       # shape: (N,1)
acc_calibrated = jnp.array(acc_calibrated)#[:6,:]   # shape: (N, 3)
gyro_calibrated = jnp.array(gyro_calibrated)#[:6,:]  # shape: (N, 3)
# print("timestamps shape", jnp.shape(timestamps))

@jax.jit
def quat_mult(q1, q2):
    """Batch quaternion multiplication: q1 ◦ q2 for (N,4) x (N,4)"""
    # print("Starting to do quaternion multiplication")
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ans = jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=1)
    # print("Shape of multi quat", jnp.shape(ans))
    return ans

@jax.jit
def quat_conj(q):
    """Quaternion conjugate for (N,4)"""
    # print("Starting to do quaternion conj")
    ans=q * jnp.array([1.0, -1.0, -1.0, -1.0])
    # print("Shape of quat conj", jnp.shape(ans))
    return ans

@jax.jit
def quat_exp(v):
    """Quaternion exponential of (N,3) => (N,4)"""
    # print("Starting to do quaternion exponential")
    theta = jnp.linalg.norm(v, axis=1, keepdims=True)
    axis = v / (theta + 1e-8)
    w = jnp.cos(theta)
    xyz = jnp.sin(theta) * axis
    ans = jnp.concatenate([w, xyz], axis=1)
    # print("Shape of quat exp", jnp.shape(ans))
    return ans

@jax.jit
def quat_log(q):
    # print("Starting to do quaternion log")
    v = q[:, 1:]
    norm_v = jnp.linalg.norm(v, axis=1)
    eps = 1e-8
    clipped_w = jnp.clip(q[:, 0], -1.0 + eps, 1.0 - eps)
    theta = 2 * jnp.arccos(clipped_w)
    axis = v / (norm_v[:, None] + eps)
    ans = theta[:, None] * axis  # (N,3)
    # print("Shape of quat log", jnp.shape(ans))
    return ans

@jax.jit
def quat_rotate(qs, v):
    # print("Starting to do quaternion rotate")
    """Rotate a constant vector v by batch of quaternions qs"""
    N = qs.shape[0]
    v_quat = jnp.tile(jnp.concatenate([jnp.array([0.0]), v]), (N, 1))  # (N,4)
    q_conj = quat_conj(qs)
    ans = quat_mult(quat_mult(q_conj, v_quat), qs)[:, 1:]  # (N,3)
    # print("Shape of quat rotate", jnp.shape(ans))
    return ans

@jax.jit
def motion_model_predict(q_t, omega_t, dt):
    # print("Starting to do motion model prediction")
    delta_q = quat_exp(0.5 * dt * omega_t)
    # print("Dim of delta_q")
    # print(jnp.shape(delta_q))
    # print("Dim of q_t")
    # print(jnp.shape(q_t))
    q_pred = quat_mult(q_t, delta_q)
    ans = q_pred / jnp.linalg.norm(q_pred, axis=1, keepdims=True)
    # print("Shape of motion model prediction", jnp.shape(ans))
    return ans

@jax.jit
def observation_model(q_t):
    # print("Starting to do observation model")
    g_world = jnp.array([0.0, 0.0, 1.0])
    ans = quat_rotate(q_t, g_world)
    # print("Shape of observation model", jnp.shape(ans))
    return ans

@jax.jit
def compute_cost(qs, acc_meas, gyro_meas, timestamps):   
    # print("Starting to compute cost")
    q_pred = motion_model_predict(qs[:-1], gyro_meas[:-1], timestamps[1:] - timestamps[:-1])
    # print("Shape of q_pred", jnp.shape(q_pred))
    relative_rot = quat_mult(quat_conj(qs[1:]), q_pred)
    # print("Shape of relative_rot", jnp.shape(relative_rot))
    motion_error = 2.0 * quat_log(relative_rot)
    # print("Shape of motion_error", jnp.shape(motion_error))
    motion_term = jnp.sum(jnp.square(motion_error), axis=1)
    motion_term = jnp.concatenate([jnp.zeros((1,)), motion_term])
    # print("Shape of obmotion_term", jnp.shape(motion_term))

    predicted_acc = observation_model(qs)
    # print("Shape of predicted acc", jnp.shape(predicted_acc))
    acc_term = jnp.sum(jnp.square(acc_meas - predicted_acc), axis=1)
    # print("Shape of acc_term", jnp.shape(acc_term))
    ans = jnp.sum(0.5 * (motion_term + acc_term))
    # print("Shape of comute cose", jnp.shape(ans))
    return ans

@jax.jit
def project_to_unit_quat(qs):
    # print("Starting to do project_to_unit_quat")
    ans = qs / jnp.linalg.norm(qs, axis=1, keepdims=True)
    # print("Shape of project_to_unit_quat", jnp.shape(ans))
    return ans

def optimize_quaternion_trajectory(qs_init, acc_meas, gyro_meas, timestamps, lr=1e-2, num_iters=200):
    print("Starting to do trajectory optimization")
    qs = qs_init.copy()

    cost_fn = lambda qs_: compute_cost(qs_, acc_meas, gyro_meas, timestamps)
    grad_fn = jax.grad(cost_fn)

    for i in range(num_iters):
        # print("Starting to do interation number", i)
        g = grad_fn(qs)
        # print("Shape of cost g", jnp.shape(g))
        qs -= lr * g
        qs = project_to_unit_quat(qs)
        # print("Shape of qs for 1 iteration", jnp.shape(qs))

    print("stopping optimization")
    return qs

qs_init = np.array(quaternion_with_timestamps)[:, 1:]  # Strip timestamps, keep only quats
qs_init = jnp.array(qs_init)  # Convert to JAX array

optimized_qs = optimize_quaternion_trajectory(
    qs_init,
    acc_calibrated,
    gyro_calibrated,
    timestamps,
    lr=1e-3,
    num_iters=2000
)

print(jnp.shape(optimized_qs))

@jax.jit
def quat_to_rpy(q):
    """
    Convert a single unit quaternion [w, x, y, z] to roll-pitch-yaw (in radians).
    Compatible with jax.jit and jax.grad.
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = jnp.where(
        jnp.abs(sinp) >= 1.0,
        jnp.sign(sinp) * jnp.pi / 2.0,
        jnp.arcsin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw])

@jax.jit
def convert_quaternion_to_rpy(q_array: jnp.ndarray) -> jnp.ndarray:
    """
    Convert (N, 4) quaternions to (N, 3) roll-pitch-yaw.
    """
    return vmap(quat_to_rpy)(q_array)

rpy_angles = convert_quaternion_to_rpy(optimized_qs)        # shape (N, 3)
rpy_angles = jnp.nan_to_num(rpy_angles)

print("Shape of rpy", jnp.shape(rpy_angles))
print("Any NaNs in rpy_deg?", jnp.isnan(rpy_angles).any())
print("Any NaNs in timestamps?", jnp.isnan(timestamps).any())
# Plot Roll
plt.figure()
plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 1], label='IMU Roll', linestyle='-')
plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 1], label='VICON Roll', linestyle='--')
plt.plot( timestamps, rpy_angles[:,0], label='Optimized Roll', linestyle='-.')
plt.xlabel("Timestamp")
plt.ylabel("Roll (rad)")
plt.title("Roll: IMU vs VICON vs Optimized")
plt.legend()
plt.grid()

# Plot Pitch
plt.figure()
plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 2], label='IMU Pitch', linestyle='-')
plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 2], label='VICON Pitch', linestyle='--')
plt.plot( timestamps, rpy_angles[:,1], label='Optimized Roll', linestyle='-.')
plt.xlabel("Timestamp")
plt.ylabel("Pitch (rad)")
plt.title("Pitch: IMU vs VICON vs Optimized")
plt.legend()
plt.grid()

# Plot Yaw
plt.figure()
plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 3], label='IMU Yaw', linestyle='-')
plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 3], label='VICON Yaw', linestyle='--')
plt.plot( timestamps, rpy_angles[:,2], label='Optimized Roll', linestyle='-.')
plt.xlabel("Timestamp")
plt.ylabel("Yaw (rad)")
plt.title("Yaw: IMU vs VICON vs Optimized")
plt.legend()
plt.grid()

plt.show()

# print("shape opti_qs", np.shape(optimized_qs))

# def convert_quaternion_to_rpy(q_array):
#     r = R.from_quat(q_array[:, [1, 2, 3, 0]])  # transforms3d: [w, x, y, z], scipy: [x, y, z, w]
#     return r.as_euler('xyz', degrees=False)

# # optimized_qs_with_timestamps = np.concatenate([timestamps, optimized_qs], axis=1)
# optimized_qs_rpy = convert_quaternion_to_rpy(optimized_qs)
# print(np.shape(optimized_qs_rpy))
# # timestamps_column = optimized_qs_with_timestamps[:, 0]
# # optimized_qs_rpy_with_timestamps = np.concatenate([timestamps_column, optimized_qs_rpy], axis=1)

# # Plot Roll
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 1], label='IMU Roll', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 1], label='VICON Roll', linestyle='--')
# plt.plot(optimized_qs_rpy_with_timestamps[:, 0], optimized_qs_rpy_with_timestamps[:, 1], label='Optimized Roll', linestyle='---')
# plt.xlabel("Timestamp")
# plt.ylabel("Roll (rad)")
# plt.title("Roll: IMU vs VICON")
# plt.legend()
# plt.grid()

# # Plot Pitch
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 2], label='IMU Pitch', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 2], label='VICON Pitch', linestyle='--')
# plt.plot(optimized_qs_rpy_with_timestamps[:, 0], optimized_qs_rpy_with_timestamps[:, 2], label='Optimized Pitch', linestyle='---')
# plt.xlabel("Timestamp")
# plt.ylabel("Pitch (rad)")
# plt.title("Pitch: IMU vs VICON")
# plt.legend()
# plt.grid()

# # Plot Yaw
# plt.figure()
# plt.plot(imu_rpy_with_timestamps[:, 0], imu_rpy_with_timestamps[:, 3], label='IMU Yaw', linestyle='-')
# plt.plot(vicon_rpy_with_timestamps[:, 0], vicon_rpy_with_timestamps[:, 3], label='VICON Yaw', linestyle='--')
# plt.plot(optimized_qs_rpy_with_timestamps[:, 0], optimized_qs_rpy_with_timestamps[:, 3], label='Optimized Yaw', linestyle='---')
# plt.xlabel("Timestamp")
# plt.ylabel("Yaw (rad)")
# plt.title("Yaw: IMU vs VICON")
# plt.legend()
# plt.grid()

# plt.show()

