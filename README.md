# 🧭 Orientation Tracking & Panorama Generation using IMU, Vicon and RGB Camera Data

> 📚 **Key Concepts Learned**
>
> - Sensor data processing and IMU calibration
> - Time synchronization of multimodal sensor data (IMU, VICON, Camera)
> - Quaternion Kinematics
> - transforms3d library 
> - Projected Gradient Descent optimization
> - JAX library for vectorized operations, JIT compilation, and gradient computation
> - RGB Camera-Based Panoramic Image Projection and Stitching
> - Data visualization using Matplotlib

This repository contains a robotics perception project developed as part of **ECE 276A: Sensing & Estimation in Robotics** at **UC San Diego**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9bbd150d-7350-421d-a4be-4925e82fa27f" alt="IMU and Camera Sensors" width="400"/>
</p>

The goal is to:
1. Estimate the 3D orientation of a rotating body using raw IMU data and optimization.
2. Use the estimated orientation (or ground-truth VICON data) to stitch RGB camera images into a panorama.

---

## 📌 Project Description

### 🔹 Orientation Tracking
Using raw IMU data (accelerometer and gyroscope), a **Projected Gradient Descent (PGD)** algorithm is implemented to estimate the body’s orientation over time in quaternion form. The estimate is compared with ground-truth VICON orientation data to evaluate accuracy.

### 🔹 Panorama Generation
Once orientation is available, each RGB image is projected onto a cylindrical surface based on its corresponding pose. These projections are then stitched together to create a smooth panoramic image of the environment captured during motion.

---

## 🧰 Requirements

You can install the following:

```bash
numpy
jax
matplotlib
transforms3d
```
---

## 🚀 Steps and Results

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/TanyaArora23/Orientation-Tracking-and-Panorama-Generation.git
cd Orientation-Tracking-and-Panorama-Generation
```
and set up the Python Environment

### 2️⃣ Quaternion-Based Orientation Tracking
By running following code it processes IMU data to estimate the orientation of a rotating body over time using quaternion integration. It first calibrates the raw accelerometer and gyroscope readings using known biases and scale factors. Then, it integrates the angular velocity measurements to compute the quaternion-based orientation trajectory. These quaternions are converted to roll-pitch-yaw (RPY) angles and compared with the ground-truth VICON data for validation. Additionally, it implements an observation model that rotates the gravity vector into the IMU frame using the estimated orientation, allowing a comparison between predicted and actual accelerations to further verify the correctness of the orientation estimate.
```bash
python pr1_code.py
```
The following Results are Observed:
<p align="center">
  <img src="https://github.com/user-attachments/assets/759dba25-5f0d-4d81-9895-37d47e75adde" alt="Pitch_without_optimization" width="300"/><img src="https://github.com/user-attachments/assets/5c9100c5-f339-459d-b4b3-8fdc0e366504" alt="Roll_without_optimization" width="300"/><img src="https://github.com/user-attachments/assets/40b1997c-750c-42b7-8f26-3522d6d17897" alt="Yaw_without_optimization" width="300"/>
</p>

### 3️⃣ Projected Gradient Descent for Optimization
By running following code implements projected gradient descent to optimize the orientation trajectory of a rotating body using IMU measurements. Starting from an initial quaternion sequence (obtained via integration of gyro data), it defines a cost function composed of two terms: (1) the motion model error, which penalizes deviation from the expected quaternion evolution using gyro data, and (2) the observation model error, which penalizes misalignment between the gravity vector predicted by orientation and the actual measured acceleration. The gradient of this cost function is computed using JAX's automatic differentiation, and the quaternions are updated iteratively while being re-normalized (projected) to unit norm after each step. The final optimized quaternion trajectory is converted to roll-pitch-yaw angles for comparison against raw IMU and VICON ground truth. This process refines the estimated orientation trajectory by jointly considering both motion dynamics and acceleration consistency.
```bash
python pr1_code_last.py
```
The following Results are Observed:
<p align="center">
  <img src="https://github.com/user-attachments/assets/1f17b947-5d3e-4871-b2aa-a45b0a099fd4" alt="Roll_with_optimization" width="500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/463663b1-4460-40f1-9e57-9becc6302bc4" alt="Pitch_with_optimization" width="500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/08a2c8d2-5d7f-4fcd-b621-043edf69cb62" alt="Yaw_with_optimization" width="500"/>
</p>

### 4️⃣ Panorama Generation 
By running the following code constructs a panoramic image by stitching together RGB frames captured by a rotating camera, using the VICON ground-truth orientation. First, it loads the camera images, their timestamps, and the corresponding rotation matrices from VICON. For each image, it finds the closest VICON pose in the past and converts each camera pixel into a 3D unit vector in the camera frame based on its known field of view. These vectors are then rotated into the world frame using the VICON rotation and a fixed camera-to-IMU transform. After transforming all vectors, they are converted to spherical coordinates (longitude and latitude), which are mapped to 2D coordinates in a predefined cylindrical panorama canvas. Each pixel from the original camera image is projected into the panorama at the corresponding location based on its orientation in the world. This results in a wide panoramic image that captures the full spatial coverage of the rotating camera.
```bash
python panorama.py
```
The following Results are Observed:
<p align="center">
  <img src="https://github.com/user-attachments/assets/bb35cad9-fd0f-4db3-a5d5-ce4723f9ba60" alt="Panorama" width="800"/>
</p>
