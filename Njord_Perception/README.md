# Njord_Perception

Real-time buoy detection and localization for an Autonomous Surface Vehicle (ASV) developed in the **Njord Challenge** student team.

This module runs onboard the ASV and detects marine navigation buoys using a ZED stereo camera with a machine learning model (using YOLOv8s), then estimates depth and approximate bearing and publishes results through ROS2.

## What I built
- **Perception pipeline with ROS2**.
  - grabs ZED stereo frames
  - runs real-time inference
  - extracts depth at detection center
  - computes an approximate bearing from pixel offset
  - publishes ranked detections by depth (nearest first) over ROS2.

## File overview
- **Buoy_detection.py** An earlier attempt that didn't use machine learning. It succesfully was able to mask green, red, yellow and black colors from the buoys, but we had problems with the cardinal marks – as the order of the colors are the distinction.
- **Buoy_detectionML.py** Real-time inference of the livefeed from the camera. This was used for developing, so we could clearly see the bounding boxes from the inference, and get correct values.
- **GNSS_publisher and GNSS_subscriber** Used to learn ROS2
- **zed_camera_node.py** The final file that runs locally on Jetson Nano on startup, when the boat is driving autonomously. We had some issues with the Jetson Nano not being able to run inference on the camera-feed for high frame rate, but the boat was moving quite slowly anyways so we didn't need more than one frame per second. 