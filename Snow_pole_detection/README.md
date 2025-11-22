# Snow Pole Detection – TDT4265 Computer Vision and Deep Learning mini-project

Mini-project built for the subject TDT4265 Computer Vision & Deep Learning.The task was to create a YOLO-based detection of snow poles using RGB and Lidar images. We trained our model using NTNUs IDUN, and got familiar with SSH and slurm. I completed the miniproject with another student, but i did not take the exam.  

## What I built
- **Perception pipeline with ROS2**.
  - grabs ZED stereo frames
  - runs real-time inference
  - extracts depth at detection center
  - computes an approximate bearing from pixel offset
  - publishes ranked detections by depth (nearest first) over ROS2.

## File overview
- **resultsLidar** The best results (YOLO) i got from training the model with Lidar data.
- **resultsRGB** The best results (YOLO) i got from training the model with RGB data.
- **yoloTrain.py** The program used to train a model with YOLO.

## Notes  

I lost most the files i used in this subject, because i had them stored on IDUN and lost them when i lost permission.