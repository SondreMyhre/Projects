import cv2
import rclpy
from rclpy.node import Node
import pyzed.sl as sl
from std_msgs.msg import String
import json  
import numpy as np
from ultralytics import YOLO
import torch
import time

"""
ROS2 node for buoy detections with ZED + YOLOv8.

Publishes:
- /zed/detections (std_msgs/String), JSON list sorted by depth.
Each detection:
  {
    "label": int,
    "position": [cx, cy],
    "depth_m": float,
    "angle_deg": float
  }

Depth units are meters (coordinate_units set explicitly).
Bearing is a simple pixel->FOV approximation.
"""

class ZEDDetectionNode(Node):
    def __init__(self):
        super().__init__('zed_detection_node')

        self.publisher_ = self.create_publisher(String, '/zed/detections', 10)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Bruker {device} for inferens.")
        torch.backends.cudnn.benchmark = True

        self.model = YOLO("best.pt").to(device)

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA  
        init_params.camera_fps = 20
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open camera, exiting.")
            exit()

        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.inference_interval_time = 1/20
        self.last_inference_time = 0

        self.timer = self.create_timer(0.05, self.process_frame)

    def process_frame(self):
        print(f"[{time.time():.2f}] process_frame() kalt")
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

            frame = self.image.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            current_time = time.time()
            if current_time - self.last_inference_time >= self.inference_interval_time:  
                self.last_inference_time = current_time  
                print(f"[{current_time:.2f}] Kjører YOLO inferens...")

                try:
                    results = self.model(frame_rgb)
                except Exception as e:
                    print(f"[{current_time:.2f}] Error during YOLO inference: {e}")
                    return

                detections = []
                for detection in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])

                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2

                    depth_value = self.depth.get_value(x_center, y_center)[1]
                    angle_degrees = calculate_angle(frame_rgb.shape[1], x_center)
                    if np.isnan(depth_value):
                        print(f"[{current_time:.2f}] Ugyldig dybdeverdi for deteksjon ved ({x_center}, {y_center}).")
                        continue

                    label = detection.cls[0]
                    detection_data = {
                        "label": int(label),
                        "position": (x_center, y_center),
                        "depth": float(depth_value),
                        "angle": float(angle_degrees),
                    }
                    detections.append(detection_data)

                detections.sort(key=lambda d: d["depth"])

                msg = String()
                msg.data = json.dumps(detections)
                self.publisher_.publish(msg)
                print(f"[{current_time:.2f}] Publisert {len(detections)} deteksjoner på /zed/detections")

def calculate_angle(image_width, object_x):
    fov_horizontal_degrees = 110
    center_x = image_width / 2
    pixel_offset = object_x - center_x
    angle_per_pixel = fov_horizontal_degrees / image_width
    angle_offset_degrees = pixel_offset * angle_per_pixel

    return angle_offset_degrees

def main(args=None):
    rclpy.init(args=args)
    node = ZEDDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()