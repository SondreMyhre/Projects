import cv2
import rclpy
from rclpy.node import Node
import pyzed.sl as sl
from std_msgs.msg import String
import json  
import numpy as np
from ultralytics import YOLO
import torch  # For CUDA-sjekk
import time

class ZEDDetectionNode(Node):
    def __init__(self):
        super().__init__('zed_detection_node')

        # ROS Publisher
        self.publisher_ = self.create_publisher(String, '/zed/detections', 10)

        # Sjekk om CUDA (GPU) er tilgjengelig
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Bruker {device} for inferens.")
        torch.backends.cudnn.benchmark = True

        # Last inn YOLO-modellen
        self.model = YOLO("best.pt").to(device)

        # Opprett ZED-kameraobjekt
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA  # Sett oppløsning til 720p
        init_params.camera_fps = 20
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Bruk ytelsesmodus for dybdemåling

        # Åpne kameraet
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open camera, exiting.")
            exit()

        # Konfigurasjoner for runtime
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.inference_interval_time = 1/20  # 20 FPS
        self.last_inference_time = 0

        # Timer for å kjøre inferens
        self.timer = self.create_timer(0.05, self.process_frame)  # 20 Hz

    def process_frame(self):
        print(f"[{time.time():.2f}] process_frame() kalt")
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)  # Hent venstre synsbilde fra ZED
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)  # Hent dybdedata

            # Konverter ZED-bildet til en numpy array som YOLO kan prosessere
            frame = self.image.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Konverter fra BGRA til BGR for OpenCV/YOLO

            # Bare kjør YOLO-inferens hvis det har gått nok tid siden forrige inferens
            current_time = time.time()
            if current_time - self.last_inference_time >= self.inference_interval_time:  
                self.last_inference_time = current_time  
                print(f"[{current_time:.2f}] Kjører YOLO inferens...")

                # Kjør YOLOv8-inferens på ZED-bildet
                try:
                    results = self.model(frame_rgb)
                except Exception as e:
                    print(f"[{current_time:.2f}] Error during YOLO inference: {e}")
                    return

                # Samle detections i en liste, sortert etter dybde
                detections = []
                for detection in results[0].boxes:
                    xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])  # YOLO gir bounding box koordinater

                    # Finn midten av bounding box
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2

                    # Hent dybden fra ZED-kameraet på punktet (x_center, y_center)
                    depth_value = self.depth.get_value(x_center, y_center)[1]  # Bruker indeksen [1] for å hente dybdeverdi
                    angle_degrees = calculate_angle(frame_rgb.shape[1], x_center)  # Henter vinkel fra midtlinje
                    if np.isnan(depth_value):  # Sjekk om dybdeverdien er NaN, og hopp over hvis den er det
                        print(f"[{current_time:.2f}] Ugyldig dybdeverdi for deteksjon ved ({x_center}, {y_center}).")
                        continue

                    label = detection.cls[0]
                    detection_data = {
                        "label": int(label),  # YOLO gir ofte klasse som float
                        "position": (x_center, y_center),
                        "depth": float(depth_value),  # Konverter til float for JSON
                        "angle": float(angle_degrees),
                    }
                    detections.append(detection_data)

                # Sorter detections etter dybde (nærmeste først)
                detections.sort(key=lambda d: d["depth"])

                # Publiser listen over deteksjoner som JSON
                msg = String()
                msg.data = json.dumps(detections)
                self.publisher_.publish(msg)
                print(f"[{current_time:.2f}] Publisert {len(detections)} deteksjoner på /zed/detections")

def calculate_angle(image_width, object_x):
    fov_horizontal_degrees = 110  # (Grader)
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