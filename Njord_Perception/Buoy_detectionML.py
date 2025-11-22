import cv2
import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
import torch  # For CUDA-sjekk
import time

"""
Standalone demo for buoy detection with a ZED stereo camera + YOLOv8.

- Grabs left RGB frame from ZED
- Runs YOLO inference
- Queries depth at bbox center
- Displays annotated feed with depth and FPS

Assumes ZED depth is returned in meters (we set coordinate_units explicitly).
"""

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Bruker {device} for inferens.")

    model = YOLO("best.pt").to(device)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera, exiting.")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    fps_time = time.time()

    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                frame = image.get_data()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                try:
                    results = model(frame_rgb)
                except Exception as e:
                    print(f"Error during YOLO inference: {e}")
                    break

                annotated_frame = results[0].plot()

                for detection in results[0].boxes:

                    xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])
                    
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2

                    depth_value = depth.get_value(x_center, y_center)[1]
                    if np.isnan(depth_value):
                        continue

                    label = detection.cls[0]

                    cv2.putText(annotated_frame, f"{label}: {depth_value:.2f}m", 
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                current_time = time.time()
                fps = 1 / (current_time - fps_time)
                fps_time = current_time

                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow("YOLOv8 ZED Camera with Depth", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
