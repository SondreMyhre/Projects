from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="rgb.yaml", epochs=100, imgsz=960, batch=16)