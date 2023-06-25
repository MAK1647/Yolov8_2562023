from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="data1.yaml", imgsz=640 , batch = 16 ,epochs=3)  # train the model
