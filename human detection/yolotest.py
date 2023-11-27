import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or yolov5n - yolov5x6, custom

# Images
img = "https://th.bing.com/th/id/OIP.9kvuvj_y9GWIs_XqzUUTTgHaEc?rs=1&pid=ImgDetMain"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.