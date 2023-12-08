#https://docs.google.com/document/d/1H_g7rkVEsXc6SeP22CpMppED-EIbIcujrkgOogyqyUY/edit?usp=sharing
#https://drive.google.com/file/d/1_HUqV5bVuOdl8WwTh4wOIPi3x_CRX1Hs/view?usp=sharing
#https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection/tree/main
https://drive.google.com/file/d/18-5e1SvWFQUSuuRLEB8AexENBXg8ZbuM/view
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or yolov5n - yolov5x6, custom

# Images
img = "https://th.bing.com/th/id/OIP.9kvuvj_y9GWIs_XqzUUTTgHaEc?rs=1&pid=ImgDetMain"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.
