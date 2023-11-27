import torch
import cv2
from PIL import Image

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5x6")  # or yolov5n - yolov5x6, custom

# Open a connection to the camera (camera index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Allow the camera to initialize
cv2.waitKey(1000)  # 1-second delay

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Unable to read a frame from the camera.")
        break

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Inference
    results = model(pil_image)

    # Get class labels and bounding boxes for persons (class 0 in COCO dataset)
    persons = [obj for obj in results.xyxy[0] if int(obj[-1]) == 0]

    # Draw bounding boxes and labels on the frame
    for person in persons:
        x, y, x1, y1, conf, class_id = person
        label = f"Person: {conf:.2f}"
        cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original frame with bounding boxes and labels
    cv2.imshow("Camera Feed", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
