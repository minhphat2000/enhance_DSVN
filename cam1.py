import torch
import cv2
from PIL import Image

def estimate_person_height(known_distance=2.0, known_height=1.7):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Create cv2.VideoCapture object for the camera
    cap = cv2.VideoCapture(1)

    # Create window for the camera
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    while True:
        # Check if the camera is opened successfully
        if not cap.isOpened():
            print(f"Error: Unable to open camera.")
            break

        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print(f"Error: Unable to read a frame from the camera.")
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

            # Draw bounding box
            cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)

            # Calculate the distance from the camera to the person
            # Assuming a simple perspective model (not considering lens distortion)
            focal_length = (frame.shape[1] * known_distance) / known_height
            distance = (known_height * focal_length) / (y1 - y)

            # Draw estimated height on the frame
            estimated_height = known_height * frame.shape[1] / (y1 - y)
            height_label = f"Height: {estimated_height:.2f} meters"
            cv2.putText(frame, height_label, (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw label
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the original frame with bounding boxes and labels
        cv2.imshow("Camera Feed", frame)

        # Set a larger window size
        cv2.resizeWindow("Camera Feed", 800, 600)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you run this script directly, execute the function with default values
    estimate_person_height()
