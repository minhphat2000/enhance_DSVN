import torch
import cv2
from PIL import Image

def estimate_person_height(caps, known_distance=2.0, known_height=1.7):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Allow the cameras to initialize
    for cap in caps:
        cv2.waitKey(1000)  # 1-second delay

    # Create separate windows for each camera
    window_names = [f"Camera {i}" for i in range(len(caps))]
    for window_name in window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

    while True:
        frames = [cap.read()[1] for cap in caps]

        # Check if the frames are read successfully
        if any(frame is None for frame in frames):
            print("Error: Unable to read frames from one of the cameras.")
            break

        for i, (cap, frame) in enumerate(zip(caps, frames)):
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
                focal_length = (frame.shape[1] * known_distance) / known_height
                distance = (known_height * focal_length) / (y1 - y)

                # Draw estimated height on the frame
                estimated_height = known_height * frame.shape[1] / (y1 - y)
                height_label = f"Height: {estimated_height:.2f} meters"
                cv2.putText(frame, height_label, (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw label
                cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the original frame with bounding boxes and labels in the respective window
            cv2.imshow(window_names[i], frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close all windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you run this script directly, execute the function with default values
    # Replace the camera_indices parameter with the actual camera indices
    cap_1 = cv2.VideoCapture(0)
    cap_2 = cv2.VideoCapture(1)

    estimate_person_height(caps=[cap_1, cap_2], known_distance=2.0, known_height=1.7)
