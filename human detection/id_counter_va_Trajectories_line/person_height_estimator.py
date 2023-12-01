import torch
import cv2
from PIL import Image
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def estimate_person_height(camera_indices=[0], known_distance=2.0, known_height=1.7, stationary_threshold=10):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Create cv2.VideoCapture objects for each camera index
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    # Create windows for each camera
    window_names = [f"Camera {index} Feed" for index in camera_indices]
    for name in window_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    trajectories = {}  # Dictionary to store trajectories of each person
    id_counter = 0     # Counter for assigning unique IDs to persons

    while True:
        for cap, window_name in zip(caps, window_names):
            # Check if the camera is opened successfully
            if not cap.isOpened():
                print(f"Error: Unable to open camera.")
                continue

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

            # Draw center points and bounding boxes on the frame
            for person in persons:
                x, y, x1, y1, conf, class_id = person

                # Calculate center coordinates
                center_x = int((x + x1) / 2)
                center_y = int((y + y1) / 2)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)

                # Check if the person is already being tracked
                person_id = None
                for pid, trajectory in trajectories.items():
                    if calculate_distance(trajectory[-1], (center_x, center_y)) < stationary_threshold:
                        person_id = pid
                        break

                # If not, assign a new ID
                if person_id is None:
                    person_id = id_counter
                    id_counter += 1
                    trajectories[person_id] = []

                # Update trajectory for the person
                trajectories[person_id].append((center_x, center_y))

                # Draw trajectory lines only if the current person is being tracked
                if len(trajectories[person_id]) > 1:
                    for i in range(1, len(trajectories[person_id])):
                        # Draw lines only for the current person
                        cv2.line(frame, trajectories[person_id][i - 1], trajectories[person_id][i], (0, 255, 0), 2)

                # Calculate person height
                person_height = calculate_distance((x, y), (x, y1))  # Assuming height is the vertical distance

                # Display person height as label
                label = f"Person {person_id}: {person_height:.2f} meters"
                cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw line across the screen
            cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 0, 255), 2)

            # Display the original frame with bounding boxes, center points, trajectories, and the line
            cv2.imshow(window_name, frame)

            # Set a larger window size for the first camera (index 0)
            if window_name == window_names[0]:
                cv2.resizeWindow(window_name, 800, 600)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close all windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you run this script directly, execute the function with default values
    estimate_person_height(camera_indices=[0, 1])
