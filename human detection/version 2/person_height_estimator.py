import torch
import cv2
from PIL import Image

def estimate_person_height(camera_indices=[0], known_distance=2.0, known_height=1.7):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Create cv2.VideoCapture objects for each camera index
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    # Create windows for each camera
    window_names = [f"Camera {index} Feed" for index in camera_indices]
    for name in window_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    initial_positions = {}
    direction_counts = {'enter': 0, 'exit': 0}

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

            # Draw bounding boxes and labels on the frame
            for person in persons:
                x, y, x1, y1, conf, class_id = person
                label = f"Person: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)

                # Calculate the distance from the initial position
                person_id = int(person[4])  # Using confidence as a unique identifier
                if person_id not in initial_positions:
                    initial_positions[person_id] = (x, y)
                else:
                    initial_x, initial_y = initial_positions[person_id]
                    distance_x = x - initial_x
                    distance_y = y - initial_y

                    # Draw estimated movement on the frame
                    movement_label = f"Movement: ({distance_x:.2f}, {distance_y:.2f})"
                    cv2.putText(frame, movement_label, (int(x), int(y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Count entering and exiting people based on movement direction
                    if distance_x > 0:
                        direction_counts['enter'] += 1
                    else:
                        direction_counts['exit'] += 1

                    # Draw count on the frame
                    count_label = f"Enter: {direction_counts['enter']} | Exit: {direction_counts['exit']}"
                    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    # Update initial position for the next iteration
                    initial_positions[person_id] = (x, y)

                # Draw label
                cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the original frame with bounding boxes and labels
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
