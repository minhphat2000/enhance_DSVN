import torch
import cv2
from PIL import Image
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height))

def estimate_person_height(camera_indices=[0], known_distance=2.0, known_height=1.7, stationary_threshold=10):
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Create cv2.VideoCapture objects for each camera index
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    # Get the frame size of the first camera
    ret, frame = caps[0].read()
    if not ret:
        print(f"Error: Unable to read a frame from the first camera.")
        return

    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Create windows for each camera
    window_names = [f"Camera {index} Feed" for index in camera_indices]
    for name in window_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # Initialize in and out lines and state variables outside the loop
    middle_x = frame.shape[1] // 2
    in_line = (middle_x - 50, 0)
    out_line = (middle_x + 50, 0)

    in_count = 0       # Counter for people going in
    out_count = 0      # Counter for people going out
    inside_area = False  # Variable to track whether a person is inside the area

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

            # Resize frame to match the size of the first camera's frame
            frame = resize_frame(frame, frame_width, frame_height)

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

                # Check if the person is crossing the in line
                if (x <= in_line[0] and x1 >= in_line[0]) or (x <= in_line[0] and x1 >= in_line[1]):
                    if not inside_area:
                        in_count += 1
                        inside_area = True
                        print(f"Person {person_id} entered. Total in: {in_count}")

                # Check if the person is crossing the out line
                elif (x <= out_line[1] and x1 >= out_line[1]) or (x <= out_line[0] and x1 >= out_line[1]):
                    if inside_area:
                        out_count += 1
                        inside_area = False
                        print(f"Person {person_id} exited. Total out: {out_count}")

            # Draw in and out lines in the middle of the frame width
            cv2.line(frame, in_line, (in_line[0], frame.shape[0]), (0, 0, 255), 2)
            cv2.line(frame, out_line, (out_line[0], frame.shape[0]), (0, 0, 255), 2)

            # Label in and out lines
            cv2.putText(frame, "In Line", (in_line[0] - 40, in_line[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Out Line", (out_line[0] + 10, out_line[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display the original frame with bounding boxes, center points, trajectories, and the lines
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

    print(f"Total people entered: {in_count}")
    print(f"Total people exited: {out_count}")

if __name__ == "__main__":
    # If you run this script directly, execute the function with default values
    estimate_person_height(camera_indices=[0, 1])
