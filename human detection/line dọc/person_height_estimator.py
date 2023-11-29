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

    left_count = 0
    right_count = 0

    prev_positions = {}  # Store previous positions of persons

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

            # Draw bounding boxes, labels, center points on the frame
            for person in persons:
                x, y, x1, y1, conf, class_id = person
                label = f"Person: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)

                # Draw center point
                center_x = int((x + x1) / 2)
                center_y = int((y + y1) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # -1 fills the circle

                # Check if the person has moved from the left to the right or vice versa
                person_id = int(person[4])  # Using confidence as a unique identifier
                if person_id not in prev_positions:
                    prev_positions[person_id] = center_x
                else:
                    prev_center_x = prev_positions[person_id]

                    # Count persons when they move from the left to the right
                    if prev_center_x < frame.shape[1] / 2 and center_x >= frame.shape[1] / 2:
                        right_count += 1
                    # Count persons when they move from the right to the left
                    elif prev_center_x >= frame.shape[1] / 2 and center_x < frame.shape[1] / 2:
                        left_count += 1

                    # Update previous position for the next iteration
                    prev_positions[person_id] = center_x

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

            # Draw counts on the frame
            count_label = f"Vao: {left_count} | Ra: {right_count}"
            cv2.putText(frame, count_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw center line outside the loop
        center_line_color = (0, 0, 255)  # Red color for the line
        center_line_thickness = 2
        cv2.line(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2), frame.shape[0]), center_line_color, center_line_thickness)

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
