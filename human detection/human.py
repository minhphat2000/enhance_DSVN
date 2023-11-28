from person_height_estimator import estimate_person_height
import cv2

if __name__ == "__main__":
    # Specify the camera indices
    camera_indices = [0, 1]

    # Call the function with your desired parameters for each camera
    estimate_person_height(camera_indices, known_distance=2.0, known_height=1.7)
