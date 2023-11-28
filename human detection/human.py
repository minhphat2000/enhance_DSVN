from person_height_estimator import estimate_person_height
import cv2

if __name__ == "__main__":
    # Call the function with your desired parameters for each camera
    cap_1 = cv2.VideoCapture(0)
    cap_2 = cv2.VideoCapture(1)

    estimate_person_height(cap_1, known_distance=2.0, known_height=1.7)
    estimate_person_height(cap_2, known_distance=2.0, known_height=1.7)
