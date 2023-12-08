import cv2

RTSP_URL = 'rtsp://admin:Kingkem2@192.168.1.6:554/11' 

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
print(cap)
if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
