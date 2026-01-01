import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano = fast

cap = cv2.VideoCapture(0)  # 0 = webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        frame = r.plot()

    cv2.imshow("Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
