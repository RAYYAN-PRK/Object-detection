import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

img = cv2.imread("image.jpg")
results = model(img)

annotated = results[0].plot()

cv2.imshow("Image Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
