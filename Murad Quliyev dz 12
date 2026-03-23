import cv2
import torch
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"  
response = requests.get(url, stream=True)
img_pil = Image.open(response.raw).convert('RGB')
img = np.array(img_pil)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 


print("Загружаем YOLOv5...")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


results = model_yolo(img)  # img - RGB numpy array


objects_of_interest = ['person', 'car']
detections = []

for *box, conf, cls in results.xyxy[0].cpu().numpy():
    class_name = model_yolo.names[int(cls)]
    if class_name in objects_of_interest:
        x1, y1, x2, y2 = map(int, box)
        detections.append((x1, y1, x2, y2, class_name))


img_result = img_bgr.copy()
for (x1, y1, x2, y2, label) in detections:
    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)


for (x, y, w, h) in faces:
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(img_result, 'face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.title('Детекция: YOLO (person, car) + Haar (face)')
plt.show()


cv2.imwrite('result_with_green_boxes.jpg', img_result)
print("Результат сохранён как result_with_green_boxes.jpg")
