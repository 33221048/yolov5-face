import cv2
import torch

# Memuat model pre-trained
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Memuat gambar
image = cv2.imread('./George_W_Bush/George_W_Bush_0001.jpg')

# Jalankan deteksi
results = model(image)

# Menampilkan hasil deteksi
for result in results.xyxy[0]:
    xmin, ymin, xmax, ymax, confidence, class_id = map(int, result)
    class_name = model.names[class_id]
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Menampilkan gambar
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()