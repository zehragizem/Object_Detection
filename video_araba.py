from ultralytics import YOLO
import cv2
import time

# Modeli yükleyin
model = YOLO("yolov8n.pt")
# Video dosyasını açın
cap = cv2.VideoCapture("Drone_Video.mp4")

# Video özelliklerini alın
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_duration = 1 / fps  # Bir kare süresi
duration_to_process = 60  # İşleme süresi (saniye)
total_frames = int(duration_to_process * fps)

# Video işlemeye başla
frame_nmr = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame_nmr >= total_frames:
        break

    # Tespitleri yap
    results = model(frame)[0]

    # Her bir tespit için işlemleri yap
    for result in results.boxes.xyxy:  # Tespit edilen kutular
        x1, y1, x2, y2 = map(int, result[:4])  # Kutunun koordinatlarını alın
        
        # Araç bölgesini işaretleyin
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sonuç görüntüsünü göster
    cv2.imshow('Vehicle Detection', frame)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_nmr += 1

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
