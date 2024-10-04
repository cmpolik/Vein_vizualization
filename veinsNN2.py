import requests
import cv2
import numpy as np
from ultralytics import YOLO

model_weights_path = "/home/samvel/Desktop/YOLOVeins/runs100_16b/segment/train/weights/best.pt"
model = YOLO(model_weights_path)

stream_url = "http://192.168.31.101:8080/stream.mjpg"

def get_frame_from_stream(stream):
    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame

def process_frames():
    stream = requests.get(stream_url, stream=True).raw
    while True:
        frame = get_frame_from_stream(stream)
        if frame is not None:
            # Обработка кадра
            frame = cv2.equalizeHist(frame)
            clahe = cv2.createCLAHE(clipLimit=24.0, tileGridSize=(6, 6))
            frame = clahe.apply(frame)
            
            # Преобразуем одноканальное изображение в трехканальное
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Обработка кадра с использованием модели YOLO
            results = model(frame, imgsz=(640, 480), conf=0.3,  device='cuda')

            # Визуализация масок на фрейме
            annotated_frame = results[0].plot()

            # Отображение результирующего фрейма
            cv2.imshow("YOLOooo", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == '__main__':
    print('Processing frames from stream...')
    process_frames()
    cv2.destroyAllWindows()

