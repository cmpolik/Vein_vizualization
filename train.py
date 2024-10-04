from ultralytics import YOLO

# Загрузка предобученной модели
model = YOLO('yolov8n-seg.pt')

# Запуск обучения
model.train(
    data='/home/samvel/Desktop/YOLOVeins/data.yaml',  # путь к вашему файлу конфигурации
    epochs=100,                    # количество эпох
    batch=16,                     # размер батча
    imgsz=(640, 480),             # размер изображения
    val=True
)

