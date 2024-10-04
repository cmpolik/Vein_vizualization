from ultralytics import YOLO
import os

model100 = "/home/samvel/Desktop/YOLOVeins/runs100_16b/segment/train/weights/best.pt"
model40 = "/home/samvel/Desktop/YOLOVeins/runs40_10b/segment/train7/weights/best.pt"

# Загрузка модели
model = YOLO(model100)  # Предобученная модель YOLOv8n

# Путь к директории с изображениями и видео для инференса
source = "/home/samvel/Desktop/YOLOVeins/test3"

# Значения порога уверенности
conf_values = [0.2, 0.3, 0.5, 0.7, 0.8]

# Функция для выполнения инференса и сохранения результатов
def run_inference_and_save(model, source, conf):
    # Создаем директорию для сохранения результатов, если она не существует
    save_dir = os.path.join(source, f"results_conf_{conf}")
    os.makedirs(save_dir, exist_ok=True)

    # Получаем список всех изображений в директории
    image_files = [os.path.join(source, f) for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    for img_path in image_files:
        # Выполнение инференса
        results = model(img_path, conf=conf, stream=True)

        # Обработка результатов
        for result in results:
            filename = os.path.basename(img_path)
            save_path = os.path.join(save_dir, filename)
            
            result.save(save_path)  # Сохранение на диск

# Запуск инференса для каждого значения порога уверенности
for conf in conf_values:
    run_inference_and_save(model, source, conf)

