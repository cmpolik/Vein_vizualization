from ultralytics import YOLO
import os
import time

model100 = "/home/samvel/Desktop/arr/YOLOVeins/runs100_16b/segment/train/weights/best.pt"
#model40 = "/home/samvel/Desktop/YOLOVeins/runs40_10b/segment/train7/weights/best.pt"

# Загрузка модели
model = YOLO(model100)  # Предобученная модель YOLOv8n

# Путь к директории с изображениями и видео для инференса
source = "/home/samvel/Desktop/arr/YOLOVeins/test5"

# Значения порога уверенности
conf_values = [0.4, 0.2, 0.3]

# Функция для выполнения инференса и сохранения результатов
def run_inference_and_save(model, source, conf):
    # Создаем директорию для сохранения результатов, если она не существует
    save_dir = os.path.join(source, f"results_conf_{conf}")
    os.makedirs(save_dir, exist_ok=True)

    # Получаем список всех изображений в директории
    image_files = [os.path.join(source, f) for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    start_time = time.time()  # Начало замера времени

    for img_path in image_files:
        # Выполнение инференса
        results = model.predict(img_path, conf=conf, stream=True, device='cuda')

        # Обработка результатов
        for result in results:
            filename = os.path.basename(img_path)
            save_path = os.path.join(save_dir, filename)
            
            result.save(save_path)  # Сохранение на диск

    end_time = time.time()  # Конец замера времени
    processing_time = end_time - start_time

    print(f"Processing time for conf={conf}: {processing_time:.2f} seconds")

# Запуск инференса для каждого значения порога уверенности
for conf in conf_values:
    run_inference_and_save(model, source, conf)

