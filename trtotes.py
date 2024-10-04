import os
import shutil

# Пути к исходным датасетам
dataset_test0 = '/home/samvel/Desktop/YOLOVeins/test0'
dataset_test1 = '/home/samvel/Desktop/YOLOVeins/test1'

# Путь к целевому датасету
dataset_test_yolov8 = '/home/samvel/Desktop/YOLOVeins'

# Создание структуры директорий
os.makedirs(os.path.join(dataset_test_yolov8, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_test_yolov8, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_test_yolov8, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_test_yolov8, 'labels', 'val'), exist_ok=True)

# Функция для копирования файлов
def copy_files(src_dir, dst_dir):
    for file_name in os.listdir(src_dir):
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst_dir)

# Копирование изображений и меток из test0
copy_files(os.path.join(dataset_test0, 'images', 'train'), os.path.join(dataset_test_yolov8, 'images', 'train'))
copy_files(os.path.join(dataset_test0, 'images', 'val'), os.path.join(dataset_test_yolov8, 'images', 'val'))
copy_files(os.path.join(dataset_test0, 'labels', 'train'), os.path.join(dataset_test_yolov8, 'labels', 'train'))
copy_files(os.path.join(dataset_test0, 'labels', 'val'), os.path.join(dataset_test_yolov8, 'labels', 'val'))

# Копирование изображений и меток из test1
copy_files(os.path.join(dataset_test1, 'images', 'train'), os.path.join(dataset_test_yolov8, 'images', 'train'))
copy_files(os.path.join(dataset_test1, 'images', 'val'), os.path.join(dataset_test_yolov8, 'images', 'val'))
copy_files(os.path.join(dataset_test1, 'labels', 'train'), os.path.join(dataset_test_yolov8, 'labels', 'train'))
copy_files(os.path.join(dataset_test1, 'labels', 'val'), os.path.join(dataset_test_yolov8, 'labels', 'val'))

print("Datasets successfully merged into test-yolov8 format.")

