import cv2 as cv
from picamera2 import Picamera2, Preview
import numpy as np
import subprocess
import os

# Настройка камеры
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480), "format": "YUV420"})
picam2.configure(config)
(w0, h0) = picam2.stream_configuration("main")["size"]
(w1, h1) = picam2.stream_configuration("lores")["size"]

venv_dir = './myenv'  # Путь к виртуальному окружению

def run_camera():
    picam2.start(show_preview=False)
    
    while True:
        array = picam2.capture_array("lores")
        if array is not None:
            cap = array[:480]
            numpycap = np.array(cap)

            # Сохранение изображения во временный файл
            image_file = 'temp_image.jpg'
            cv.imwrite(image_file, numpycap)

            # Записываем временный скрипт для выполнения в виртуальном окружении
            script = f"""
import cv2
from mjpeg_streamer import MjpegServer, Stream

# Чтение изображения
image_file = 'temp_image.jpg'
frame = cv2.imread(image_file)

# Настройка стримера
server = MjpegServer(port=8080)
server.add_frame(cv2.imencode('.jpg', frame)[1].tobytes())

# Запуск стримера
server.run()
"""

            script_file = 'script_in_venv.py'
            with open(script_file, 'w') as f:
                f.write(script)

            # Выполняем скрипт в виртуальном окружении
            subprocess.run([os.path.join(venv_dir, 'bin', 'python'), script_file])

            # Удаляем временные файлы
            os.remove(image_file)
            os.remove(script_file)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

if __name__ == '__main__':
    print('Starting camera stream...')
    run_camera()
    cv.destroyAllWindows()

