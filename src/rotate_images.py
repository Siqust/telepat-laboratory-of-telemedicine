import pandas as pd
import cv2
import pytesseract
import numpy as np
from tqdm import tqdm
import os
import multiprocessing

# Путь к папке с изображениями
IMAGE_DIR = 'input'
SHOW = False
ROTATIONS_FILE = 'data/rotations.py'

def resize_image(img, max_width=1400, max_height=800):
    try:
        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(f"Ошибка при масштабировании изображения: {e}")
        return None

# Функция для определения угла поворота
def correct_image_rotation(image_input):
    try:
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("image_input должен быть строкой или массивом OpenCV")

        osd = pytesseract.image_to_osd(img)
        angle = 0
        for line in osd.split('\n'):
            if 'Rotate' in line:
                angle = int(line.split(': ')[1])
                break

        if angle == 180:
            angle = 0

        # Вращение изображения не требуется, так как мы возвращаем только угол
        # Это немного ускорит процесс, избегая ненужных операций с изображением

        return angle
    except Exception as e:
        print(f"Ошибка при исправлении поворота изображения {image_input}: {e}")
        return None

def worker(task):
    """Функция-обертка для параллельной обработки"""
    filename, image_path = task
    angle = correct_image_rotation(image_path)
    return filename, angle

def main():
    # Загрузка датасета
    print('reading dataset')
    df = pd.read_csv('data/tinder.csv')

    # Выборка
    ones = df[df['label'] == 1].sample(frac=1, random_state=42)
    zeros = df[df['label'] == 0].sample(frac=1, random_state=42)
    sampled_df = pd.concat([ones, zeros]).reset_index(drop=True)
    print('read dataset')

    # Подготовка задач для пула процессов
    tasks = [
        (row['filename'], os.path.join(IMAGE_DIR, row['filename']))
        for _, row in sampled_df.iterrows()
    ]

    rotations = {}
    # Загружаем существующие данные, если они есть
    if os.path.exists(ROTATIONS_FILE):
        try:
            # Используем exec для безопасной загрузки словаря
            with open(ROTATIONS_FILE, 'r') as f:
                content = f.read()
                exec(content, globals())
        except Exception as e:
            print(f"Не удалось загрузить существующий файл rotations.py: {e}")
            rotations = {}


    print('processing rotations in parallel...')
    
    # Создаем пул процессов
    # Используем все доступные ядра, но не более 8, чтобы не перегружать систему
    num_processes = min(multiprocessing.cpu_count(), 8)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Используем imap_unordered для получения результатов по мере их готовности
        # Оборачиваем в tqdm для отображения прогресс-бара
        results_iterator = pool.imap_unordered(worker, tasks)
        
        for filename, angle in tqdm(results_iterator, total=len(tasks)):
            if angle is not None:
                rotations[filename] = angle
                # Сохраняем после каждого изменения
                try:
                    with open(ROTATIONS_FILE, 'w') as f:
                        f.write(f'rotations = {rotations}')
                except Exception as e:
                    print(f'Ошибка при сохранении rotations: {e}')

    print(f'saved rotations to {ROTATIONS_FILE}')

if __name__ == '__main__':
    # Обязательная конструкция для multiprocessing в Windows
    multiprocessing.freeze_support()
    main()
