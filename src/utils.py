"""
Вспомогательные функции для работы с файлами и изображениями
"""

import os
import mimetypes
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

def get_file_type(file_path: str) -> str:
    """
    Определяет тип файла по его расширению
    
    Args:
        file_path: путь к файлу
        
    Returns:
        str: тип файла ('image', 'pdf', 'unknown')
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type and mime_type.startswith('image/'):
        return 'image'
    elif mime_type == 'application/pdf':
        return 'pdf'
    return 'unknown'

def load_image(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Загружает изображение из файла (поддерживает PDF и изображения)
    
    Args:
        file_path: путь к файлу
        
    Returns:
        Tuple[np.ndarray, str]: (изображение, тип файла)
    """
    file_type = get_file_type(file_path)
    
    if file_type == 'pdf':
        # Конвертируем первую страницу PDF в изображение
        images = convert_from_path(file_path, first_page=1, last_page=1, poppler_path=r"C:\Users\Администратор\Documents\0_PROJECTS\telepat-laboratory-of-telemedicine\side-modules\poppler-24.08.0\Library\bin")
        if not images:
            raise ValueError("Не удалось конвертировать PDF в изображение")
        # Конвертируем PIL Image в numpy array
        image = np.array(images[0])
        # Конвертируем RGB в BGR (для OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif file_type == 'image':
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")
    else:
        raise ValueError(f"Неподдерживаемый тип файла: {file_type}")
    
    return image, file_type

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Предобработка изображения для улучшения качества OCR
    
    Args:
        image: входное изображение
        
    Returns:
        np.ndarray: обработанное изображение
    """
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Улучшаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Убираем шум
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    return denoised

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Сохраняет изображение в файл
    
    Args:
        image: изображение для сохранения
        output_path: путь для сохранения
    """
    # Создаем директорию если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Сохраняем изображение
    cv2.imwrite(output_path, image)

def generate_document_id() -> str:
    """
    Генерирует уникальный идентификатор для документа
    
    Returns:
        str: уникальный идентификатор
    """
    import uuid
    return str(uuid.uuid4()) 