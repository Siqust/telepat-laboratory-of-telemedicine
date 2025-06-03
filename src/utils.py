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
import logging

logger = logging.getLogger(__name__)

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

def get_poppler_path() -> str:
    """
    Получает путь к poppler
    
    Returns:
        str: путь к директории с poppler
    """
    logger.info("Начинаем поиск poppler...")
    
    # Пробуем найти poppler в разных местах
    possible_paths = [
        # Относительный путь в проекте
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'side-modules', 'poppler-24.08.0', 'Library', 'bin'),
        # Путь в системных директориях Windows
        r'C:\Program Files\poppler-24.08.0\Library\bin',
        r'C:\Program Files (x86)\poppler-24.08.0\Library\bin',
        # Путь в пользовательской директории
        os.path.expanduser('~\\poppler-24.08.0\\Library\\bin')
    ]
    
    logger.info(f"Проверяем следующие пути:")
    for path in possible_paths:
        logger.info(f"- {path}")
        if os.path.exists(path):
            logger.info(f"  Директория существует")
            if os.path.isdir(path):
                logger.info(f"  Это директория")
                # Проверяем наличие основных файлов poppler
                required_files = ['pdftoppm.exe', 'pdftotext.exe']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
                if not missing_files:
                    logger.info(f"  Найдены все необходимые файлы poppler")
                    return path
                else:
                    logger.warning(f"  Отсутствуют файлы: {', '.join(missing_files)}")
            else:
                logger.warning(f"  Это не директория")
        else:
            logger.info(f"  Директория не существует")
            
    error_msg = "Не найден poppler. Пожалуйста, установите poppler и укажите путь к нему"
    logger.error(error_msg)
    raise ValueError(error_msg)

def load_image(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Загружает изображение из файла (поддерживает PDF и изображения)
    
    Args:
        file_path: путь к файлу
        
    Returns:
        Tuple[np.ndarray, str]: (изображение, тип файла)
    """
    logger.info(f"Начинаем загрузку файла: {file_path}")
    
    file_type = get_file_type(file_path)
    logger.info(f"Определен тип файла: {file_type}")
    
    if file_type == 'pdf':
        try:
            logger.info("Обработка PDF файла...")
            poppler_path = get_poppler_path()
            logger.info(f"Используем poppler из: {poppler_path}")
            
            # Проверяем существование файла
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            logger.info(f"Файл существует, размер: {os.path.getsize(file_path)} байт")
            
            # Конвертируем первую страницу PDF в изображение
            logger.info("Начинаем конвертацию PDF в изображение...")
            images = convert_from_path(
                file_path, 
                first_page=1, 
                last_page=1, 
                poppler_path=poppler_path
            )
            
            if not images:
                raise ValueError("Не удалось конвертировать PDF в изображение")
            logger.info(f"PDF успешно конвертирован, получено {len(images)} страниц")
            
            # Конвертируем PIL Image в numpy array
            logger.info("Конвертируем PIL Image в numpy array...")
            pil_image = images[0]
            logger.info(f"PIL Image mode: {pil_image.mode}, size: {pil_image.size}")
            
            # Проверяем и конвертируем режим изображения если нужно
            if pil_image.mode != 'RGB':
                logger.info(f"Конвертируем режим изображения из {pil_image.mode} в RGB")
                pil_image = pil_image.convert('RGB')
            
            image = np.array(pil_image)
            logger.info(f"Размер numpy array: {image.shape}, dtype: {image.dtype}")
            
            # Проверяем валидность массива
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Некорректный тип данных после конвертации: {type(image)}")
            if image.dtype != np.uint8:
                raise ValueError(f"Некорректный тип данных: {image.dtype}, ожидается uint8")
            if len(image.shape) != 3:
                raise ValueError(f"Некорректная размерность: {len(image.shape)}, ожидается 3")
            if image.shape[2] != 3:
                raise ValueError(f"Некорректное количество каналов: {image.shape[2]}, ожидается 3")
            
            # Конвертируем RGB в BGR (для OpenCV)
            logger.info("Конвертируем RGB в BGR...")
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                logger.info(f"После конвертации в BGR: shape={image.shape}, dtype={image.dtype}")
            except Exception as e:
                logger.error(f"Ошибка при конвертации RGB в BGR: {str(e)}", exc_info=True)
                # Пробуем альтернативный способ конвертации
                logger.info("Пробуем альтернативный способ конвертации...")
                image = image[:, :, ::-1].copy()  # Инвертируем каналы вручную
                logger.info(f"После альтернативной конвертации: shape={image.shape}, dtype={image.dtype}")
            
            logger.info("Обработка PDF завершена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка при конвертации PDF: {str(e)}", exc_info=True)
            raise ValueError(f"Не удалось обработать PDF файл: {str(e)}")
    elif file_type == 'image':
        logger.info("Обработка изображения...")
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")
        logger.info(f"Изображение успешно загружено, размер: {image.shape}")
    else:
        error_msg = f"Неподдерживаемый тип файла: {file_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Финальная проверка изображения
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Некорректный тип данных в результате: {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Некорректный тип данных в результате: {image.dtype}")
    if len(image.shape) != 3:
        raise ValueError(f"Некорректная размерность в результате: {len(image.shape)}")
    if image.shape[2] != 3:
        raise ValueError(f"Некорректное количество каналов в результате: {image.shape[2]}")
    
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