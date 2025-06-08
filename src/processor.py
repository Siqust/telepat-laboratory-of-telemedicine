import os
import re
from typing import Dict, List, Tuple, Optional, Set
import cv2
import numpy as np
import pytesseract
import logging
from data_manager import DataManager
import uuid
import utils
import torch
import stanza
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from medical_terms import MedicalTermsManager
from loguru import logger
from deepseek_client import DeepSeekClient
import asyncio
from pdf2image import convert_from_path
import tempfile
import subprocess
import win32api
import atexit
import shutil
from PIL import Image
import io
import time
from gigachat_client import GigaChatClient


def get_short_path(path: str) -> str:
    """
    Получает короткий путь Windows (8.3 формат) для указанного пути

    Args:
        path: путь для конвертации

    Returns:
        str: короткий путь Windows
    """
    try:
        return win32api.GetShortPathName(str(path))
    except Exception as e:
        logger.warning(f"Не удалось получить короткий путь для {path}: {e}")
        return str(path)


# Путь к локальному poppler (Windows версия)
POPPLER_PATH = Path("side-modules/poppler-24.08.0/Library/bin").absolute()
PDFTOPPM = POPPLER_PATH / "pdftoppm.exe"

# Создаем временную директорию в корне проекта
TEMP_DIR = Path("temp_pdf_images").absolute()
TEMP_DIR.mkdir(exist_ok=True)

# Проверяем существование файла pdftoppm.exe
if not PDFTOPPM.exists():
    logger.error(f"Файл pdftoppm.exe не найден по пути: {PDFTOPPM}")
    raise RuntimeError(f"Файл pdftoppm.exe не найден по пути: {PDFTOPPM}")

# Проверяем доступность poppler
try:
    result = subprocess.run([str(PDFTOPPM), "-v"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Poppler не найден: {result.stderr}")
    logger.info(f"Poppler найден: {result.stdout.strip()}")
except Exception as e:
    logger.error(f"Ошибка при проверке poppler: {str(e)}")
    raise RuntimeError("Poppler не установлен или недоступен")

# Добавляем путь к poppler в системную переменную PATH
os.environ["PATH"] = str(POPPLER_PATH) + os.pathsep + os.environ["PATH"]

logger = logging.getLogger(__name__)

# Словарь для транслитерации
TRANSLIT_DICT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
}


def transliterate(text: str) -> str:
    """
    Транслитерирует русский текст в латиницу
    
    Args:
        text: текст для транслитерации
        
    Returns:
        str: транслитерированный текст
    """
    result = []
    for char in text:
        result.append(TRANSLIT_DICT.get(char, char))
    return ''.join(result)


class DocumentProcessor:
    """Класс для обработки медицинских документов"""

    async def __init__(self, data_manager: DataManager):
        """
        Инициализация процессора документов

        Args:
            data_manager: менеджер данных для сохранения результатов
        """
        self.data_manager = data_manager
        self.deepseek_client = DeepSeekClient()
        self.gigachat_client = GigaChatClient()
        
        # Инициализация списков слов
        self.allowed_words = set([
            'врач', 'доктор', 'профессор', 'академик', 'заведующий',
            'главный', 'старший', 'младший', 'ординатор', 'интерн',
            'медицинский', 'клинический', 'диагноз', 'заключение',
            'рекомендации', 'лечение', 'терапия', 'процедура', 'анализ',
            'исследование', 'обследование', 'консультация', 'прием',
            'отделение', 'кабинет', 'клиника', 'больница', 'поликлиника',
            'центр', 'лаборатория', 'диспансер', 'санаторий'
        ])
        
        # Инициализация паттернов для числовых данных
        self.numeric_patterns = {
            'inn': {
                'pattern': r'^(?:\d{10}|\d{12}|(?:\d{2}[-\s]?){5}\d{2}|(?:\d{3}[-\s]?){3}\d{3}|(?:\d{4}[-\s]?){2}\d{4}|(?:\d{6}[-\s]?)\d{4}|(?:\d{5}[-\s]?)\d{5}|(?:\d{4}[-\s]?)\d{6}|(?:\d{3}[-\s]?)\d{7}|(?:\d{2}[-\s]?)\d{8}|\d{1}[-\s]?\d{9})$',
                'description': 'ИНН',
                'ocr_errors': {
                    '0': ['O', 'o', 'О', 'о'],
                    '1': ['I', 'i', 'l', 'L', '|'],
                    '2': ['Z', 'z', 'З', 'з'],
                    '3': ['З', 'з', 'Э', 'э'],
                    '4': ['Ч', 'ч'],
                    '5': ['S', 's', 'Б', 'б'],
                    '6': ['G', 'g', 'б', 'Б'],
                    '7': ['Т', 'т'],
                    '8': ['В', 'в'],
                    '9': ['g', 'G', 'д', 'Д']
                }
            },
            'phone': {
                'pattern': r'^(?:\+?7|8)?[-\s\(]*\d{2,4}[-\s\)]*\d{2,4}[-\s]*\d{2,4}[-\s]*\d{2,4}$|^fax[:\s]*\d{2,4}[-\s\(]*\d{2,4}[-\s\)]*\d{2,4}[-\s]*\d{2,4}[-\s]*\d{2,4}$|тел\.[\s\/]*факс[:\s]*\d{2,4}[-\s\(]*\d{2,4}[-\s\)]*\d{2,4}[-\s]*\d{2,4}[-\s]*\d{2,4}$|телефон[:\s]*\d{2,4}[-\s\(]*\d{2,4}[-\s\)]*\d{2,4}[-\s]*\d{2,4}[-\s]*\d{2,4}$',
                'description': 'Телефон',
                'ocr_errors': {
                    '0': ['O', 'o', 'О', 'о'],
                    '1': ['I', 'i', 'l', 'L', '|'],
                    '2': ['Z', 'z', 'З', 'з'],
                    '3': ['З', 'з', 'Э', 'э'],
                    '4': ['Ч', 'ч'],
                    '5': ['S', 's', 'Б', 'б'],
                    '6': ['G', 'g', 'б', 'Б'],
                    '7': ['Т', 'т'],
                    '8': ['В', 'в'],
                    '9': ['g', 'G', 'д', 'Д'],
                    '+': ['t', 'Т', 'т'],
                    '(': ['С', 'с', 'C', 'c'],
                    ')': ['С', 'с', 'C', 'c']
                }
            },
            'passport': {
                'pattern': r'^\d{4}\s?\d{6}$',
                'description': 'Паспорт'
            },
            'snils': {
                'pattern': r'^\d{3}-\d{3}-\d{3}\s?\d{2}$',
                'description': 'СНИЛС'
            },
            'policy': {
                'pattern': r'^\d{16}$',
                'description': 'Полис ОМС'
            },
            'med_card': {
                'pattern': r'^[А-Я]\d{6}|\d{6}[А-Я]$',
                'description': 'Номер медкарты'
            },
            'birth_date': {
                'pattern': r'^\d{2}\.\d{2}\.\d{4}$|^\d{8}$|^\d{1,2}\s(янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)\s\d{4}$',
                'description': 'Дата рождения',
                'ocr_errors': {
                    '0': ['O', 'o', 'О', 'о'],
                    '1': ['I', 'i', 'l', 'L', '|'],
                    '2': ['Z', 'z', 'З', 'з'],
                    '3': ['З', 'з', 'Э', 'э'],
                    '4': ['Ч', 'ч'],
                    '5': ['S', 's', 'Б', 'б'],
                    '6': ['G', 'g', 'б', 'Б'],
                    '7': ['Т', 'т'],
                    '8': ['В', 'в'],
                    '9': ['g', 'G', 'д', 'Д']
                }
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'description': 'Email'
            },
            'postal_code': {
                'pattern': r'^\d{6}$|^\d{3}\s\d{3}$',
                'description': 'Почтовый индекс',
                'ocr_errors': {
                    '0': ['O', 'o', 'О', 'о'],
                    '1': ['I', 'i', 'l', 'L', '|'],
                    '2': ['Z', 'z', 'З', 'з'],
                    '3': ['З', 'з', 'Э', 'э'],
                    '4': ['Ч', 'ч'],
                    '5': ['S', 's', 'Б', 'б'],
                    '6': ['G', 'g', 'б', 'Б'],
                    '7': ['Т', 'т'],
                    '8': ['В', 'в'],
                    '9': ['g', 'G', 'д', 'Д']
                }
            }
        }
        
        # Ключевые слова адресов, которые могут быть короткими или иметь нетипичный регистр, но важны для маскирования
        self.address_keywords = {
            'ул.', 'улица', 'пер.', 'переулок', 'пр.', 'проспект', 'ш.', 'шоссе', 
            'наб.', 'набережная', 'пл.', 'площадь', 'д.', 'дом', 'к.', 'корп.', 
            'корпус', 'кв.', 'квартира', 'стр.', 'строение', 'литера', 'лит.'
        }

        # Медицинские паттерны
        self.medical_patterns = {
            'date': r'(\d{2}\.\d{2}\.\d{4})',
            'blood_pressure': r'(\d{2,3}/\d{2,3})',
            'temperature': r'(\d{2}\.\d)',
            'pulse': r'(\d{2,3})\s*уд/мин',
            'weight': r'(\d{2,3}(?:\.\d)?)\s*кг',
            'height': r'(\d{3})\s*см',
            'diagnosis': r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:болезнь|синдром|симптом|патология|состояние)\b',
            'procedure': r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:терапия|лечение|процедура|манипуляция)\b',
            'department': r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:отделение|кабинет|палата|центр)\b',
            'analysis': r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:анализ|исследование|тест|проба)\b',
            'medication': r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:препарат|лекарство|средство|медикамент)\b'
        }
        
        self.surnames = set()  # Будет заполнено в _load_surnames
        self.translit_surnames = set()  # Будет заполнено в _load_surnames
        self.cities_to_mask = set()  # Будет заполнено в _load_cities
        self.medical_terms = set()  # Будет заполнено в _load_medical_terms
        self.patronymics = set()  # Будет заполнено в _load_patronymics
        self.names = set()  # Будет заполнено в _load_names
        
        # Загрузка медицинских терминов и других данных
        await self._load_medical_terms()
        await self._load_surnames()
        await self._load_cities()
        await self._load_patronymics()
        await self._load_names()
        
        # Проверяем доступность API
        await self.deepseek_client._check_api_availability()
        
        # Создаем временную директорию для обработки PDF
        self.temp_dir = Path('temp_pdf_images')
        self.temp_dir.mkdir(exist_ok=True)

    async def _load_surnames(self) -> None:
        """
        Загружает список фамилий из файла
        """
        try:
            surnames_file = Path('src/data/russian-words/russian_surnames.txt')
            if not surnames_file.exists():
                logger.error(f"Файл со списком фамилий не найден: {surnames_file}")
                return

            with open(surnames_file, 'r', encoding='windows-1251') as f:
                self.surnames = {line.strip().lower() for line in f if line.strip()}
                self.translit_surnames = {transliterate(surname) for surname in self.surnames}
            
            logger.info(f"Загружено {len(self.surnames)} фамилий из файла")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке фамилий: {e}")
            self.surnames = set()
        self.translit_surnames = set()

    async def _load_cities(self) -> None:
        """
        Загружает список городов из JSON файла
        """
        try:
            cities_file = Path('src/data/cities.json')
            if not cities_file.exists():
                logger.error(f"Файл со списком городов не найден: {cities_file}")
                return

            with open(cities_file, 'r', encoding='utf-8') as f:
                cities_data = json.load(f)
                if not isinstance(cities_data, list):
                    raise ValueError("Файл городов должен содержать список")

                for city in cities_data:
                    if isinstance(city, str):
                        city_lower = city.lower()
                        self.cities_to_mask.add(city.strip())
                        self.cities_to_mask.add(city_lower.strip())
                        self.cities_to_mask.add(city.upper().strip())
                        self.cities_to_mask.add(city.capitalize().strip())

            logger.info(f"Загружено {len(cities_data)} городов из JSON файла")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке городов: {e}")
            self.cities_to_mask = set()

    async def _load_patronymics(self) -> None:
        """
        Загружает список отчеств из файла
        """
        try:
            patronymics_file = Path('src/data/russian_patronymics.txt')
            if not patronymics_file.exists():
                logger.error(f"Файл со списком отчеств не найден: {patronymics_file}")
                return

            with open(patronymics_file, 'r', encoding='utf-8') as f:
                self.patronymics = {line.strip().lower() for line in f if line.strip()}
                self.translit_patronymics = {transliterate(patronymic) for patronymic in self.patronymics}
            
            logger.info(f"Загружено {len(self.patronymics)} отчеств из файла")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке отчеств: {e}")
            self.patronymics = set()
            self.translit_patronymics = set()

    async def _load_names(self) -> None:
        """
        Загружает список имен из файла
        """
        try:
            names_file = Path('src/data/russian_names.txt')
            if not names_file.exists():
                logger.error(f"Файл со списком имен не найден: {names_file}")
                return

            with open(names_file, 'r', encoding='utf-8') as f:
                self.names = {line.strip().lower() for line in f if line.strip()}
                self.translit_names = {transliterate(name) for name in self.names}
            
            logger.info(f"Загружено {len(self.names)} имен из файла")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке имен: {e}")
            self.names = set()
            self.translit_names = set()

    async def _load_medical_terms(self) -> None:
        """
        Загружает медицинские термины из JSON файлов
        """
        medical_terms_files = [
            'src/data/medical_abbreviations.json',
            'src/data/medical_terms_whitelist.json',
            'src/data/russian_stats_terms.json',
            'src/data/russian_medical_terms_whitelist.json',
            'src/data/english_stats_terms.json',
            'src/data/english_medical_terms.json',
            'src/data/russian_medical_terms_symptoms.json',
            'src/data/russian_medical_terms_diagnosis.json',
            'src/data/russian_medical_terms_anatomy.json',
            'src/data/russian_medical_terms_drugs.json',
            'src/data/russian_medical_terms_procedures.json'
        ]

        for terms_file in medical_terms_files:
            try:
                with open(terms_file, 'r', encoding='utf-8') as f:
                    terms = json.load(f)
                    if isinstance(terms, list):
                        self.medical_terms.update(term.lower() for term in terms)
                    elif isinstance(terms, dict):
                        self.medical_terms.update(term.lower() for term in terms.keys())
                logger.info(f"Загружено терминов из {terms_file}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке терминов из {terms_file}: {e}")

        # Расширяем список медицинских терминов
        self.medical_terms.update({
            'анамнез', 'диагноз', 'жалобы', 'обследование', 'терапия',
            'лечение', 'процедура', 'манипуляция', 'операция', 'реабилитация',
            'консультация', 'наблюдение', 'диспансеризация', 'скрининг',
            'профилактика', 'вакцинация', 'иммунизация', 'диета', 'режим',
            'рекомендации', 'назначения', 'показания', 'противопоказания',
            'осложнения', 'побочные', 'эффекты', 'аллергия', 'непереносимость',
            'хронический', 'острый', 'подострый', 'ремиссия', 'обострение',
            'прогрессирование', 'стабилизация', 'улучшение', 'ухудшение',
            'выздоровление', 'рецидив', 'метастаз', 'метастазирование'
        })

        # Добавляем контекстные маркеры для разных типов документов
        self.document_contexts = {
            'medical_card': {
                'markers': [
                    r'МЕДИЦИНСКАЯ КАРТА',
                    r'ИСТОРИЯ БОЛЕЗНИ',
                    r'ДАННЫЕ ПАЦИЕНТА',
                    r'ЖАЛОБЫ',
                    r'АНАМНЕЗ'
                ],
                'sensitive_sections': ['ДАННЫЕ ПАЦИЕНТА'],
                'medical_sections': ['ЖАЛОБЫ', 'АНАМНЕЗ', 'ДИАГНОЗ', 'ЛЕЧЕНИЕ']
            },
            'discharge_summary': {
                'markers': [
                    r'ВЫПИСКА',
                    r'ЭПИКРИЗ',
                    r'ЗАКЛЮЧЕНИЕ',
                    r'РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ'
                ],
                'sensitive_sections': ['ПАЦИЕНТ'],
                'medical_sections': ['ДИАГНОЗ', 'ЛЕЧЕНИЕ', 'РЕЗУЛЬТАТЫ']
            },
            'operation_report': {
                'markers': [
                    r'ПРОТОКОЛ ОПЕРАЦИИ',
                    r'ОПЕРАЦИЯ',
                    r'ХИРУРГИЧЕСКОЕ ВМЕШАТЕЛЬСТВО'
                ],
                'sensitive_sections': ['ПАЦИЕНТ'],
                'medical_sections': ['ДИАГНОЗ', 'ОПЕРАЦИЯ', 'ОСЛОЖНЕНИЯ']
            }
        }

    @classmethod
    async def create(cls, data_manager: DataManager) -> 'DocumentProcessor':
        """
        Фабричный метод для создания экземпляра DocumentProcessor
        
        Args:
            data_manager: менеджер данных для сохранения результатов
            
        Returns:
            DocumentProcessor: инициализированный экземпляр процессора
        """
        processor = cls.__new__(cls)
        await processor.__init__(data_manager)
        return processor

    def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        temp_dir = Path('temp_pdf_images')
        if temp_dir.exists():
            try:
                # Добавляем небольшую задержку перед удалением
                time.sleep(1)  # Даем время на освобождение файлов

                for file in temp_dir.glob('*'):
                    try:
                        # Проверяем, не используется ли файл
                        if file.is_file():
                            try:
                                # Пробуем открыть файл на запись, чтобы
                                # проверить, не занят ли он
                                with open(file, 'a'):
                                    pass
                                # Если файл не занят, удаляем его
                                file.unlink()
                                logger.debug(f"Удален временный файл: {file}")
                            except PermissionError:
                                logger.warning(
                                    f"Файл {file} все еще используется, пропускаем")
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Не удалось удалить файл {file}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Ошибка при обработке файла {file}: {str(e)}")
                        continue
                
                # Пробуем удалить саму директорию
                try:
                    temp_dir.rmdir()
                    logger.debug("Временная директория успешно удалена")
                except Exception as e:
                    logger.warning(
                        f"Не удалось удалить временную директорию: {str(e)}")

            except Exception as e:
                logger.error(f"Ошибка при очистке временных файлов: {str(e)}")

    def _is_medical_term(self, word: str) -> bool:
        """
        Проверяет, является ли слово медицинским термином
        
        Args:
            word: проверяемое слово
            
        Returns:
            bool: является ли слово медицинским термином
        """
        return word.lower() in self.medical_terms

    def _fix_image_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Определяет и исправляет ориентацию изображения
        
        Args:
            image: исходное изображение
            
        Returns:
            np.ndarray: изображение с исправленной ориентацией
        """
        try:
            logger.info("Проверка ориентации изображения...")
            
            # Пробуем определить ориентацию с помощью Tesseract
            osd = pytesseract.image_to_osd(image)
            angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
            
            if angle != 0:
                logger.info(f"Обнаружен поворот изображения на {angle} градусов")
                # Поворачиваем изображение
                if angle == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                logger.info("Изображение повернуто")
                return image
                
            # Если Tesseract не определил поворот, проверяем направление текста
            # Получаем все строки текста
            data = pytesseract.image_to_data(
                image, 
                lang='rus+eng',
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Считаем количество строк с разной ориентацией
            horizontal_lines = 0
            vertical_lines = 0
            
            current_line = []
            current_top = None
            
            for i in range(len(data['text'])):
                if not data['text'][i].strip():
                    continue
                    
                if current_top is None:
                    current_top = data['top'][i]
                    current_line.append(data['left'][i])
                elif abs(data['top'][i] - current_top) < data['height'][i] * 0.5:
                    # Слова на одной строке
                    current_line.append(data['left'][i])
                else:
                    # Новая строка
                    if len(current_line) > 1:
                        # Проверяем направление строки
                        if max(current_line) - min(current_line) > data['height'][i]:
                            horizontal_lines += 1
                        else:
                            vertical_lines += 1
                    current_line = [data['left'][i]]
                    current_top = data['top'][i]
            
            # Проверяем последнюю строку
            if len(current_line) > 1:
                if max(current_line) - min(current_line) > data['height'][i]:
                    horizontal_lines += 1
                else:
                    vertical_lines += 1
            
            # Если вертикальных строк больше, поворачиваем изображение
            if vertical_lines > horizontal_lines:
                logger.info("Обнаружена преимущественно вертикальная ориентация текста")
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                logger.info("Изображение повернуто на 90 градусов")
            
            return image
            
        except Exception as e:
            logger.warning(f"Ошибка при определении ориентации изображения: {str(e)}")
            return image

    def _detect_pen(self, image: np.ndarray) -> bool:
        """
        Определяет наличие ручки на изображении
        
        Args:
            image: изображение для анализа
            
        Returns:
            bool: True если ручка обнаружена, False если нет
        """
        try:
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем размытие для уменьшения шума
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Находим края с помощью детектора Канни
            edges = cv2.Canny(blurred, 50, 150)
            
            # Находим линии с помощью преобразования Хафа
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return False
                
            # Конвертируем в HSV для поиска характерных цветов ручки
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Определяем диапазоны цветов, характерных для ручек
            # Синий
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            # Красный (два диапазона из-за особенностей HSV)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            # Создаем маски для каждого цвета
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Объединяем маски (только синий и красный)
            pen_colors_mask = cv2.bitwise_or(blue_mask, red_mask)
            
            # Находим контуры на маске цветов
            contours, _ = cv2.findContours(pen_colors_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Анализируем найденные линии и контуры
            pen_detected = False
            
            # Проверяем линии на характерные признаки ручки
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Ручка обычно длинная и прямая
                if length > 100 and (angle < 20 or angle > 160):
                    # Проверяем, есть ли рядом контуры характерных цветов
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Игнорируем слишком маленькие области
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Проверяем, находится ли контур рядом с линией
                                if (min(x1, x2) - 20 <= cx <= max(x1, x2) + 20 and
                                    min(y1, y2) - 20 <= cy <= max(y1, y2) + 20):
                                    pen_detected = True
                                    break
                
                    if pen_detected:
                        break
            
            return pen_detected
            
        except Exception as e:
            logger.error(f"Ошибка при определении ручки: {str(e)}")
            return False

    async def process_document(self, file_path: str, clinic_name: str, output_dir: str) -> Tuple[str, Dict]:
        """Обработка документа: конвертация, распознавание текста, маскирование и сохранение"""
        # Инициализация переменных
        mask_context_active_for_line = False
        current_line_words_data = []
        sensitive_regions = []
        temp_files = []
        temp_dir = Path('temp_pdf_images')
        temp_dir.mkdir(exist_ok=True)
        output_file = None
        image = None

        try:
            # Подготовка путей
            file_path = Path(file_path).absolute()
            output_dir = Path(output_dir).absolute()
            output_dir.mkdir(parents=True, exist_ok=True)

            # Создаем директорию для результатов AI в корне проекта
            ai_results_dir = Path('ai-result/deepseek').absolute()
            ai_results_dir.mkdir(parents=True, exist_ok=True)

            # Конвертация PDF или подготовка изображения
            if file_path.suffix.lower() == '.pdf':
                from pdf2image import convert_from_path
                images = convert_from_path(file_path)
                for i, image_pil in enumerate(images):
                    temp_file = temp_dir / f"temp_{uuid.uuid4().hex[:8]}-{i+1}.jpg"
                    temp_files.append(temp_file)
                    image_pil.save(str(temp_file), 'JPEG', quality=95)
                    image_pil.close()
            else:
                temp_file = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.jpg"
                temp_files.append(temp_file)
                with Image.open(file_path) as img:
                    img.save(str(temp_file), 'JPEG', quality=95)

            # Обработка изображения
            if not temp_files:
                raise ValueError("Не удалось создать временные файлы")

            temp_file = temp_files[-1]
            with Image.open(temp_file) as pil_image:
                max_size = 3000
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {temp_file}")

            # Исправляем ориентацию изображения
            image = self._fix_image_orientation(image)

            # Распознавание текста и получение уверенности
            text_data, average_confidence = self._recognize_text(image)

            # Анализ текста и определение регионов для маскирования
            for i, word_data in enumerate(text_data):
                word_text = word_data['text'].strip()
                if not word_text:
                    continue

                # Логируем слова с низкой уверенностью для отладки проблем с OCR
                confidence = word_data.get('conf', 0)
                if confidence < 60:  # Порог уверенности (например, 60%)
                    logger.warning(f"Низкая уверенность распознавания для слова '{word_text}' (уверенность: {confidence}, координаты: {word_data['left']},{word_data['top']},{word_data['width']},{word_data['height']})")

                # Определение новой строки
                is_new_line = False
                if current_line_words_data:
                    line_break_threshold = word_data['height'] * 0.7
                    if abs(word_data['top'] - current_line_words_data[0]['top']) > line_break_threshold:
                        is_new_line = True
                else:
                    is_new_line = True

                # Обработка предыдущей строки
                if is_new_line and current_line_words_data:
                    if mask_context_active_for_line:
                        city_index = None
                        for idx, wd in enumerate(current_line_words_data):
                            clean_wd_text = re.sub(r'[.,!?;:]+$', '', wd['text'].strip()).strip()
                            if (wd['text'] and wd['text'][0].isupper() and
                                (clean_wd_text in self.cities_to_mask or
                                 clean_wd_text.lower() in self.cities_to_mask)):
                                city_index = idx
                                break

                        if city_index is not None:
                            min_left = current_line_words_data[city_index]['left']
                            max_right = max(r['left'] + r['width'] for r in current_line_words_data[city_index:])
                            min_top = min(r['top'] for r in current_line_words_data[city_index:])
                            max_bottom = max(r['top'] + r['height'] for r in current_line_words_data[city_index:])

                            padding = 2
                            min_left = max(0, min_left - padding)
                            max_right = min(image.shape[1], max_right + padding)
                            min_top = max(0, min_top - padding)
                            max_bottom = min(image.shape[0], max_bottom + padding)

                            sensitive_regions.append({
                                'left': min_left,
                                'top': min_top,
                                'width': max_right - min_left,
                                'height': max_bottom - min_top,
                                'type': 'city_line',
                                'text': ' '.join(r['text'] for r in current_line_words_data[city_index:]),
                                'is_full_line': False
                            })

                    mask_context_active_for_line = False
                    current_line_words_data = []

                current_line_words_data.append(word_data)

                # Проверяем на персональные данные
                clean_word_text = re.sub(r'[.,!?;:]+$', '', word_text).strip()
                
                # Проверка на город
                if (word_text and word_text[0].isupper() and
                    (clean_word_text in self.cities_to_mask or
                     clean_word_text.lower() in self.cities_to_mask)):
                    mask_context_active_for_line = True
                    sensitive_regions.append({
                        'left': word_data['left'],
                        'top': word_data['top'],
                        'width': word_data['width'],
                        'height': word_data['height'],
                        'type': 'city',
                        'text': word_text
                    })
                
                # Проверка на персональные данные
                is_personal, data_type = self._is_numeric_personal_data(word_text)
                if is_personal:
                    sensitive_regions.append({
                        'left': word_data['left'],
                        'top': word_data['top'],
                        'width': word_data['width'],
                        'height': word_data['height'],
                        'type': data_type,
                        'text': word_text
                    })
                
                # Проверяем на фамилии и имена
                if not self._is_allowed_word(word_text, text_data, i):
                    sensitive_regions.append({
                        'left': word_data['left'],
                        'top': word_data['top'],
                        'width': word_data['width'],
                        'height': word_data['height'],
                        'type': 'personal_name',
                        'text': word_text
                    })

            # Группируем компоненты адресов, если они есть
            sensitive_context = [] # Необходимо собрать контекст для группировки
            for region in sensitive_regions:
                if region['type'] in ['city', 'street', 'house_number', 'building_number', 'flat_number', 'postal_code']:
                    sensitive_context.append(region)
            grouped_addresses = self._group_address_components(sensitive_context)
            sensitive_regions.extend(grouped_addresses)

            # Проверка наличия требуемых персональных данных (Фамилия, Имя, Полис ОМС, СНИЛС)
            required_data_types = ['personal_name', 'policy', 'snils'] # Исключил 'passport'
            found_required_data = any(region['type'] in required_data_types for region in sensitive_regions)

            if not found_required_data:
                logger.warning("Не найдено ни Фамилии, ни Имени, ни Полиса ОМС, ни СНИЛС. Отправляем в папку unreadable")
                output_file = Path(output_dir) / "unreadable" / f"{uuid.uuid4().hex}_unreadable.jpg"
                output_file.parent.mkdir(exist_ok=True)
                
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil_image.save(str(output_file), 'JPEG', quality=100)
                if not output_file.exists():
                    raise RuntimeError("Файл не был создан после сохранения")
                pil_image.close()
                
                # Создаем файлы с текстом "Не распознано" для всех нейросетей (исключая GPT)
                ai_dirs = {
                    'deepseek': 'ai-result/deepseek',
                    'gigachat': 'ai-result/gigachat'
                }
                
                for ai_name, ai_dir in ai_dirs.items():
                    analysis_file = Path(ai_dir) / f"{output_file.stem}.json"
                    analysis_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump({"status": "Не распознано", "reason": "Отсутствуют требуемые персональные данные"}, f, ensure_ascii=False, indent=2)
                    logger.info(f"Создан файл анализа для {ai_name}: {analysis_file}")
                
                return str(output_file), {"status": "Не распознано", "reason": "Отсутствуют требуемые персональные данные"}

            # Маскирование найденных регионов (выполняется только если found_required_data == True)
            masked_regions_count = 0
            for region in sensitive_regions:
                try:
                    left = int(region['left'])
                    top = int(region['top'])
                    width = int(region['width'])
                    height = int(region['height'])
                    
                    # Проверяем корректность координат
                    if (left >= 0 and top >= 0 and width > 0 and height > 0 and
                        left + width <= image.shape[1] and top + height <= image.shape[0]):
                        # Добавляем небольшой отступ
                        padding = 2
                        left = max(0, left - padding)
                        top = max(0, top - padding)
                        width = min(image.shape[1] - left, width + 2 * padding)
                        height = min(image.shape[0] - top, height + 2 * padding)
                        
                        # Маскируем регион
                        image[top:top + height, left:left + width] = 0
                        masked_regions_count += 1
                except Exception as e:
                    logger.warning(f"Ошибка при маскировании региона {region}: {str(e)}")
                    continue
            
            output_file = Path(output_dir) / f"{uuid.uuid4().hex}_depersonalized.jpg"
            
            # Сохраняем через PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_image.save(str(output_file), 'JPEG', quality=100)
            if not output_file.exists():
                raise RuntimeError("Файл не был создан после сохранения")
            pil_image.close()
            
            # Отправляем на анализ в DeepSeek
            logger.info(f"Отправка файла {output_file.name} на анализ в DeepSeek...")
            deepseek_analysis_result = await self.deepseek_client.analyze_medical_report(str(output_file))
            
            # Отправляем на анализ в GigaChat
            logger.info(f"Отправка файла {output_file.name} на анализ в GigaChat...")
            gigachat_analysis_result = await self.gigachat_client.analyze_medical_report(str(output_file))

            # Сохраняем результаты анализа (исключая GPT)
            final_analysis_result = {}

            if deepseek_analysis_result:
                deepseek_analysis_file = Path('ai-result/deepseek').absolute() / f"{output_file.stem}.json"
                deepseek_analysis_file.parent.mkdir(parents=True, exist_ok=True)
                with open(deepseek_analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(deepseek_analysis_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Анализ DeepSeek сохранен в {deepseek_analysis_file}")
                final_analysis_result['deepseek'] = deepseek_analysis_result
            else:
                logger.error(f"Не удалось получить анализ от DeepSeek для файла {output_file.name}")

            if gigachat_analysis_result:
                gigachat_analysis_file = Path('ai-result/gigachat').absolute() / f"{output_file.stem}.json"
                gigachat_analysis_file.parent.mkdir(parents=True, exist_ok=True)
                with open(gigachat_analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(gigachat_analysis_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Анализ GigaChat сохранен в {gigachat_analysis_file}")
                final_analysis_result['gigachat'] = gigachat_analysis_result
            else:
                logger.error(f"Не удалось получить анализ от GigaChat для файла {output_file.name}")

            extracted_data = self._extract_data(text_data)
            found_name = extracted_data.get('found_names', [])
            found_surname = extracted_data.get('found_surnames', [])
            found_policy = any(r['type'] == 'Полис ОМС' for r in extracted_data['sensitive_regions'])
            found_snils = any(r['type'] == 'СНИЛС' for r in extracted_data['sensitive_regions'])
            if not (found_name or found_surname or found_policy or found_snils):
                # В unreadable и return
                unreadable_dir = Path(output_dir) / "unreadable"
                unreadable_dir.mkdir(exist_ok=True, parents=True)
                output_file = unreadable_dir / f"{uuid.uuid4().hex}_unreadable.jpg"
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil_image.save(str(output_file), 'JPEG', quality=100)
                pil_image.close()
                ai_dirs = {
                    'deepseek': 'ai-result/deepseek',
                    'gigachat': 'ai-result/gigachat'
                }
                for ai_name, ai_dir in ai_dirs.items():
                    analysis_file = Path(ai_dir) / f"{output_file.stem}.json"
                    analysis_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "status": "Не распознано",
                            "reason": "Не найдено ни одного из требуемых персональных данных (Имя, Фамилия, ОМС, СНИЛС)"
                        }, f, ensure_ascii=False, indent=2)
                    logger.info(f"Создан файл анализа для {ai_name}: {analysis_file}")
                return str(output_file), {
                    "status": "Не распознано",
                    "reason": "Не найдено ни одного из требуемых персональных данных (Имя, Фамилия, ОМС, СНИЛС)"
                }

            return str(output_file), final_analysis_result

        except Exception as e:
            logger.error(f"Ошибка при обработке документа {file_path}: {str(e)}")
            if output_file and output_file.exists():
                try:
                    output_file.unlink()
                except:
                    pass
            return None, {}
        finally:
            # Очистка временных файлов
            self._cleanup_temp_files()
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except:
                    pass

    async def analyze_document(self, file_path: str) -> Optional[str]:
        """
        Анализирует деперсонализированный документ через DeepSeek
        
        Args:
            file_path: путь к деперсонализированному документу
            
        Returns:
            Optional[str]: результат анализа или None в случае ошибки
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Файл для анализа не найден: {file_path}")
                return None

            logger.info(
                f"Отправка документа {file_path} на анализ в DeepSeek...")
            analysis_result = await self.deepseek_client.analyze_medical_report(str(file_path))

            if analysis_result:
                # Выводим результат в консоль
                logger.info("Результат анализа DeepSeek:")
                logger.info("-" * 80)
                logger.info(analysis_result)
                logger.info("-" * 80)

                # Сохраняем результат
                self.deepseek_client.save_analysis_result(
                    file_path, analysis_result)
                return analysis_result
            else:
                logger.error("Не удалось получить анализ от DeepSeek")
                return None

        except Exception as e:
            logger.error(f"Ошибка при анализе документа {file_path}: {str(e)}")
            return None

    async def process_directory(self,
                                input_dir: str,
                                clinic_name: str,
                                output_dir: str) -> List[Tuple[str,
                                                               Dict]]:
        """
        Обрабатывает все документы в директории последовательно
        """
        results = []
        processed_files = []
        failed_files = []

        try:
            # Проверяем входную директорию
            input_path = Path(input_dir).absolute()
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Входная директория не найдена: {input_dir}")
            if not input_path.is_dir():
                raise NotADirectoryError(
                    f"Указанный путь не является директорией: {input_dir}")

            # Проверяем выходную директорию
            output_path = Path(output_dir).absolute()
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Выходная директория {output_path} создана или уже существует")
            except Exception as e:
                logger.error(
                    f"Не удалось создать выходную директорию {output_path}: {e}")
                raise

            # Получаем список всех файлов изображений и PDF
            files = [
                f for f in input_path.glob('*') if f.is_file() and f.suffix.lower() in (
                    '.png',
                    '.jpg',
                    '.jpeg',
                    '.tiff',
                    '.bmp',
                    '.pdf')]

            if not files:
                logger.warning(
                    f"В директории {input_dir} не найдено подходящих файлов")
                return []

            logger.info(f"Найдено {len(files)} файлов для обработки")

            # Сначала обрабатываем все файлы (деперсонализация)
            for file_path in files:
                try:
                    logger.info(f"Обработка файла: {file_path.name}")
                    result = await self.process_document(str(file_path), clinic_name, output_dir)
                    results.append(result)
                    # Сохраняем путь к обработанному файлу
                    processed_files.append(result[0])
                    logger.info(f"Файл {file_path.name} успешно обработан")
                except Exception as e:
                    logger.error(
                        f"Ошибка при обработке файла {file_path.name}: {str(e)}")
                    failed_files.append((str(file_path), str(e)))
                    continue

            # Выводим итоговую статистику
            logger.info(
                f"Обработка завершена. Успешно: {len(processed_files)}, Ошибок: {len(failed_files)}")
            if failed_files:
                logger.warning("Список файлов с ошибками:")
                for file_path, error in failed_files:
                    logger.warning(f"- {file_path}: {error}")

            return results

        except Exception as e:
            logger.error(
                f"Критическая ошибка при обработке директории {input_dir}: {str(e)}")
            raise

    def _recognize_text(self, image: np.ndarray) -> List[Dict]:
        """
        Распознавание текста на изображении с поддержкой русского и английского языков
        """
        try:
            logger.info("Начало распознавания текста...")
            data = pytesseract.image_to_data(
                image, 
                lang='rus+eng',
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )

            # Вычисляем среднюю уверенность
            confidences = [float(c) for c in data['conf'] if float(c) != -1]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0
            logger.info(f"Средняя уверенность распознавания: {average_confidence:.2f}")

            # Выводим статистику распознавания
            total_words = len([t for t in data['text'] if t.strip()])
            logger.info(f"Всего распознано слов: {total_words}")

            if total_words < 5:
                logger.info(
                    "Мало слов распознано на русском, пробуем английский...")
                eng_data = pytesseract.image_to_data(
                    image,
                    lang='eng',
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                eng_words = len([t for t in eng_data['text'] if t.strip()])
                logger.info(f"Распознано слов на английском: {eng_words}")

                if eng_words > total_words:
                    logger.info(
                        "Используем результаты английского распознавания")
                    data = eng_data
                    # Пересчитываем уверенность для английского языка
                    confidences = [float(c) for c in data['conf'] if float(c) != -1]
                    average_confidence = sum(confidences) / len(confidences) if confidences else 0
                    logger.info(f"Средняя уверенность после переключения на английский: {average_confidence:.2f}")

            words = []

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    confidence = float(data['conf'][i])
                    # Порог уверенности для отдельных слов может быть ниже, чем для всего документа
                    # Удаляем фильтрацию по уверенности, чтобы получить все слова для анализа
                    # if confidence < 10: 
                    #     logger.debug(
                    #         f"Пропуск слова '{text}' из-за низкой уверенности: {confidence}")
                    #     continue

                    word = {
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': confidence,
                        'lang': 'eng' if all(
                            c.isascii() for c in text) else 'rus',
                        'position': i # Добавляем позицию слова в списке
                    }
                    words.append(word)
                    logger.debug(
                        f"Распознано слово: '{text}' (уверенность: {confidence}, язык: {word['lang']})")

            logger.info(f"Итоговое количество распознанных слов: {len(words)}")

            return words, average_confidence # Возвращаем только слова и среднюю уверенность
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста: {str(e)}")
            raise

    def _extract_data(self, text_data: List[Dict]) -> Dict:
        """
        Извлекает данные из распознанного текста
        """
        sensitive_regions = []
        found_surnames = []
        found_names = []  # <--- добавлено
        medical_data = []
        sensitive_context = []
        try:
            for i, word in enumerate(text_data):
                try:
                    word_text = word['text'].strip()
                    if not word_text:
                        continue

                    # Улучшенная проверка на валидность слова для отсеивания шума OCR
                    alpha_chars = sum(1 for c in word_text if c.isalpha())
                    total_chars = len(word_text)

                    if total_chars == 0: 
                        continue

                    # Если слово имеет менее 2 буквенных символов и очень короткое (менее 4 символов всего),
                    # или если доля буквенных символов меньше 30% для более длинных слов - считаем шумом.
                    if (alpha_chars < 2 and total_chars < 4) or (alpha_chars / total_chars < 0.3):
                        logger.debug(f"Пропуск слова '{word_text}' из-за низкой доли буквенных символов: {alpha_chars}/{total_chars}")
                        continue
                    
                    # Identify name/surname/patronymic candidates for the 'unreadable' check
                    word_lower = word_text.lower()
                    translit_word_lower = transliterate(word_lower)

                    is_surname_candidate = (word_lower in self.surnames or translit_word_lower in self.translit_surnames)
                    is_name_candidate = (word_lower in self.names or translit_word_lower in self.translit_names or \
                                         word_lower in self.patronymics or translit_word_lower in self.translit_patronymics)

                    if is_surname_candidate:
                        found_surnames.append(word_text)
                        logger.debug(f"Обнаружена фамилия-кандидат для проверки 'нечитаемости': {word_text}")
                    if is_name_candidate:
                        found_names.append(word_text)
                        logger.debug(f"Обнаружено имя-кандидат для проверки 'нечитаемости': {word_text}")

                    # Проверяем на почтовый индекс
                    is_personal_data, data_type = self._is_numeric_personal_data(word_text)
                    if data_type == 'Почтовый индекс':
                        sensitive_regions.append({
                            'text': word_text,
                            'confidence': word.get('conf', 0),
                            'left': word.get('left', 0),
                            'top': word.get('top', 0),
                            'width': word.get('width', 0),
                            'height': word.get('height', 0),
                            'type': data_type,
                            'position': i # Добавляем позицию
                        })
                        logger.info(f"Найден почтовый индекс для маскирования: {word_text}")
                        continue # Продолжаем, чтобы остальные части адреса тоже были проверены
                    
                    # Проверяем на город (в любом регистре)
                    if word_text in self.cities_to_mask:
                        sensitive_context.append({
                            'text': word_text,
                            'confidence': word.get('conf', 0),
                            'left': word.get('left', 0),
                            'top': word.get('top', 0),
                            'width': word.get('width', 0),
                            'height': word.get('height', 0),
                            'type': 'city',
                            'position': i
                        })
                        logger.info(f"Найден город для маскирования: {word_text}")
                        # Продолжаем, чтобы другие части адреса могли быть добавлены
                        
                    # Проверяем на компоненты адреса (улица, дом, корпус, квартира)
                    address_component_patterns = {
                        'street': r'(?:ул\.?|улица|пер\.?|переулок|пр\.?|проспект|ш\.?|шоссе|наб\.?|набережная|пл\.?|площадь)\s+[А-Яа-яЁё\d\s\.\-]+',
                        'house_number': r'(?:д\.?|дом)\s*\d+(?:[а-яА-ЯёЁ])?(?:\s*к\.?\s*\d+)?(?::\s*\d+)?',
                        'building_number': r'(?:корп\.?|корпус)\s*\d+',
                        'flat_number': r'(?:кв\.?|квартира)\s*\d+'
                    }
                    
                    found_address_component = False
                    for addr_type, pattern in address_component_patterns.items():
                        context_start = max(0, i - 5) # 5 слов назад
                        context_end = min(len(text_data), i + 5) # 5 слов вперед
                        context_text = " ".join([w['text'] for w in text_data[context_start:context_end]])
                        
                        if re.search(pattern, context_text, re.IGNORECASE):
                            sensitive_context.append({
                                'text': word_text,
                                'confidence': word.get('conf', 0),
                                'left': word.get('left', 0),
                                'top': word.get('top', 0),
                                'width': word.get('width', 0),
                                'height': word.get('height', 0),
                                'type': addr_type,
                                'position': i
                            })
                            found_address_component = True
                            logger.info(f"Найден компонент адреса: {word_text} ({addr_type})")
                            break
                    
                    if found_address_component: # Если это компонент адреса, не проверяем другие типы
                        continue
                    
                    # Проверяем числовые данные (паспорт, СНИЛС, телефон, мед. карта, дата рождения, email)
                    if is_personal_data:
                        sensitive_regions.append({
                            'text': word_text,
                            'confidence': word.get('conf', 0),
                            'left': word.get('left', 0),
                            'top': word.get('top', 0),
                            'width': word.get('width', 0),
                            'height': word.get('height', 0),
                            'type': data_type,
                            'position': i # Добавляем позицию
                        })
                        logger.info(f"Найдены числовые персональные данные: {word_text} ({data_type})")
                        continue

                    # Теперь определяем, нужно ли слово маскировать (используя логику _is_allowed_word)
                    if not self._is_allowed_word(word_text, text_data, i): # _is_allowed_word возвращает False, если слово должно быть замаскировано
                        sensitive_regions.append({
                            'text': word_text,
                            'confidence': word.get('conf', 0),
                            'left': word.get('left', 0),
                            'top': word.get('top', 0),
                            'width': word.get('width', 0),
                            'height': word.get('height', 0),
                            'type': 'personal_name_or_surname', # Общий тип для маскирования имен/фамилий
                            'position': i
                        })
                        logger.info(f"Слово '{word_text}' помечено для маскирования.")

                    # Проверяем на медицинские термины
                    for pattern_name, pattern in self.medical_patterns.items():
                        matches = re.finditer(pattern, word_text)
                        for match in matches:
                            value = match.group(1) if match.groups() else match.group(0)
                            if value:
                                medical_data.append({
                                    'type': pattern_name,
                                    'value': value,
                                    'text': word_text,
                                    'confidence': word.get('conf', 0),
                                    'left': word.get('left', 0),
                                    'top': word.get('top', 0),
                                    'width': word.get('width', 0),
                                    'height': word.get('height', 0)
                                })
                except Exception as e:
                    logger.error(f"Ошибка при обработке слова '{word.get('text', '')}': {str(e)}")
                    continue
            
            # После обработки всех слов, группируем компоненты адресов
            grouped_addresses = self._group_address_components(sensitive_context)
            sensitive_regions.extend(grouped_addresses)
            
            # Проверяем медицинский контекст
            self._verify_medical_context(text_data, {'medical_data': medical_data})

            # Проверяем согласованность диагнозов
            self._verify_diagnoses({'medical_data': medical_data})
        
            return {
                'sensitive_regions': sensitive_regions,
                'medical_data': medical_data,
                'found_surnames': found_surnames,
                'found_names': found_names  # <--- добавлено
            }

        except Exception as e:
            logger.error(f"Ошибка при извлечении данных: {str(e)}")
            return {
                'sensitive_regions': [],
                'medical_data': [],
                'found_surnames': [],
                'found_names': []  # <--- добавлено
            }

    def _mask_sensitive_data(self, image: np.ndarray, sensitive_regions: List[Dict]) -> np.ndarray:
        """
        Маскирует чувствительные данные на изображении черными прямоугольниками
        
        Args:
            image: исходное изображение
            sensitive_regions: список регионов для маскирования
            
        Returns:
            np.ndarray: изображение с замаскированными регионами
        """
        try:
            logger.info("Начало процесса маскирования...")
            logger.info(f"Размер изображения: {image.shape}")
            logger.info(f"Количество регионов для маскирования: {len(sensitive_regions)}")

            # Создаем копию изображения для маскирования
            masked_image = image.copy()

            if not sensitive_regions:
                logger.warning("Нет регионов для маскирования")
                return masked_image

            # Маскируем все регионы
            for i, region in enumerate(sensitive_regions):
                try:
                    left = int(region.get('left', 0))
                    top = int(region.get('top', 0))
                    width = int(region.get('width', 0))
                    height = int(region.get('height', 0))

                    # Проверяем корректность координат
                    if (left < 0 or top < 0 or width <= 0 or height <= 0 or
                        left + width > masked_image.shape[1] or
                        top + height > masked_image.shape[0]):
                        logger.warning(f"Некорректные координаты для региона {i+1}: "
                                     f"left={left}, top={top}, width={width}, height={height}")
                        continue

                    # Добавляем небольшой отступ
                    padding = 2
                    left = max(0, left - padding)
                    top = max(0, top - padding)
                    width = min(masked_image.shape[1] - left, width + 2 * padding)
                    height = min(masked_image.shape[0] - top, height + 2 * padding)

                    # Маскируем регион
                    masked_image[top:top + height, left:left + width] = 0

                    # Проверяем, что маскирование прошло успешно
                    if np.all(masked_image[top:top + height, left:left + width] == 0):
                        logger.info(f"Успешно замаскирован регион {i+1}: "
                                  f"left={left}, top={top}, width={width}, height={height}, "
                                  f"текст: '{region.get('text', '')}', "
                                  f"тип: '{region.get('type', 'N/A')}'")
                    else:
                        logger.warning(f"Неполное маскирование региона {i+1}: '{region.get('text', '')}'")

                except Exception as e:
                    logger.error(f"Ошибка при маскировании региона {i+1}: {str(e)}")
                    continue

            return masked_image

        except Exception as e:
            logger.error(f"Критическая ошибка при маскировании: {str(e)}")
            return image

    def _is_allowed_word(self, word: str, text_data: List[Dict], current_index: int) -> bool:
        """
        Проверяет, является ли слово разрешенным (не подлежащим маскированию)
        
        Args:
            word: проверяемое слово
            text_data: список словарей с распознанным текстом
            current_index: индекс текущего слова в text_data
            
        Returns:
            bool: True если слово разрешено, False если подлежит маскированию
        """
        word_lower = word.lower()

        # 1. Проверка на ключевые слова адресов (должны быть замаскированы)
        if word_lower in self.address_keywords:
            return False

        # 2. Базовые проверки длины и содержания
        if not word or not word.strip():
            return True
        letters_only = ''.join(c for c in word if c.isalpha())
        if not letters_only:
            return True
        if len(letters_only) < 3: # Слова короче 3 букв пропускаем (кроме инициалов, которые обрабатываются позже)
            return True
        if not word[0].isupper(): # Если слово не начинается с заглавной буквы, обычно не является именем собственным
            return True

        # 3. Проверка на наличие "ФИО" (явное указание на персональные данные)
        if 'фио' in word_lower:
            return False
        
        # Проверяем контекст на наличие "ФИО"
        # Получаем ближайший контекст (2 слова слева и справа)
        start_idx_context = max(0, current_index - 2)
        end_idx_context = min(len(text_data), current_index + 3)
        context_words = [w['text'].lower() for w in text_data[start_idx_context:end_idx_context]]
        context_text = ' '.join(context_words)
        if 'фио' in context_text:
            return False

        # 4. Проверка на UUID и длинные буквенно-цифровые коды (сертификаты)
        if re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', word_lower):
            return False
        # Сертификат: пример: 74 89 C7 97 45 6A DB 47 2A 02 16 BC A2 4F 22 F0 (пробелы удаляются для проверки)
        if re.match(r'^([0-9a-fA-F]{2}\\s){15}[0-9a-fA-F]{2}$|^[0-9a-fA-F]{32,48}$', word.lower().replace(' ', '')):
            return False

        # 5. Проверка на медицинские термины и разрешенные слова (не маскируются)
        if word_lower in self.medical_terms or word_lower in self.allowed_words:
            return True

        # 6. Проверка на числовые персональные данные (паспорт, СНИЛС, телефон и т.д.)
        is_personal_numeric, _ = self._is_numeric_personal_data(word)
        if is_personal_numeric:
            return False

        # 7. Проверка на фамилии, отчества или имена (самая сложная часть, требует контекста)
        is_personal_name_component = (
            word_lower in self.surnames or 
            word_lower in self.patronymics or
            word_lower in self.names or
            transliterate(word_lower) in self.surnames or
            transliterate(word_lower) in self.patronymics or
            transliterate(word_lower) in self.names
        )

        if is_personal_name_component:
            # Проверяем, является ли предыдущее слово медицинским званием/должностью
            is_preceded_by_medical_title = False
            if current_index > 0:
                prev_word_text = text_data[current_index - 1]['text'].lower()
                medical_titles = [
                    'врач', 'доктор', 'профессор', 'академик', 'заведующий',
                    'главный', 'старший', 'младший', 'ординатор', 'интерн',
                    'биолог'
                ]
                if prev_word_text in medical_titles:
                    is_preceded_by_medical_title = True

            # Проверяем на шаблоны медицинских документов, указывающие на ФИО пациента
            # (например, "Пациент ФИО", "Фамилия:")
            patient_context_patterns = [
                r'(?:^|\s)пациент[а-я]*\s+',
                r'(?:^|\s)больной\s+',
                r'(?:^|\s)ф\.и\.о\.\s*',
                r'(?:^|\s)фамилия[:\s]+',
                r'(?:^|\s)имя[:\s]+',
                r'(?:^|\s)отчество[:\s]+'
            ]
            is_in_patient_context = False
            # Проверяем не только слово, но и ближайший контекст
            for pattern in patient_context_patterns:
                if re.search(pattern, context_text, re.IGNORECASE):
                    is_in_patient_context = True
                    break

            # Логика маскирования для компонентов личных имен:
            # Маскируем, если:
            #   - НЕ предшествует медицинскому званию (т.е. это не ФИО врача/биолога), ИЛИ
            #   - Находится в явном контексте пациента (например, "Пациент Федоренко")
            if (not is_preceded_by_medical_title) or is_in_patient_context:
                return False # Маскируем
            else:
                return True # Разрешаем (это ФИО мед. персонала, не в контексте пациента)

        # 8. Проверка на инициалы (пропускаем)
        if len(word) <= 2 and word[0].isupper():
            # Если это последнее слово или следующее слово тоже инициал - пропускаем
            if (current_index == len(text_data) - 1 or
                (current_index + 1 < len(text_data) and
                 (text_data[current_index + 1]['text'].endswith('.') or
                  (len(text_data[current_index + 1]['text']) <= 2 and
                   text_data[current_index + 1]['text'][0].isupper())))):
                    return True

        # 9. По умолчанию разрешаем слово
        return True

    def _group_address_components(self, sensitive_context: List[Dict]) -> List[Dict]:
        """
        Группирует компоненты адреса (город, улица, номер дома и т.д.) в полные адреса.
        
        Args:
            sensitive_context: список словарей с распознанными компонентами адреса.
            
        Returns:
            List[Dict]: список словарей, представляющих сгруппированные адреса для маскирования.
        """
        if not sensitive_context:
            return []

        # Сортируем компоненты по их позиции в тексте
        sensitive_context.sort(key=lambda x: x['position'])

        grouped_addresses = []
        current_address_group = []

        for i, component in enumerate(sensitive_context):
            if not current_address_group:
                current_address_group.append(component)
            else:
                # Проверяем, находится ли текущий компонент рядом с предыдущим
                # Если слова находятся на одной строке и не слишком далеко друг от друга,
                # считаем их частью одного адреса.
                prev_component = current_address_group[-1]
                
                # Разница по вертикали должна быть небольшой (на одной строке)
                is_same_line = abs(component['top'] - prev_component['top']) < (prev_component['height'] * 0.7)
                
                # Разница по горизонтали не должна быть слишком большой (слова рядом)
                is_close_horizontally = (component['left'] - (prev_component['left'] + prev_component['width'])) < (prev_component['width'] * 2) # Максимум 2 ширины предыдущего слова между ними
                
                # Дополнительная проверка на разрыв строк - если есть большая вертикальная разница,
                # это, скорее всего, новый адрес или другой блок текста
                if (is_same_line and
                    is_close_horizontally and
                    component['position'] == prev_component['position'] + 1): # Проверяем, что слова идут последовательно
                    current_address_group.append(component)
                else:
                    # Если группа прервалась, добавляем текущую группу в результат
                    if current_address_group:
                        # Создаем единый регион для маскирования
                        min_left = min(r['left'] for r in current_address_group)
                        min_top = min(r['top'] for r in current_address_group)
                        max_right = max(r['left'] + r['width'] for r in current_address_group)
                        max_bottom = max(r['top'] + r['height'] for r in current_address_group)
                        
                        full_address_text = " ".join(r['text'] for r in current_address_group)
                        
                        grouped_addresses.append({
                            'text': full_address_text,
                            'left': min_left,
                            'top': min_top,
                            'width': max_right - min_left,
                            'height': max_bottom - min_top,
                            'type': 'full_address'
                        })
                    current_address_group = [component]

        # Добавляем последнюю сгруппированную группу, если она есть
        if current_address_group:
            min_left = min(r['left'] for r in current_address_group)
            min_top = min(r['top'] for r in current_address_group)
            max_right = max(r['left'] + r['width'] for r in current_address_group)
            max_bottom = max(r['top'] + r['height'] for r in current_address_group)
            
            full_address_text = " ".join(r['text'] for r in current_address_group)
            
            grouped_addresses.append({
                'text': full_address_text,
                'left': min_left,
                'top': min_top,
                'width': max_right - min_left,
                'height': max_bottom - min_top,
                'type': 'full_address'
            })
            
        return grouped_addresses

    def _extract_patient_info(self, text_data: List[Dict]) -> Optional[Dict]:
        """
        Извлекает информацию о пациенте из распознанного текста
        
        Args:
            text_data: список распознанных слов с координатами
            
        Returns:
            Optional[Dict]: словарь с информацией о пациенте или None, если информация не найдена
        """
        logger.info("Поиск информации о пациенте в тексте...")
        
        # Паттерны для поиска ФИО
        name_patterns = [
            r'(?:Пациент|ФИО|Ф\.И\.О\.|Фамилия)[:\s]+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})',
            r'(?:Фамилия)[:\s]+([А-ЯЁ][а-яё]+)',
            r'(?:Имя)[:\s]+([А-ЯЁ][а-яё]+)']
        
        # Собираем весь текст в одну строку для поиска
        full_text = ' '.join(word['text'] for word in text_data)
        
        # Ищем фамилию и имя
        surname = None
        name = None
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                text = match.group(1).strip()
                words = text.split()
                
                if len(words) >= 2:  # Если найдено полное ФИО
                    surname = words[0]
                    name = words[1]
                    break
                elif len(words) == 1:  # Если найдена только фамилия или имя
                    if 'фамилия' in match.group(0).lower():
                        surname = words[0]
                    elif 'имя' in match.group(0).lower():
                        name = words[0]
                
                if surname and name:
                    break
            
            if surname and name:
                break
        
        # Возвращаем найденную информацию без проверки по self.surnames
        if surname or name:
            logger.info(f"Найдена информация о пациенте: Фамилия: {surname}, Имя: {name}")
            return {
                'surname': surname,
                'name': name
            }
        
        logger.warning("Не удалось найти достоверную информацию о пациенте")
        return None

    def _is_numeric_personal_data(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет, является ли текст числовыми персональными данными
        
        Args:
            text: проверяемый текст
            
        Returns:
            Tuple[bool, str]: (является ли персональными данными, тип данных)
        """
        # Очищаем текст от пробелов и дефисов для проверки (для некоторых паттернов)
        clean_text = re.sub(r'[\s\-]', '', text)

        # Проверяем каждый паттерн из self.numeric_patterns
        for pattern_name, pattern_info in self.numeric_patterns.items():
            pattern = pattern_info['pattern']
            description = pattern_info['description']

            # Для email и birth_date используем исходный текст, для остальных - очищенный
            text_to_match = text if pattern_name in ['email', 'birth_date'] else clean_text

            if re.match(pattern, text_to_match, re.IGNORECASE): # Добавил re.IGNORECASE
                return True, description

        return False, ""

    def _verify_medical_context(
            self,
            text_data: List[Dict],
            extracted_data: Dict) -> None:
        """
        Проверка контекста медицинских терминов

        Args:
            text_data: список распознанных слов
            extracted_data: извлеченные данные
        """
        text = ' '.join(word['text'] for word in text_data)

        for medical_item in extracted_data.get('medical_data', []):
            term = medical_item['value'].lower()
            category = None

            # Определяем категорию термина
            for cat, terms in self.medical_categories.items():
                if term in terms:
                    category = cat
                    break

            if category:
                # Проверяем контекст в зависимости от категории
                if category == 'diagnosis':
                    if not self._verify_diagnosis_context(text, term):
                        logger.warning(
                            f"Диагноз '{term}' использован вне медицинского контекста")
                elif category == 'procedure':
                    if not self._verify_procedure_context(text, term):
                        logger.warning(
                            f"Процедура '{term}' использована вне медицинского контекста")
                elif category == 'drugs':
                    if not self._verify_medication_context(text, term):
                        logger.warning(
                            f"Препарат '{term}' использован вне медицинского контекста")

    def _verify_diagnosis_context(self, text: str, diagnosis: str) -> bool:
        """
        Проверка контекста диагноза

        Args:
            text: полный текст
            diagnosis: диагноз для проверки

        Returns:
            bool: является ли контекст медицинским
        """
        # Ищем контекст вокруг диагноза
        start = text.lower().find(diagnosis.lower())
        if start == -1:
            return False

        context_start = max(0, start - 200)
        context_end = min(len(text), start + len(diagnosis) + 200)
        context = text[context_start:context_end].lower()

        # Проверяем наличие медицинских терминов в контексте
        required_terms = self.medical_protocols['diagnosis']['required_context']
        return any(term in context for term in required_terms)

    def _verify_procedure_context(self, text: str, procedure: str) -> bool:
        """
        Проверка контекста процедуры

        Args:
            text: полный текст
            procedure: процедура для проверки

        Returns:
            bool: является ли контекст медицинским
        """
        # Ищем контекст вокруг процедуры
        start = text.lower().find(procedure.lower())
        if start == -1:
            return False

        context_start = max(0, start - 200)
        context_end = min(len(text), start + len(procedure) + 200)
        context = text[context_start:context_end].lower()

        # Проверяем наличие медицинских терминов в контексте
        required_terms = self.medical_protocols['procedure']['required_context']
        return any(term in context for term in required_terms)

    def _verify_medication_context(self, text: str, medication: str) -> bool:
        """
        Проверка контекста лекарственного препарата

        Args:
            text: полный текст
            medication: препарат для проверки

        Returns:
            bool: является ли контекст медицинским
        """
        # Ищем контекст вокруг препарата
        start = text.lower().find(medication.lower())
        if start == -1:
            return False

        context_start = max(0, start - 200)
        context_end = min(len(text), start + len(medication) + 200)
        context = text[context_start:context_end].lower()

        # Проверяем наличие медицинских терминов в контексте
        medication_terms = {
            'дозировка',
            'прием',
            'назначение',
            'курс',
            'лечение',
            'терапия'}
        return any(term in context for term in medication_terms)

    def _verify_diagnoses(self, extracted_data: Dict) -> None:
        """
        Проверка согласованности диагнозов

        Args:
            extracted_data: извлеченные данные
        """
        diagnoses = []
        for medical_item in extracted_data.get('medical_data', []):
            if medical_item.get('type') == 'diagnosis':
                diagnoses.append(medical_item['value'])

        if len(diagnoses) > 1:
            # Проверяем согласованность диагнозов
            for i, diagnosis1 in enumerate(diagnoses):
                for diagnosis2 in diagnoses[i + 1:]:
                    if not self._are_diagnoses_compatible(
                            diagnosis1, diagnosis2):
                        logger.warning(
                            f"Возможно несовместимые диагнозы: '{diagnosis1}' и '{diagnosis2}'"
                        )

    def _are_diagnoses_compatible(
            self,
            diagnosis1: str,
            diagnosis2: str) -> bool:
        """
        Проверка совместимости двух диагнозов

        Args:
            diagnosis1: первый диагноз
            diagnosis2: второй диагноз

        Returns:
            bool: совместимы ли диагнозы
        """
        # Здесь можно добавить более сложную логику проверки совместимости диагнозов
        # Например, проверку по медицинским классификациям
        return True  # Пока всегда возвращаем True

    def _is_personal_data(self, value: str, clean_value: str,
                          pattern_name: str) -> Tuple[bool, str]:
        """
        Проверяет, является ли значение персональными данными

        Args:
            value: исходное значение
            clean_value: очищенное значение (без пробелов и дефисов)
            pattern_name: тип паттерна

        Returns:
            Tuple[bool, str]: (является ли персональными данными, описание типа данных)
        """
        # Проверяем числовые персональные данные
        if pattern_name in [
            'snils',
            'policy',
            'passport',
            'phone',
                'med_card']:
            # Проверяем длину очищенного значения
            if len(clean_value) in self.numeric_lengths:
                # Проверяем каждый паттерн
                for data_type, pattern_info in self.numeric_patterns.items():
                    if re.match(pattern_info['pattern'], clean_value):
                        return True, pattern_info['description']

        # Проверяем другие типы персональных данных
        personal_patterns = {
            'birth_date': {
                'pattern': r'^\d{2}\.\d{2}\.\d{4}$',
                'description': 'Дата рождения'
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'description': 'Email'
            },
            'gender': {
                'pattern': r'^(?:м|ж|муж|жен|мужской|женский)$',
                'description': 'Пол'
            }
        }

        if pattern_name in personal_patterns:
            if re.match(
                    personal_patterns[pattern_name]['pattern'],
                    value.lower()):
                return True, personal_patterns[pattern_name]['description']

        return False, ""
