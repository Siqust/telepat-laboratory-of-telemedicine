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
    result = subprocess.run([str(PDFTOPPM), "-v"], capture_output=True, text=True)
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

    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Инициализация процессора документов
        """
        self.data_manager = data_manager
        self.medical_terms = set()
        self.deepseek_client = DeepSeekClient()
        
        # Загружаем список городов из JSON файла
        self.cities_to_mask = set()
        cities_file = Path('src/data/cities.json')
        logger.info(f"Попытка загрузки городов из файла: {cities_file.absolute()}")
        
        if cities_file.exists():
            try:
                with open(cities_file, 'r', encoding='utf-8') as f:
                    cities_data = json.load(f)
                    if not isinstance(cities_data, list):
                        logger.error(f"Неверный формат данных в файле городов: ожидался список, получен {type(cities_data)}")
                        raise ValueError("Файл городов должен содержать список")
                    
                    # Выводим первые несколько городов для проверки
                    logger.info(f"Первые 5 городов из файла: {cities_data[:5]}")
                    
                    # Добавляем все варианты написания города (в разных регистрах)
                    for city in cities_data:
                        if not isinstance(city, str):
                            logger.warning(f"Пропуск некорректного значения города: {city} (тип: {type(city)})")
                            continue
                            
                        city_lower = city.lower()
                        self.cities_to_mask.add(city.strip())  # оригинальное написание
                        self.cities_to_mask.add(city_lower.strip())  # нижний регистр
                        self.cities_to_mask.add(city.upper().strip())  # верхний регистр
                        self.cities_to_mask.add(city.capitalize().strip())  # с заглавной буквы
                        
                logger.info(f"Загружено {len(cities_data)} городов из JSON файла")
                logger.info(f"Всего вариантов написания городов: {len(self.cities_to_mask)}")
                
                # Проверяем наличие некоторых известных городов, включая проблемный
                test_cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань', 'Космодемьянских']
                for city in test_cities:
                    variants = [
                        city.strip(),
                        city.lower().strip(),
                        city.upper().strip(),
                        city.capitalize().strip()
                    ]
                    found = [v for v in variants if v in self.cities_to_mask]
                    logger.info(f"Проверка города '{city}': найдены варианты: {found}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка при разборе JSON файла городов: {e}")
                self.cities_to_mask = set()
            except Exception as e:
                logger.error(f"Ошибка при загрузке городов из JSON: {e}")
                self.cities_to_mask = set()
        else:
            logger.error(f"Файл со списком городов не найден: {cities_file.absolute()}")
            self.cities_to_mask = set()
        
        # Загружаем медицинские термины из JSON файлов
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
            'src/data/russian_medical_terms_procedures.json',
            'src/data/russian_medical_terms_1.json'
        ]
        
        # Категории медицинских терминов
        self.medical_categories = {
            'symptoms': set(),
            'diagnosis': set(),
            'anatomy': set(),
            'drugs': set(),
            'procedures': set(),
            'abbreviations': set(),
            'general': set()
        }
        
        for terms_file in medical_terms_files:
            try:
                with open(terms_file, 'r', encoding='utf-8') as f:
                    terms = json.load(f)
                    if isinstance(terms, list):
                        self.medical_terms.update(term.lower() for term in terms)
                        # Определяем категорию по имени файла
                        category = 'general'
                        if 'symptoms' in terms_file:
                            category = 'symptoms'
                        elif 'diagnosis' in terms_file:
                            category = 'diagnosis'
                        elif 'anatomy' in terms_file:
                            category = 'anatomy'
                        elif 'drugs' in terms_file:
                            category = 'drugs'
                        elif 'procedures' in terms_file:
                            category = 'procedures'
                        elif 'abbreviations' in terms_file:
                            category = 'abbreviations'
                        self.medical_categories[category].update(term.lower() for term in terms)
                    elif isinstance(terms, dict):
                        self.medical_terms.update(term.lower() for term in terms.keys())
                        # Для словарей берем категорию из значения
                        for term, category in terms.items():
                            if category in self.medical_categories:
                                self.medical_categories[category].add(term.lower())
                logger.info(f"Загружено {len(terms)} терминов из {terms_file}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке терминов из {terms_file}: {e}")
        
        # Базовые атрибуты для проверки данных
        self.numeric_lengths = {10, 11, 12, 13, 14, 15, 16}
        self.numeric_patterns = {
            'passport': {'pattern': r'^\d{4}\s?\d{6}$', 'description': 'Паспорт'},
            'snils': {'pattern': r'^\d{3}-\d{3}-\d{3}\s?\d{2}$', 'description': 'СНИЛС'},
            'policy': {'pattern': r'^\d{16}$', 'description': 'Полис ОМС'},
            'phone': {'pattern': r'^\+?[78][\s\-\(]?\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}$', 'description': 'Телефон'},
            'med_card': {'pattern': r'^[А-Я]\d{6}|\d{6}[А-Я]$', 'description': 'Номер медкарты'}
        }
        
        # Расширенные медицинские паттерны
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
        
        # Контексты, где персональные данные допустимы
        self.allowed_contexts = {
            'doctor_signature': True,
            'medical_staff': True,
            'department_name': True,
            'medical_title': True,
            'medical_degree': True
        }
        
        # Медицинские протоколы и классификации
        self.medical_protocols = {
            'diagnosis': {
                'format': r'^[А-Я][а-яё]+(?:\s+[а-яё]+)*$',
                'required_context': ['симптом', 'признак', 'жалоба', 'анамнез', 'обследование']
            },
            'procedure': {
                'format': r'^[А-Я][а-яё]+(?:\s+[а-яё]+)*\s+(?:терапия|лечение|процедура|манипуляция)$',
                'required_context': ['показание', 'противопоказание', 'методика', 'результат']
            }
        }
        
        # Загружаем список фамилий
        self.russian_surnames = set()
        self.translit_surnames = set()
        
        # Загружаем фамилии из файла
        surnames_file = Path('src/data/russian-words/russian_surnames.txt')
        if not surnames_file.exists():
            raise FileNotFoundError(f"Файл со списком фамилий не найден: {surnames_file}")
            
        try:
            with open(surnames_file, 'r', encoding='windows-1251') as f:
                self.russian_surnames = {line.strip().lower() for line in f if line.strip()}
                self.translit_surnames = {transliterate(surname) for surname in self.russian_surnames}
            logger.info(f"Загружено {len(self.russian_surnames)} фамилий из файла")
        except Exception as e:
            logger.error(f"Ошибка при загрузке фамилий: {e}")
            raise
        
        # Список временных файлов для удаления при завершении
        self._temp_files: Set[Path] = set()
        atexit.register(self._cleanup_temp_files)

    def _cleanup_temp_files(self):
        """Удаляет все временные файлы при завершении работы"""
        logger.info("Начало очистки временных файлов...")
        
        # Сначала пытаемся удалить все файлы
        for file_path in self._temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Временный файл {file_path} успешно удален")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
        
        # Затем пытаемся удалить саму директорию
        try:
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
                logger.info(f"Временная директория {TEMP_DIR} успешно удалена")
        except Exception as e:
            logger.warning(f"Не удалось удалить временную директорию {TEMP_DIR}: {e}")

    def _load_medical_terms(self):
        """
        Загружает базу медицинских терминов из файла или создает её при первом запуске
        """
        terms_file = Path('data/medical_terms.json')
        
        if not terms_file.exists():
            logger.info("База медицинских терминов не найдена. Начинаем загрузку...")
            try:
                # Создаем директорию если её нет
                terms_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Загружаем термины из нескольких источников
                terms = set()
                
                # 1. Загружаем из Медицинского словаря
                try:
                    response = requests.get('https://medical-dictionary.thefreedictionary.com/')
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Ищем все термины на странице (пример селектора, нужно уточнить)
                        for term in soup.select('.term-list a'):
                            terms.add(term.text.strip().lower())
                except Exception as e:
                    logger.error(f"Ошибка при загрузке терминов из Medical Dictionary: {str(e)}")

                # 2. Загружаем из Медицинской энциклопедии
                try:
                    response = requests.get('https://www.medical-enc.ru/')
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Ищем все термины на странице (пример селектора, нужно уточнить)
                        for term in soup.select('.encyclopedia-list a'):
                            terms.add(term.text.strip().lower())
                except Exception as e:
                    logger.error(f"Ошибка при загрузке терминов из Медицинской энциклопедии: {str(e)}")

                # 3. Добавляем базовые медицинские термины
                basic_terms = {
                    # Анатомические термины
                    'голова', 'шея', 'грудь', 'спина', 'живот', 'таз', 'рука', 'нога',
                    'кисть', 'стопа', 'палец', 'глаз', 'ухо', 'нос', 'рот', 'зуб',
                    'язык', 'горло', 'пищевод', 'желудок', 'кишка', 'печень', 'почка',
                    'сердце', 'легкое', 'мозг', 'позвоночник', 'кость', 'мышца', 'связка',
                    'сустав', 'кровь', 'лимфа', 'нерв', 'сосуд', 'артерия', 'вена',
                    
                    # Медицинские процедуры
                    'осмотр', 'пальпация', 'перкуссия', 'аускультация', 'рентген',
                    'узи', 'кт', 'мрт', 'экг', 'эндоскопия', 'биопсия', 'операция',
                    'перевязка', 'инъекция', 'капельница', 'массаж', 'физиотерапия',
                    
                    # Симптомы и состояния
                    'боль', 'температура', 'кашель', 'насморк', 'тошнота', 'рвота',
                    'диарея', 'запор', 'головокружение', 'слабость', 'усталость',
                    'сонливость', 'бессонница', 'тревога', 'депрессия', 'стресс',
                    
                    # Диагнозы
                    'грипп', 'орви', 'пневмония', 'бронхит', 'гастрит', 'язва',
                    'гипертония', 'гипотония', 'диабет', 'артрит', 'артроз',
                    'остеохондроз', 'сколиоз', 'мигрень', 'инсульт', 'инфаркт',
                    
                    # Лекарства и препараты
                    'антибиотик', 'анальгетик', 'антисептик', 'витамин', 'гормон',
                    'иммуномодулятор', 'пробиотик', 'фермент', 'антигистамин',
                    
                    # Единицы измерения
                    'миллиметр', 'сантиметр', 'метр', 'миллилитр', 'литр',
                    'миллиграмм', 'грамм', 'килограмм', 'миллимоль', 'моль',
                    'миллиграмм-процент', 'миллиграмм-децилитр', 'миллиграмм-литр',
                    'микрограмм-литр', 'наномоль-литр', 'миллимоль-литр',
                    'единица-литр', 'международная-единица', 'миллиединица',
                    'килоединица', 'миллиединица-миллилитр', 'грамм-литр',
                    'пикограмм-миллилитр', 'фемтограмм', 'микрометр', 'нанометр',
                    'паскаль', 'килопаскаль', 'миллиметр-ртутного-столба',
                    'секунда', 'минута', 'час', 'сутки', 'неделя', 'месяц', 'год',
                    'удар-в-минуту', 'дыхание-в-минуту', 'миллиметр-в-час',
                    'грамм-в-сутки', 'миллиграмм-в-сутки', 'миллиграмм-на-килограмм',
                    'микрограмм-на-килограмм', 'миллиграмм-процент', 'миллиграмм-миллилитр',
                    'микрограмм-миллилитр', 'наномоль-литр', 'микромоль-литр',
                    'миллимоль-литр', 'моль-литр', 'единица-литр', 'международная-единица-литр',
                    'килоединица-литр', 'миллиединица-литр', 'относительная-единица',
                    'международное-нормализованное-отношение', 'протромбиновый-индекс',
                    'активированное-частичное-тромбопластиновое-время', 'фибриноген',
                    'тромбиновое-время', 'д-димер', 'растворимые-фибрин-мономерные-комплексы',
                    'протромбиновое-время', 'активность-протромбина',
                    
                    # Аббревиатуры
                    'оак', 'оам', 'бак', 'экг', 'узи', 'мрт', 'кт', 'рег', 'ээг',
                    'фгдс', 'фкс', 'ифа', 'пцр', 'соэ', 'лдг', 'алт', 'аст', 'ггт',
                    'щф', 'хс', 'лпнп', 'лпвп', 'тг', 'гп', 'сд', 'ад', 'чсс', 'чдд',
                    'spo2', 'sao2', 'fio2', 'pao2', 'paco2', 'ph', 'be', 'hb', 'rbc',
                    'wbc', 'plt', 'hct', 'mcv', 'mch', 'mchc', 'rdw', 'rdw-sd', 'rdw-cv',
                    'pct', 'mpv', 'pdw', 'esr', 'crp', 'ast', 'alt', 'ggt', 'alp', 'tbil',
                    'dbil', 'ibil', 'tp', 'alb', 'glob', 'urea', 'crea', 'glu', 'chol',
                    'tg', 'hdl', 'ldl', 'vldl', 'ck', 'ck-mb', 'ldh', 'amyl', 'lipase',
                    'na', 'k', 'cl', 'ca', 'p', 'mg', 'fe', 'ferr', 'tibc', 'uibc',
                    'transferrin', 'crp', 'procalcitonin', 'lactate', 'crp', 'esr', 'rf',
                    'aslo', 'ana', 'anca', 'dsdna', 'ena', 'ccp', 'apla', 'coags', 'pt',
                    'inr', 'aptt', 'tt', 'fibrinogen', 'd-dimer', 'fdp', 'atiii', 'pc',
                    'ps', 'ua', 'le', 'rbc', 'wbc', 'epi', 'cyl', 'crystals', 'bacteria',
                    'yeast', 'ph', 'sg', 'prot', 'glu', 'ket', 'bil', 'uro', 'nit', 'le',
                    'rbc', 'wbc', 'epi', 'cyl', 'csf', 'prot', 'glu', 'cells', 'cl',
                    'lactate', 'ecg', 'eeg', 'emg', 'enmg', 'echocg', 'us', 'ct', 'mri',
                    'x-ray', 'pet-ct', 'spect', 'fgds', 'fcs', 'colono', 'broncho',
                    'gastro', 'laparos', 'thoracos', 'arthros', 'ecmo', 'ivl', 'cpap',
                    'bipap', 'niv', 'hfnc', 'o2', 'iv', 'im', 'sc', 'po', 'pr', 'pv',
                    'sl', 'td', 'ih', 'neb', 'et', 'io', 'it', 'cpr', 'aed', 'acls',
                    'bls', 'pals', 'nrp', 'icd-10', 'icd-11', 'who', 'fda', 'ema', 'cdc',
                    'nih', 'nice', 'sign', 'uptodate', 'medlineplus', 'pubmed', 'elibrary',
                    'cyberleninka', 'google-scholar', 'web-of-science', 'scopus', 'ринц'
                }
                terms.update(basic_terms)

                # Сохраняем термины в файл
                with open(terms_file, 'w', encoding='utf-8') as f:
                    json.dump(list(terms), f, ensure_ascii=False, indent=2)
                
                logger.info(f"База медицинских терминов успешно создана: {len(terms)} терминов")
                
            except Exception as e:
                logger.error(f"Ошибка при создании базы медицинских терминов: {str(e)}")
                logger.exception("Полный стек ошибки:")
                return

        # Загружаем термины из файла
        try:
            with open(terms_file, 'r', encoding='utf-8') as f:
                self.medical_terms = set(json.load(f))
            logger.info(f"База медицинских терминов загружена: {len(self.medical_terms)} терминов")
        except Exception as e:
            logger.error(f"Ошибка при загрузке базы медицинских терминов: {str(e)}")
            logger.exception("Полный стек ошибки:")

    def _is_medical_term(self, word: str) -> bool:
        """
        Проверяет, является ли слово медицинским термином
        
        Args:
            word: проверяемое слово
            
        Returns:
            bool: является ли слово медицинским термином
        """
        return word.lower() in self.medical_terms

    async def process_document(self, file_path: str, clinic_name: str, output_dir: str) -> Tuple[str, Dict]:
        # Инициализируем флаг для маскирования контекста и отслеживания строк
        mask_context_active_for_line = False # Флаг активности маскирования контекста (город и слова после него) для ТЕКУЩЕЙ строки
        current_line_top = None  # Для отслеживания верхней границы текущей строки
        current_line_words_data = [] # Для сбора данных слов в текущей строке (включая координаты и текст)
        sensitive_regions = [] # Общий список регионов для маскирования (отдельные персональные данные и регионы «город + слова после»)

        try:
            file_path = Path(file_path).absolute()
            output_dir = Path(output_dir).absolute()

            logger.info(f"Начало обработки документа: {file_path}")
            logger.info(f"Выходная директория: {output_dir}")

            # Проверяем и создаем выходную директорию
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Выходная директория {output_dir} создана или уже существует")

                # Проверяем права на запись
                test_file = output_dir / "test_write.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    logger.info("Права на запись в выходную директорию подтверждены")
                except Exception as e:
                    logger.error(f"Нет прав на запись в директорию {output_dir}: {e}")
                    raise PermissionError(f"Нет прав на запись в директорию {output_dir}")

            except Exception as e:
                logger.error(f"Ошибка при создании/проверке выходной директории {output_dir}: {e}")
                raise

            image = None
            temp_image_path = None

            # Если это PDF файл, конвертируем его в изображение
            if file_path.suffix.lower() == '.pdf':
                logger.info(f"Конвертация PDF файла {file_path} в изображение...")
                try:
                    # Используем короткое имя для временного файла
                    temp_name = f"temp_{uuid.uuid4().hex[:8]}.jpg"
                    temp_path = TEMP_DIR / temp_name
                    self._temp_files.add(temp_path)
                    temp_image_path = temp_path

                    # Получаем короткие пути для Windows
                    short_pdf_path = get_short_path(file_path)
                    short_temp_dir = get_short_path(TEMP_DIR)

                    # Конвертируем PDF в изображение с меньшим DPI
                    images = convert_from_path(
                        pdf_path=short_pdf_path,
                        poppler_path=str(POPPLER_PATH),
                        output_folder=short_temp_dir,
                        fmt='jpeg',
                        dpi=200,
                        output_file=temp_name[:-4]
                    )

                    if not images:
                        raise ValueError(f"Не удалось конвертировать PDF файл: {file_path}")

                    # Ищем созданный файл
                    found_files = list(TEMP_DIR.glob(f"{temp_name[:-4]}*.jpg"))
                    if not found_files:
                        raise ValueError(f"Не удалось найти конвертированное изображение в {TEMP_DIR}")

                    # Берем первый найденный файл
                    image_path = found_files[0]
                    temp_image_path = image_path
                    self._temp_files.add(image_path)
                    short_image_path = get_short_path(image_path)

                    # Читаем изображение через PIL
                    pil_image = Image.open(short_image_path)
                    # Уменьшаем размер изображения если оно слишком большое
                    max_size = 3000
                    if max(pil_image.size) > max_size:
                        ratio = max_size / max(pil_image.size)
                        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                        logger.info(f"Изображение уменьшено до размера {new_size}")

                    # Конвертируем в формат для OpenCV
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    if image is None:
                        raise ValueError(f"Не удалось прочитать конвертированное изображение: {image_path}")

                    logger.info("PDF успешно конвертирован в изображение")
                except Exception as e:
                    logger.error(f"Ошибка при конвертации PDF: {str(e)}")
                    raise
            else:
                # Для обычных изображений используем PIL
                short_file_path = get_short_path(file_path)
                pil_image = Image.open(short_file_path)
                # Уменьшаем размер изображения если оно слишком большое
                max_size = 3000
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Изображение уменьшено до размера {new_size}")

                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {file_path}")

            # Распознаем текст
            text_data = self._recognize_text(image)

            # Выводим статистику распознанного текста
            total_words = len(text_data)
            logger.info(f"Всего распознано слов: {total_words}")
            logger.info("Первые 20 распознанных слов для анализа:")
            for i, word in enumerate(text_data[:20]):
                 logger.info(f"  {i}: '{word.get('text', '')}' (conf: {word.get('conf', 0):.2f})")

            # Проходим по распознанным словам для определения регионов маскирования
            for i, word_data in enumerate(text_data):
                word_text = word_data['text'].strip()
                if not word_text:
                    continue # Пропускаем пустые слова

                # Определяем, является ли это началом новой строки
                is_new_line = False
                if current_line_words_data:
                    # Сравниваем верхнюю границу текущего слова с верхней границей первого слова в собранных данных строки
                    # Порог определяем как 70% высоты текущего слова
                    line_break_threshold = word_data['height'] * 0.7
                    if abs(word_data['top'] - current_line_words_data[0]['top']) > line_break_threshold:
                         is_new_line = True
                else: # Если нет слов в текущей строке, это первое слово на новой строке
                     is_new_line = True

                # Если обнаружена новая строка, обрабатываем предыдущую (если она не пуста)
                if is_new_line and current_line_words_data:
                    # Если в предыдущей строке был город (mask_context_active_for_line), то маскируем только город и слова ПОСЛЕ него (до конца строки)
                    if mask_context_active_for_line:
                        # Ищем индекс слова-города (первое слово, которое совпадает с городом)
                        city_index = None
                        for idx, wd in enumerate(current_line_words_data):
                            clean_wd_text = re.sub(r'[.,!?;:]+$', '', wd['text'].strip()).strip()
                            # Проверяем, что слово начинается с заглавной буквы и является городом
                            if wd['text'] and wd['text'][0].isupper() and (clean_wd_text in self.cities_to_mask or clean_wd_text.lower() in self.cities_to_mask):
                                city_index = idx
                                break
                        if city_index is not None:
                            # Маскируем только город и слова после него (слева – левая граница города, справа – правая граница последнего слова)
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
                            logger.info(f"Добавлена маска для города и слов после него (переход на новую строку): {min_left}, {min_top}, {max_right-min_left}, {max_bottom-min_top} (текст: {sensitive_regions[-1]['text']})")
                        else:
                            logger.warning("В строке не найден город, хотя mask_context_active_for_line=True (не должно происходить).")

                      # Сбрасываем флаг контекста для новой строки и очищаем данные предыдущей строки
                    mask_context_active_for_line = False
                    current_line_words_data = []
                    current_line_top = word_data['top'] # Устанавливаем верхнюю границу для новой строки

                # Добавляем текущее слово в список данных текущей строки
                current_line_words_data.append(word_data)

                # Временно удаляем знаки препинания для проверки на город
                clean_word_text = re.sub(r'[.,!?;:]+$', '', word_text).strip()

                # Проверяем, является ли текущее слово городом (в любом регистре) используя очищенное слово
                is_city = False
                if word_text and word_text[0].isupper():
                    if clean_word_text in self.cities_to_mask or clean_word_text.lower() in self.cities_to_mask:
                        is_city = True

                if is_city:
                      # Если найден город, включаем флаг маскирования контекста для ОСТАТКА текущей строки
                      mask_context_active_for_line = True
                      logger.info(f"Найден город для маскирования: '{word_text}' (очищено: '{clean_word_text}', позиция {i})")
                      logger.info(f"Включен флаг маскирования контекста (город и слова после него) для текущей строки")

                # Логика добавления регионов для маскирования:
                # Если слово является отдельными персональными данными (и мы НЕ в контексте города, где маскируется город и слова после него),
                # добавляем его как отдельный регион.
                is_personal_data = False
                data_type = None
                if not mask_context_active_for_line:
                    is_numeric_personal, numeric_type = self._is_numeric_personal_data(word_text)
                    if is_numeric_personal:
                        is_personal_data = True
                        data_type = numeric_type
                    if not is_personal_data and not self._is_allowed_word(word_text, text_data, i):
                        is_personal_data = True
                        data_type = 'personal'
                if is_personal_data and not mask_context_active_for_line:
                    region = {
                        'left': word_data['left'],
                        'top': word_data['top'],
                        'width': word_data['width'],
                        'height': word_data['height'],
                        'type': data_type,
                        'text': word_text,
                        'position': i,
                        'is_personal': True,
                        'is_full_line': False # Это отдельный регион, не вся строка
                    }
                    sensitive_regions.append(region)
                    logger.info(f"Добавлен регион для маскирования персональных данных: '{word_text}' (тип: {region['type']}, позиция: {i}, координаты: {word_data['left']}, {word_data['top']}, размер: {word_data['width']}x{word_data['height']})")

            # Обрабатываем последнюю строку после завершения цикла по всем словам (если она не была пуста)
            if current_line_words_data:
                # Если в последней собранной строке был город (mask_context_active_for_line), маскируем только город и слова ПОСЛЕ него
                if mask_context_active_for_line:
                    city_index = None
                    for idx, wd in enumerate(current_line_words_data):
                        clean_wd_text = re.sub(r'[.,!?;:]+$', '', wd['text'].strip()).strip()
                        # Проверяем, что слово начинается с заглавной буквы и является городом
                        if wd['text'] and wd['text'][0].isupper() and (clean_wd_text in self.cities_to_mask or clean_wd_text.lower() in self.cities_to_mask):
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
                        logger.info(f"Добавлена маска для города и слов после него (конец обработки): {min_left}, {min_top}, {max_right-min_left}, {max_bottom-min_top} (текст: {sensitive_regions[-1]['text']})")
                    else:
                        logger.warning("В последней строке не найден город, хотя mask_context_active_for_line=True (не должно происходить).")

            # Проверяем, что у нас есть регионы для маскирования
            if not sensitive_regions:
                logger.warning("Не найдено регионов для маскирования!")
                logger.info("Проверка списка городов:")
                logger.info(f"Количество городов в базе: {len(self.cities_to_mask)}")
                logger.info(f"Примеры городов в базе: {list(self.cities_to_mask)[:10]}")
            else:
                logger.info(f"Найдено {len(sensitive_regions)} регионов для маскирования")
                personal_regions = [r for r in sensitive_regions if r.get('is_personal', False)] # Проверяем наличие ключа 'is_personal'
                city_lines = [r for r in sensitive_regions if r.get('type') == 'city_line']
                logger.info(f"Из них: отдельные персональные данные: {len(personal_regions)}, маски (город и слова после): {len(city_lines)}")

            # Маскируем найденные регионы (отдельные персональные данные и регионы «город + слова после»)
            masked_image = self._mask_sensitive_data(image, sensitive_regions)

            # Сохраняем деперсонализированное изображение
            output_filename = file_path.stem + '_depersonalized.jpg'
            output_path = output_dir / output_filename
            
            logger.info(f"Подготовка к сохранению деперсонализированного изображения: {output_path}")
            
            try:
                # Проверяем, существует ли уже файл
                if output_path.exists():
                    logger.warning(f"Файл {output_path} уже существует, будет перезаписан")
                    try:
                        output_path.unlink()
                    except Exception as e:
                        logger.error(f"Не удалось удалить существующий файл {output_path}: {e}")
                        raise
                
                # Сохраняем через PIL для лучшего контроля качества
                try:
                    # Конвертируем обратно в RGB для PIL
                    pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
                    
                    # Пробуем сохранить с разными параметрами качества
                    saved = False
                    for quality in [95, 90, 85, 80]:
                        try:
                            logger.info(f"Попытка сохранения с качеством {quality}...")
                            pil_image.save(output_path, 'JPEG', quality=quality, optimize=True)
                            
                            # Проверяем, что файл создался и имеет ненулевой размер
                            if output_path.exists() and output_path.stat().st_size > 0:
                                logger.info(f"Изображение успешно сохранено с качеством {quality}")
                                saved = True
                                break
                            else:
                                logger.warning(f"Файл создан, но имеет нулевой размер при качестве {quality}")
                        except Exception as e:
                            logger.warning(f"Не удалось сохранить с качеством {quality}: {e}")
                            continue
                    
                    if not saved:
                        # Если не удалось сохранить через PIL, пробуем через OpenCV
                        logger.info("Попытка сохранения через OpenCV...")
                        cv2.imwrite(str(output_path), masked_image)
                        
                        if output_path.exists() and output_path.stat().st_size > 0:
                            logger.info("Изображение успешно сохранено через OpenCV")
                        else:
                            raise ValueError("Не удалось сохранить изображение ни одним способом")
                            
                except Exception as e:
                    logger.error(f"Ошибка при сохранении через PIL/OpenCV: {e}")
                    raise
                    
                # Финальная проверка
                if not output_path.exists():
                    raise FileNotFoundError(f"Файл не был создан: {output_path}")
                if output_path.stat().st_size == 0:
                    raise ValueError(f"Файл создан, но имеет нулевой размер: {output_path}")
                    
                logger.info(f"Деперсонализированное изображение успешно сохранено: {output_path}")
                logger.info(f"Размер файла: {output_path.stat().st_size} байт")
                
            except Exception as e:
                logger.error(f"Критическая ошибка при сохранении файла {output_path}: {e}")
                raise
            
            return str(output_path), {'sensitive_regions': sensitive_regions}
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документа {file_path}: {str(e)}")
            raise
        finally:
            # Очищаем временные файлы
            if 'temp_image_path' in locals() and temp_image_path and temp_image_path.exists():
                try:
                    temp_image_path.unlink() # Удаляем временный файл сразу после использования
                    logger.info(f"Временный файл {temp_image_path} удален")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_image_path}: {e}")
            # Глобальная очистка временной директории при завершении программы
            # (зарегистрировано через atexit.register(self._cleanup_temp_files) в __init__)
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
                
            logger.info(f"Отправка документа {file_path} на анализ в DeepSeek...")
            analysis_result = await self.deepseek_client.analyze_medical_report(str(file_path))
            
            if analysis_result:
                # Выводим результат в консоль
                logger.info("Результат анализа DeepSeek:")
                logger.info("-" * 80)
                logger.info(analysis_result)
                logger.info("-" * 80)
                
                # Сохраняем результат
                self.deepseek_client.save_analysis_result(file_path, analysis_result)
                return analysis_result
            else:
                logger.error("Не удалось получить анализ от DeepSeek")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при анализе документа {file_path}: {str(e)}")
            return None

    async def process_directory(self, input_dir: str, clinic_name: str, output_dir: str) -> List[Tuple[str, Dict]]:
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
                raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
            if not input_path.is_dir():
                raise NotADirectoryError(f"Указанный путь не является директорией: {input_dir}")
                
            # Проверяем выходную директорию
            output_path = Path(output_dir).absolute()
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Выходная директория {output_path} создана или уже существует")
            except Exception as e:
                logger.error(f"Не удалось создать выходную директорию {output_path}: {e}")
                raise
            
            # Получаем список всех файлов изображений и PDF
            files = [f for f in input_path.glob('*') 
                    if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')]
            
            if not files:
                logger.warning(f"В директории {input_dir} не найдено подходящих файлов")
                return []
                
            logger.info(f"Найдено {len(files)} файлов для обработки")
            
            # Сначала обрабатываем все файлы (деперсонализация)
            for file_path in files:
                try:
                    logger.info(f"Обработка файла: {file_path.name}")
                    result = await self.process_document(str(file_path), clinic_name, output_dir)
                    results.append(result)
                    processed_files.append(result[0])  # Сохраняем путь к обработанному файлу
                    logger.info(f"Файл {file_path.name} успешно обработан")
                except Exception as e:
                    logger.error(f"Ошибка при обработке файла {file_path.name}: {str(e)}")
                    failed_files.append((str(file_path), str(e)))
                    continue
            
            # Выводим итоговую статистику
            logger.info(f"Обработка завершена. Успешно: {len(processed_files)}, Ошибок: {len(failed_files)}")
            if failed_files:
                logger.warning("Список файлов с ошибками:")
                for file_path, error in failed_files:
                    logger.warning(f"- {file_path}: {error}")
            
            return results
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке директории {input_dir}: {str(e)}")
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

            # Выводим статистику распознавания
            total_words = len([t for t in data['text'] if t.strip()])
            logger.info(f"Всего распознано слов: {total_words}")

            if total_words < 5:
                logger.info("Мало слов распознано на русском, пробуем английский...")
                eng_data = pytesseract.image_to_data(
                    image,
                    lang='eng',
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                eng_words = len([t for t in eng_data['text'] if t.strip()])
                logger.info(f"Распознано слов на английском: {eng_words}")
                
                if eng_words > total_words:
                    logger.info("Используем результаты английского распознавания")
                    data = eng_data

            words = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    confidence = float(data['conf'][i])
                    if confidence < 30:
                        logger.debug(f"Пропуск слова '{text}' из-за низкой уверенности: {confidence}")
                        continue

                    word = {
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': confidence,
                        'lang': 'eng' if all(c.isascii() for c in text) else 'rus'
                    }
                    words.append(word)
                    logger.debug(f"Распознано слово: '{text}' (уверенность: {confidence}, язык: {word['lang']})")

            logger.info(f"Итоговое количество распознанных слов: {len(words)}")
            return words
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста: {str(e)}")
            raise

    def _extract_data(self, text_data: List[Dict], mask_context: bool = False) -> Dict:
        """
        Извлекает данные из распознанного текста
        
        Args:
            text_data: список распознанных слов с координатами
            mask_context: флаг для отслеживания контекста после города
            
        Returns:
            Dict: словарь с извлеченными данными
        """
        sensitive_regions = []
        found_surnames = []
        medical_data = []

        for i, word in enumerate(text_data):
            try:
                word_text = word['text'].strip()
                if not word_text:
                    continue

                # Проверяем на город (в любом регистре)
                if word_text in self.cities_to_mask:
                    mask_context = True
                    logger.info(f"Найден город для маскирования: {word_text}")
                
                # Если мы в контексте после города, маскируем все слова
                if mask_context:
                    sensitive_regions.append({
                        'text': word_text,
                        'confidence': word.get('conf', 0),
                        'left': word.get('left', 0),
                        'top': word.get('top', 0),
                        'width': word.get('width', 0),
                        'height': word.get('height', 0),
                        'type': 'city_context'
                    })
                    logger.info(f"Маскирование слова в контексте города: {word_text}")
                    continue

                # Проверяем числовые данные
                is_personal_data, data_type = self._is_numeric_personal_data(word_text)
                if is_personal_data:
                    sensitive_regions.append({
                        'text': word_text,
                        'confidence': word.get('conf', 0),
                        'left': word.get('left', 0),
                        'top': word.get('top', 0),
                        'width': word.get('width', 0),
                        'height': word.get('height', 0),
                        'type': data_type
                    })
                    logger.info(f"Найдены числовые персональные данные: {word_text} ({data_type})")
                    continue

                # Проверяем на фамилии и персональные данные
                if not self._is_allowed_word(word_text, text_data, i):
                    found_surnames.append(word_text)
                    sensitive_regions.append({
                        'text': word_text,
                        'confidence': word.get('conf', 0),
                        'left': word.get('left', 0),
                        'top': word.get('top', 0),
                        'width': word.get('width', 0),
                        'height': word.get('height', 0),
                        'type': 'surname'
                    })
                    logger.info(f"Найдена фамилия для маскирования: {word_text}")

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

        # Проверяем медицинский контекст
        self._verify_medical_context(text_data, {'medical_data': medical_data})
        
        # Проверяем согласованность диагнозов
        self._verify_diagnoses({'medical_data': medical_data})
        
        return {
            'sensitive_regions': sensitive_regions,
            'medical_data': medical_data,
            'found_surnames': found_surnames
        }

    def _mask_sensitive_data(self, image: np.ndarray, sensitive_regions: List[Dict]) -> np.ndarray:
        """
        Маскирует чувствительные данные на изображении черными прямоугольниками
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

            # Группируем регионы по типам
            city_context_regions = [r for r in sensitive_regions if r.get('type') == 'city_context']
            other_regions = [r for r in sensitive_regions if r.get('type') != 'city_context']

            logger.info(f"Регионы для маскирования: всего {len(sensitive_regions)}, "
                       f"из них контекст города: {len(city_context_regions)}")

            # Сначала маскируем контекст городов
            if city_context_regions:
                # Сортируем регионы по позиции в тексте
                city_context_regions.sort(key=lambda x: x['position'])
                
                # Находим непрерывные последовательности регионов
                current_group = []
                groups = []
                for region in city_context_regions:
                    if not current_group or region['position'] == current_group[-1]['position'] + 1:
                        current_group.append(region)
                    else:
                        if current_group:
                            groups.append(current_group)
                        current_group = [region]
                if current_group:
                    groups.append(current_group)
                
                logger.info(f"Сформировано {len(groups)} групп для маскирования контекста города")
                
                # Маскируем каждую группу отдельно
                for i, group in enumerate(groups):
                    try:
                        # Находим общую область для группы
                        min_left = min(r['left'] for r in group)
                        min_top = min(r['top'] for r in group)
                        max_right = max(r['left'] + r['width'] for r in group)
                        max_bottom = max(r['top'] + r['height'] for r in group)
                        
                        # Добавляем небольшой отступ
                        padding = 2
                        min_left = max(0, min_left - padding)
                        min_top = max(0, min_top - padding)
                        max_right = min(masked_image.shape[1], max_right + padding)
                        max_bottom = min(masked_image.shape[0], max_bottom + padding)
                        
                        # Проверяем корректность координат
                        if (min_left >= max_right or min_top >= max_bottom or
                            max_right > masked_image.shape[1] or max_bottom > masked_image.shape[0]):
                            logger.warning(f"Некорректные координаты для группы {i+1}: "
                                         f"left={min_left}, top={min_top}, right={max_right}, bottom={max_bottom}")
                            continue
                        
                        # Маскируем всю область одним прямоугольником
                        masked_image[min_top:max_bottom, min_left:max_right] = 0
                        
                        # Проверяем, что маскирование прошло успешно
                        if np.all(masked_image[min_top:max_bottom, min_left:max_right] == 0):
                            logger.info(f"Успешно замаскирована группа {i+1}: "
                                      f"left={min_left}, top={min_top}, right={max_right}, bottom={max_bottom}, "
                                      f"размер={max_right-min_left}x{max_bottom-min_top}, "
                                      f"слова: '{group[0]['text']}' - '{group[-1]['text']}'")
                        else:
                            logger.warning(f"Неполное маскирование группы {i+1}")
                            
                    except Exception as e:
                        logger.error(f"Ошибка при маскировании группы {i+1}: {str(e)}")
                        continue

            # Затем маскируем остальные регионы
            for region in other_regions:
                try:
                    left = int(region.get('left', 0))
                    top = int(region.get('top', 0))
                    width = int(region.get('width', 0))
                    height = int(region.get('height', 0))
                    
                    # Проверяем корректность координат
                    if (left < 0 or top < 0 or width <= 0 or height <= 0 or
                        left + width > masked_image.shape[1] or
                        top + height > masked_image.shape[0]):
                        logger.warning(f"Пропуск некорректного региона: "
                                     f"left={left}, top={top}, width={width}, height={height}")
                        continue

                    # Добавляем небольшой отступ
                    padding = 2
                    left = max(0, left - padding)
                    top = max(0, top - padding)
                    width = min(masked_image.shape[1] - left, width + 2 * padding)
                    height = min(masked_image.shape[0] - top, height + 2 * padding)

                    # Маскируем регион
                    masked_image[top:top+height, left:left+width] = 0
                    
                    # Проверяем результат маскирования
                    if np.all(masked_image[top:top+height, left:left+width] == 0):
                        logger.info(f"Успешно замаскирован регион: '{region.get('text', '')}' "
                                  f"left={left}, top={top}, width={width}, height={height}")
                    else:
                        logger.warning(f"Неполное маскирование региона: '{region.get('text', '')}'")

                except Exception as e:
                    logger.error(f"Ошибка при маскировании региона {region}: {str(e)}")
                    continue

            # Проверяем, что маскирование прошло успешно
            total_masked_pixels = np.sum(masked_image == 0)
            total_pixels = masked_image.size
            masked_percentage = (total_masked_pixels / total_pixels) * 100
            
            logger.info(f"Маскирование завершено. Замаскировано {total_masked_pixels} пикселей "
                       f"({masked_percentage:.2f}% от общего количества)")
            
            return masked_image
            
        except Exception as e:
            logger.error(f"Критическая ошибка при маскировании: {str(e)}")
            raise

    def _is_allowed_word(self, word: str, text_data: List[Dict], current_index: int) -> bool:
        """
        Проверяет, является ли слово разрешенным (не персональными данными)
        
        Args:
            word: слово для проверки
            text_data: все слова в документе
            current_index: индекс текущего слова
            
        Returns:
            bool: является ли слово разрешенным
        """
        # Базовая проверка длины и содержания
        if not word or not word.strip():
            return True
            
        # Удаляем все небуквенные символы
        letters_only = ''.join(c for c in word if c.isalpha())
        if not letters_only:
            return True
            
        # Проверяем длину слова после удаления небуквенных символов
        if len(letters_only) < 3:  # Слова короче 3 букв пропускаем
            return True
            
        # Проверяем, начинается ли слово с заглавной буквы
        if not word[0].isupper():
            return True

        word_lower = letters_only.lower()
        
        # Проверяем на медицинские термины
        if self._is_medical_term(word):
            return True
        
        # Проверяем на фамилии
        is_surname = (word_lower in self.russian_surnames or 
                     word_lower in self.translit_surnames or
                     transliterate(word_lower) in self.translit_surnames)

        # Если слово похоже на фамилию, проверяем контекст
        if is_surname:
            # Проверяем на типичные медицинские контексты
            medical_contexts = [
                'врач', 'доктор', 'профессор', 'академик', 'заведующий',
                'главный', 'старший', 'младший', 'ординатор', 'интерн'
            ]
            
            # Получаем ближайший контекст (2 слова слева и справа)
            start_idx = max(0, current_index - 2)
            end_idx = min(len(text_data), current_index + 3)
            context_words = [w['text'].lower() for w in text_data[start_idx:end_idx]]
            context_text = ' '.join(context_words)
            
            # Если слово в медицинском контексте - пропускаем
            if any(ctx in context_text for ctx in medical_contexts):
                return True
            
            # Проверяем на шаблоны медицинских документов
            template_patterns = [
                r'(?:^|\s)пациент[а-я]*\s+[А-Я][а-я]+(?:\s|$)',
                r'(?:^|\s)больной\s+[А-Я][а-я]+(?:\s|$)',
                r'(?:^|\s)ф\.и\.о\.\s*[А-Я][а-я]+(?:\s|$)',
                r'(?:^|\s)фамилия\s*[А-Я][а-я]+(?:\s|$)',
                r'(?:^|\s)имя\s*[А-Я][а-я]+(?:\s|$)',
                r'(?:^|\s)отчество\s*[А-Я][а-я]+(?:\s|$)'
            ]
            
            if any(re.search(pattern, context_text, re.IGNORECASE) for pattern in template_patterns):
                return False
            
            # Если слово похоже на фамилию и не в медицинском контексте - маскируем
            return False

        # Проверяем на инициалы
        if len(word) <= 2 and word[0].isupper():
            # Проверяем, является ли это инициалом (следуют ли за ним другие инициалы или точка)
            next_word = next((w['text'] for w in text_data[current_index+1:current_index+2] if w['text'].strip()), '')
            if next_word and (next_word.endswith('.') or (len(next_word) <= 2 and next_word[0].isupper())):
                return False

        return True

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
            r'(?:Имя)[:\s]+([А-ЯЁ][а-яё]+)'
        ]
        
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
        
        # Проверяем, что фамилия есть в базе данных
        if surname and surname.lower() in self.russian_surnames:
            logger.info(f"Найдена фамилия в базе данных: {surname}")
            if name:
                logger.info(f"Найдено имя: {name}")
                return {
                    'surname': surname,
                    'name': name
                }
        
        logger.warning("Не удалось найти достоверную информацию о пациенте")
        return None

    def _is_numeric_personal_data(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет, является ли числовая строка персональными данными
        
        Args:
            text: строка для проверки
            
        Returns:
            Tuple[bool, str]: (является ли персональными данными, описание типа данных)
        """
        # Удаляем все нецифровые символы для проверки длины
        digits_only = ''.join(c for c in text if c.isdigit())
        
        # Проверяем длину
        if len(digits_only) not in self.numeric_lengths:
            # Дополнительные проверки из project2
            # Проверка на числа в формате даты (например, 01021990)
            if len(text) == 8 and text.isdigit():
                try:
                    day = int(text[:2])
                    month = int(text[2:4])
                    year = int(text[4:])
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                        return True, "Дата рождения"
                except:
                    pass
                    
            # Проверка на номера телефонов в разных форматах
            phone_patterns = [
                r'\+?[78][\s\-\(]?\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}',  # +7(999)123-45-67
                r'\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}',                   # 999-123-45-67
            ]
            
            for pattern in phone_patterns:
                if re.match(pattern, text):
                    return True, "Номер телефона"
                    
            # Проверка на номера медицинских карт
            med_card_patterns = [
                r'[А-Я]\d{6}',  # А123456
                r'\d{6}[А-Я]',  # 123456А
            ]
            
            for pattern in med_card_patterns:
                if re.match(pattern, text):
                    return True, "Номер медицинской карты"
                    
            return False, ""
            
        # Проверяем каждый паттерн (существующая логика)
        for data_type, pattern_info in self.numeric_patterns.items():
            if re.match(pattern_info['pattern'], text):
                return True, pattern_info['description']
                
        # Если длина совпадает с длиной паспорта (10 цифр), считаем это паспортными данными
        if len(digits_only) == 10:
            return True, "Паспорт"
            
        return False, ""

    def _verify_medical_context(self, text_data: List[Dict], extracted_data: Dict) -> None:
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
                        logger.warning(f"Диагноз '{term}' использован вне медицинского контекста")
                elif category == 'procedure':
                    if not self._verify_procedure_context(text, term):
                        logger.warning(f"Процедура '{term}' использована вне медицинского контекста")
                elif category == 'drugs':
                    if not self._verify_medication_context(text, term):
                        logger.warning(f"Препарат '{term}' использован вне медицинского контекста")

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
        medication_terms = {'дозировка', 'прием', 'назначение', 'курс', 'лечение', 'терапия'}
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
                for diagnosis2 in diagnoses[i+1:]:
                    if not self._are_diagnoses_compatible(diagnosis1, diagnosis2):
                        logger.warning(
                            f"Возможно несовместимые диагнозы: '{diagnosis1}' и '{diagnosis2}'"
                        )

    def _are_diagnoses_compatible(self, diagnosis1: str, diagnosis2: str) -> bool:
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

    def _is_personal_data(self, value: str, clean_value: str, pattern_name: str) -> Tuple[bool, str]:
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
        if pattern_name in ['snils', 'policy', 'passport', 'phone', 'med_card']:
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
            if re.match(personal_patterns[pattern_name]['pattern'], value.lower()):
                return True, personal_patterns[pattern_name]['description']
        
        return False, ""