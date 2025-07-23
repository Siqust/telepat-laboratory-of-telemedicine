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


# Путь к локальному poppler (Windows версия)
POPPLER_PATH = Path("../side-modules/poppler-24.08.0/Library/bin").absolute()
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
        self.numeric_lengths = {10, 11, 12, 13, 14, 15, 16}
        self.numeric_patterns = {
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
            'phone': {
                'pattern': r'^\+?[78][\s\-\(]?\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}$',
                'description': 'Телефон'
            },
            'med_card': {
                'pattern': r'^[А-Я]\d{6}|\d{6}[А-Я]$',
                'description': 'Номер медкарты'
            }
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

        # Загрузка медицинских терминов и других данных
        await self._load_medical_terms()
        await self._load_surnames()
        await self._load_cities()

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
                time.sleep(0.1)  # Даем время на освобождение файлов

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
                    except Exception as e:
                        logger.warning(
                            f"Не удалось удалить файл {file}: {str(e)}")
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

    async def process_document(self, file_path: str, clinic_name: str, output_dir: str) -> Tuple[str, Dict]:
        """Обработка документа: конвертация, распознавание текста, маскирование и сохранение"""
        # Инициализация переменных
        temp_files = []
        temp_dir = Path('temp_pdf_images')
        temp_dir.mkdir(exist_ok=True)
        output_files = []  # Список всех созданных файлов
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
                try:
                    from pdf2image import convert_from_path
                    import utils
                    poppler_path = utils.get_poppler_path()
                    logger.info(f"Использую poppler_path: {poppler_path}")
                    images = convert_from_path(file_path, poppler_path=poppler_path)
                    logger.info(f"PDF файл содержит {len(images)} страниц")
                except Exception as e:
                    logger.error(f"Ошибка при конвертации PDF: {str(e)}")
                    return None, {}
                # Создаем отдельную папку для этого PDF файла
                pdf_output_dir = output_dir / f"{file_path.stem}_pages"
                pdf_output_dir.mkdir(exist_ok=True)
                # PREDICT_IMAGE фильтрация для всех страниц
                import sys
                sys.path.append(str(Path(__file__).parent))
                import predict_image
                temp_files_for_predict = []
                for i, image in enumerate(images):
                    temp_file = temp_dir / f"temp_{uuid.uuid4().hex[:8]}-{i + 1}.jpg"
                    temp_files_for_predict.append(temp_file)
                    image.save(str(temp_file), 'JPEG', quality=95)
                    image.close()
                # Проверяем все страницы через predict_image
                skip_pdf = False
                for temp_file in temp_files_for_predict:
                    try:
                        pred = predict_image.predict_image(str(temp_file))
                        logger.info(f"predict_image для {temp_file}: {pred}")
                        if pred == 0:
                            logger.info(
                                f"PDF {file_path} пропущен: predict_image=0 хотя бы для одной страницы ({temp_file})")
                            skip_pdf = True
                            break
                    except Exception as e:
                        logger.error(f"Ошибка при вызове predict_image: {str(e)}")
                        skip_pdf = True
                        break
                if skip_pdf:
                    return None, {}
                # Если все страницы прошли, обрабатываем их
                for i, temp_file in enumerate(temp_files_for_predict):
                    page_output_file = await self._process_single_page(
                        temp_file, pdf_output_dir, f"page_{i + 1:03d}"
                    )
                    if page_output_file:
                        output_files.append(page_output_file)

            else:
                # Для обычных изображений
                temp_file = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.jpg"
                temp_files.append(temp_file)
                with Image.open(file_path) as img:
                    img.save(str(temp_file), 'JPEG', quality=95)
                # PREDICT_IMAGE фильтрация
                import sys
                sys.path.append(str(Path(__file__).parent))
                import predict_image
                try:
                    pred = predict_image.predict_image(str(temp_file))
                    logger.info(f"predict_image для {temp_file}: {pred}")
                    if pred == 0:
                        logger.info(f"Файл {temp_file} пропущен по результату предсказания (0)")
                        return None, {}
                except Exception as e:
                    logger.error(f"Ошибка при вызове predict_image: {str(e)}")
                    return None, {}
                # Обрабатываем одиночное изображение
                single_output_file = await self._process_single_page(
                    temp_file, output_dir, file_path.stem
                )
                if single_output_file:
                    output_files.append(single_output_file)

            # Анализ через AI (отправляем все файлы одним запросом)
            if output_files:
                analysis_result = await self._analyze_multiple_files(output_files, ai_results_dir)
                return str(output_files[0]), analysis_result  # Возвращаем первый файл как основной
            else:
                logger.error("Не удалось создать ни одного выходного файла")
                return None, {}

        except Exception as e:
            logger.error(f"Ошибка при обработке документа {file_path}: {str(e)}")
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

    async def _process_single_page(self, temp_file: Path, output_dir: Path, page_name: str) -> Optional[Path]:
        try:
            # Читаем изображение
            with Image.open(temp_file) as pil_image:
                max_size = 3000
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {temp_file}")

            # Распознавание текста
            text_data = self._recognize_text(image)

            # Анализ текста и определение регионов для маскирования
            sensitive_regions = []
            mask_context_active_for_line = False
            current_line_words_data = []

            for i, word_data in enumerate(text_data):
                word_text = word_data['text'].strip()
                if not word_text:
                    continue

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

                # Проверка на фамилии и имена
                if not self._is_allowed_word(word_text, text_data, i):
                    sensitive_regions.append({
                        'left': word_data['left'],
                        'top': word_data['top'],
                        'width': word_data['width'],
                        'height': word_data['height'],
                        'type': 'personal_name',
                        'text': word_text
                    })

            # Маскирование найденных регионов
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
                except Exception as e:
                    logger.warning(f"Ошибка при маскировании региона: {str(e)}")

            # Сохранение результата
            output_file = output_dir / f"{page_name}_depersonalized.jpg"

            # Конвертируем numpy array в PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Сохраняем через PIL
            pil_image.save(str(output_file), 'JPEG', quality=100)
            # Проверяем, что файл создался
            if not output_file.exists():
                raise RuntimeError("Файл не был создан после сохранения")
            # Проверяем размер файла
            if output_file.stat().st_size == 0:
                raise RuntimeError("Файл создан, но пустой")

            # Закрываем изображение
            pil_image.close()

            logger.info(f"Страница {page_name} успешно обработана и сохранена: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Ошибка при обработке страницы {page_name}: {str(e)}")
            return None

    async def _analyze_multiple_files(self, file_paths: List[Path], ai_results_dir: Path) -> Dict:
        """Анализирует несколько файлов через AI, отправляя их одним запросом"""
        try:
            if not file_paths:
                logger.warning("Нет файлов для анализа")
                return {}

            logger.info(f"Отправка {len(file_paths)} файлов на анализ в AI модели...")
            results = {}

            # Анализ через DeepSeek
            try:
                analysis_result = await self.deepseek_client.analyze_multiple_medical_reports(
                    [str(f) for f in file_paths]
                )
                if analysis_result:
                    results['deepseek'] = analysis_result
                    base_name = file_paths[0].stem.replace('_depersonalized', '')
                    analysis_file = ai_results_dir / f"{base_name}_deepseek_multi_page_analysis.json"
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Анализ DeepSeek для {len(file_paths)} файлов сохранен в {analysis_file}")
                else:
                    logger.warning("Не удалось получить анализ от DeepSeek")
            except Exception as e:
                logger.error(f"Ошибка при анализе через DeepSeek: {str(e)}")

            # Анализ через GigaChat
            try:
                analysis_result = await self.gigachat_client.analyze_multiple_medical_reports(
                    [str(f) for f in file_paths]
                )
                if analysis_result:
                    results['gigachat'] = analysis_result
                    base_name = file_paths[0].stem.replace('_depersonalized', '')
                    analysis_file = ai_results_dir / f"{base_name}_gigachat_multi_page_analysis.json"
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Анализ GigaChat для {len(file_paths)} файлов сохранен в {analysis_file}")
                else:
                    logger.warning("Не удалось получить анализ от GigaChat")
            except Exception as e:
                logger.error(f"Ошибка при анализе через GigaChat: {str(e)}")

            # Анализ через ChatGPT
            try:
                from chatgpt_client import ChatGPTClient
                chatgpt_client = ChatGPTClient()
                analysis_result = await chatgpt_client.analyze_multiple_medical_reports(
                    [str(f) for f in file_paths]
                )
                if analysis_result:
                    results['chatgpt'] = analysis_result
                    base_name = file_paths[0].stem.replace('_depersonalized', '')
                    analysis_file = ai_results_dir / f"{base_name}_chatgpt_multi_page_analysis.json"
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Анализ ChatGPT для {len(file_paths)} файлов сохранен в {analysis_file}")
                else:
                    logger.warning("Не удалось получить анализ от ChatGPT")
            except Exception as e:
                logger.error(f"Ошибка при анализе через ChatGPT: {str(e)}")

            if results:
                base_name = file_paths[0].stem.replace('_depersonalized', '')
                summary_file = ai_results_dir / f"{base_name}_multi_page_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Сводный анализ для {len(file_paths)} файлов сохранен в {summary_file}")
                return results
            else:
                logger.error(f"Не удалось получить анализ ни от одной AI модели для {len(file_paths)} файлов")
                return {}
        except Exception as e:
            logger.error(f"Ошибка при анализе нескольких файлов: {str(e)}")
            return {}


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

            words = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:
                    confidence = float(data['conf'][i])
                    if confidence < 30:
                        logger.debug(
                            f"Пропуск слова '{text}' из-за низкой уверенности: {confidence}")
                        continue

                    word = {
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': confidence,
                        'lang': 'eng' if all(
                            c.isascii() for c in text) else 'rus'}
                    words.append(word)
                    logger.debug(
                        f"Распознано слово: '{text}' (уверенность: {confidence}, язык: {word['lang']})")

            logger.info(f"Итоговое количество распознанных слов: {len(words)}")
            return words
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста: {str(e)}")
            raise

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
        if word_lower in self.medical_terms:
            return True

        # Проверяем на разрешенные слова
        if word_lower in self.allowed_words:
            return True

        # Проверяем на фамилии
        is_surname = (word_lower in self.surnames or
                      transliterate(word_lower) in self.surnames or
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
            # Если это последнее слово или следующее слово тоже инициал - пропускаем
            if (current_index == len(text_data) - 1 or
                    (current_index + 1 < len(text_data) and
                     (text_data[current_index + 1]['text'].endswith('.') or
                      (len(text_data[current_index + 1]['text']) <= 2 and
                       text_data[current_index + 1]['text'][0].isupper())))):
                return True

        # Проверяем на числовые персональные данные
        is_personal, _ = self._is_numeric_personal_data(word)
        if is_personal:
            return False

        # По умолчанию разрешаем слово
        return True


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
                except BaseException:
                    pass

            # Проверка на номера телефонов в разных форматах
            phone_patterns = [
                # +7(999)123-45-67
                r'\+?[78][\s\-\(]?\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}',
                # 999-123-45-67
                r'\d{3}[\s\-\(]?\d{3}[\s\-\(]?\d{2}[\s\-\(]?\d{2}',
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

        # Если длина совпадает с длиной паспорта (10 цифр), считаем это
        # паспортными данными
        if len(digits_only) == 10:
            return True, "Паспорт"

        return False, ""
