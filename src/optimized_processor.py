"""
Оптимизированный модуль для обработки медицинских документов
"""

import os
import re
import uuid
import logging
import numpy as np
import pytesseract
import cv2
from typing import Dict, List, Optional, Tuple
from src.database import Database
from src.medical_terms import MedicalTermsManager
from src import utils

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Класс для обработки медицинских документов.
    
    Основные функции:
    - Распознавание текста на изображениях
    - Извлечение медицинских данных
    - Проверка медицинских терминов
    - Проверка контекста терминов
    - Проверка согласованности диагнозов
    - Проверка протоколов процедур
    - Логирование процесса обработки
    
    Attributes:
        db (Optional[Database]): База данных для сохранения результатов
        medical_terms (MedicalTermsManager): Менеджер медицинских терминов
        logger (logging.Logger): Логгер для записи информации о процессе
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Инициализация процессора
        
        Args:
            db: экземпляр базы данных (опционально)
        """
        self.db = db
        self.medical_terms = MedicalTermsManager()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация DocumentProcessor")
        
        # Паттерны для медицинских данных
        self.medical_patterns = {
            'birth_date': r'(?:дата рождения|д\.р\.|др)[:\s]*(\d{2}[\.\s]\d{2}[\.\s]\d{4})',
            'gender': r'(?:пол)[:\s]*([мж])',
            'blood_group': r'(?:Группа крови|Гр\.крови)[:\s]*([\dIV]+(?:\s?[+\-])?)',
            'rh_factor': r'(?:Резус-фактор|Rh-фактор)[:\s]*([+\-])',
            'snils': r'(?:СНИЛС|СНИЛС №)[:\s]*(\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2})',
            'oms_policy': r'(?:Полис ОМС|ОМС)[:\s]*(\d{16})',
            'passport_details': r'(?:Паспорт|Серия и номер|Серия №)[:\s]*(\d{4}[-\s]?\d{6})',
            'phone_number': r'(?:Телефон|Тел|Моб\.|Мобильный)[:\s]*([\d\s\-\(\)\+]+)',
            'email': r'(?:Email|E-mail|Почта)[:\s]*([\w\.\-]+@[\w\.\-]+)',
            'date': r'(\d{2}[\.\s]\d{2}[\.\s]\d{4})',
            'time': r'(\d{2}:\d{2})',
            'datetime': r'(\d{2}[\.\s]\d{2}[\.\s]\d{4}\s+\d{2}:\d{2})'
        }
        
        # Паттерны для числовых персональных данных
        self.numeric_patterns = {
            'oms': {
                'length': 16,
                'pattern': r'^\d{16}$',
                'description': 'Полис ОМС',
                'keywords': ['полис', 'омс', 'страховой']
            },
            'snils': {
                'length': 11,
                'pattern': r'^\d{11}$',
                'description': 'СНИЛС',
                'keywords': ['снилс', 'страховой номер']
            },
            'passport': {
                'length': 10,
                'pattern': r'^\d{10}$',
                'description': 'Паспорт',
                'keywords': ['паспорт', 'серия', 'номер']
            }
        }
        
        # Список длин числовых данных для проверки
        self.numeric_lengths = {pattern['length'] for pattern in self.numeric_patterns.values()}
        
        # Минимальный порог уверенности для распознавания текста
        self.min_confidence = 60
        
        self.logger.info("Инициализация завершена")
    
    def process_document(self, file_path: str, clinic_name: str, output_dir: str) -> Tuple[str, Dict]:
        """
        Обработка одного документа
        
        Args:
            file_path: путь к файлу
            clinic_name: название клиники
            output_dir: директория для сохранения результатов
            
        Returns:
            Tuple[str, Dict]: путь к обработанному файлу и извлеченные данные
        """
        # Генерируем уникальный ID документа и анализа
        document_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        
        # Загружаем изображение
        image, file_type = utils.load_image(file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")
        
        # Распознаем текст
        text_data = self._recognize_text(image)
        
        # Извлекаем данные
        extracted_data = self._extract_data(text_data)
        
        # Проверяем контекст медицинских терминов
        self._verify_medical_context(text_data, extracted_data)
        
        # Проверяем согласованность диагнозов
        self._verify_diagnoses(extracted_data)
        
        # Проверяем протоколы процедур
        self._verify_procedures(extracted_data)
        
        # Маскируем персональные данные на изображении
        masked_image = self._mask_sensitive_data(image, text_data, extracted_data)
        
        # Сохраняем обработанное изображение
        output_filename = f"analysis_{analysis_id}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        utils.save_image(masked_image, output_path)
        
        # Сохранение в базу данных
        if self.db is not None:
            self._save_to_database(
                document_id=document_id,
                original_filename=file_path,
                processed_filename=output_path,
                clinic_name=clinic_name,
                extracted_data=extracted_data
            )
        
        return output_path, extracted_data
    
    def _recognize_text(self, image: np.ndarray) -> List[Dict]:
        """
        Распознавание текста на изображении
        
        Args:
            image: изображение для распознавания
            
        Returns:
            List[Dict]: список распознанных слов с их координатами
        """
        try:
            data = pytesseract.image_to_data(
                image, 
                lang='rus+eng',
                config='--psm 6 --oem 1',  # Используем LSTM OCR Engine Mode
                output_type=pytesseract.Output.DICT
            )
            
            if len([t for t in data['text'] if t.strip()]) < 5:
                eng_data = pytesseract.image_to_data(
                    image,
                    lang='eng',
                    config='--psm 6 --oem 1',
                    output_type=pytesseract.Output.DICT
                )
                
                if len([t for t in eng_data['text'] if t.strip()]) > len([t for t in data['text'] if t.strip()]):
                    data = eng_data
            
            words = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    confidence = float(data['conf'][i])
                    if confidence < self.min_confidence:
                        continue
                    
                    word = {
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': confidence,
                        'lang': 'eng' if all(c.isascii() for c in data['text'][i]) else 'rus'
                    }
                    words.append(word)
            
            return words
        except Exception as e:
            self.logger.error(f"Ошибка при распознавании текста: {str(e)}")
            raise
    
    def _extract_data(self, text_data: List[Dict]) -> Dict:
        """
        Извлечение данных из распознанного текста
        
        Args:
            text_data: список распознанных слов
            
        Returns:
            Dict: извлеченные данные
        """
        medical_data = []
        sensitive_data = []
        
        try:
            # Извлекаем медицинские данные по паттернам
            text = ' '.join(word['text'] for word in text_data)
            
            # Проверяем каждый паттерн
            for pattern_name, pattern in self.medical_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1).strip()
                    full_text = match.group(0)
                    
                    # Очищаем значение от пробелов и дефисов для проверки
                    clean_value = re.sub(r'[\s\-]', '', value)
                    
                    # Проверяем, является ли это медицинским термином
                    if self.medical_terms.is_medical_term(value):
                        medical_data.append({
                            'type': pattern_name,
                            'value': value,
                            'text': full_text,
                            'category': self.medical_terms.get_term_category(value)
                        })
                    else:
                        # Проверяем на персональные данные
                        is_personal, data_type = self._is_personal_data(value, clean_value, pattern_name)
                        if is_personal:
                            sensitive_data.append({
                                'type': pattern_name,
                                'value': value,
                                'clean_value': clean_value,  # Добавляем очищенное значение
                                'text': full_text,
                                'data_type': data_type
                            })
                            
                            self.logger.info(f"Обнаружены персональные данные: {data_type} - {value}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении данных: {str(e)}")
        
        return {
            'medical_data': medical_data,
            'sensitive_data': sensitive_data
        }
    
    def _verify_medical_context(self, text_data: List[Dict], extracted_data: Dict) -> None:
        """
        Проверка контекста медицинских терминов
        
        Args:
            text_data: список распознанных слов
            extracted_data: извлеченные данные
        """
        text = ' '.join(word['text'] for word in text_data)
        
        for medical_item in extracted_data.get('medical_data', []):
            if not self.medical_terms.check_term_context(text, medical_item['value']):
                self.logger.warning(
                    f"Медицинский термин '{medical_item['value']}' "
                    f"использован вне медицинского контекста"
                )
    
    def _verify_diagnoses(self, extracted_data: Dict) -> None:
        """
        Проверка согласованности диагнозов
        
        Args:
            extracted_data: извлеченные данные
        """
        for medical_item in extracted_data.get('medical_data', []):
            if medical_item.get('category') == 'diagnosis':
                if not self.medical_terms.check_diagnosis_consistency(medical_item['text']):
                    self.logger.warning(
                        f"Диагноз '{medical_item['value']}' "
                        f"не соответствует формату или не согласован"
                    )
    
    def _verify_procedures(self, extracted_data: Dict) -> None:
        """
        Проверка протоколов процедур
        
        Args:
            extracted_data: извлеченные данные
        """
        for medical_item in extracted_data.get('medical_data', []):
            if medical_item.get('category') == 'procedures':
                if not self.medical_terms.check_procedure_protocol(medical_item['text']):
                    self.logger.warning(
                        f"Процедура '{medical_item['value']}' "
                        f"не соответствует протоколу"
                    )
    
    def _is_medical_term(self, text: str) -> bool:
        """
        Проверяет, является ли текст медицинским термином
        
        Args:
            text: проверяемый текст
            
        Returns:
            bool: является ли текст медицинским термином
        """
        return self.medical_terms.is_medical_term(text)
    
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
        if pattern_name in ['snils', 'oms_policy', 'passport_details']:
            # Проверяем длину очищенного значения
            if len(clean_value) in self.numeric_lengths:
                # Проверяем каждый паттерн
                for data_type, pattern_info in self.numeric_patterns.items():
                    if re.match(pattern_info['pattern'], clean_value):
                        return True, pattern_info['description']
        
        # Проверяем другие типы персональных данных
        personal_patterns = {
            'birth_date': 'Дата рождения',
            'phone_number': 'Номер телефона',
            'email': 'Email',
            'gender': 'Пол'
        }
        
        if pattern_name in personal_patterns:
            return True, personal_patterns[pattern_name]
        
        return False, ""
    
    def _save_to_database(self, document_id: str, original_filename: str,
                         processed_filename: str, clinic_name: str,
                         extracted_data: Dict) -> None:
        """
        Сохранение результатов в базу данных
        
        Args:
            document_id: ID документа
            original_filename: путь к исходному файлу
            processed_filename: путь к обработанному файлу
            clinic_name: название клиники
            extracted_data: извлеченные данные
        """
        if self.db is None:
            return
        
        try:
            self.db.add_document(
                document_id=document_id,
                original_filename=original_filename,
                processed_filename=processed_filename,
                clinic_name=clinic_name,
                medical_data=extracted_data.get('medical_data', []),
                sensitive_data=extracted_data.get('sensitive_data', [])
            )
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в базу данных: {str(e)}")
            raise

    def _mask_sensitive_data(self, image: np.ndarray, text_data: List[Dict], 
                           extracted_data: Dict) -> np.ndarray:
        """
        Маскирование персональных данных на изображении
        
        Args:
            image: исходное изображение
            text_data: список распознанных слов
            extracted_data: извлеченные данные
            
        Returns:
            np.ndarray: изображение с замаскированными персональными данными
        """
        masked_image = image.copy()
        
        try:
            # Получаем список чувствительных данных для маскирования
            sensitive_items = extracted_data.get('sensitive_data', [])
            
            # Создаем маску для каждого чувствительного элемента
            for item in sensitive_items:
                # Ищем все вхождения этого значения в тексте
                for word in text_data:
                    # Очищаем текст слова для сравнения
                    clean_word = re.sub(r'[\s\-]', '', word['text'])
                    
                    # Проверяем совпадение с очищенным значением
                    if item['clean_value'] in clean_word:
                        # Получаем координаты слова
                        x = word['left']
                        y = word['top']
                        w = word['width']
                        h = word['height']
                        
                        # Проверяем, не является ли это медицинским термином из белого списка
                        if not self.medical_terms.is_medical_term(word['text']):
                            # Маскируем область с персональными данными
                            cv2.rectangle(
                                masked_image,
                                (x, y),
                                (x + w, y + h),
                                (0, 0, 0),  # Черный цвет
                                -1  # Заполненный прямоугольник
                            )
                            
                            # Добавляем текст "***" поверх замаскированной области
                            cv2.putText(
                                masked_image,
                                "***",
                                (x, y + h - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,  # Размер шрифта
                                (255, 255, 255),  # Белый цвет
                                1  # Толщина линии
                            )
                            
                            self.logger.info(
                                f"Замаскированы персональные данные: {item['data_type']} "
                                f"в позиции ({x}, {y})"
                            )
            
            # Дополнительная проверка на числовые последовательности
            for word in text_data:
                # Очищаем текст слова
                clean_word = re.sub(r'[\s\-]', '', word['text'])
                
                # Проверяем, не является ли это медицинским термином
                if not self.medical_terms.is_medical_term(word['text']):
                    # Проверяем на персональные данные
                    is_personal, data_type = self._is_personal_data(word['text'], clean_word, '')
                    if is_personal:
                        x = word['left']
                        y = word['top']
                        w = word['width']
                        h = word['height']
                        
                        # Маскируем область
                        cv2.rectangle(
                            masked_image,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 0),
                            -1
                        )
                        
                        # Добавляем текст "***"
                        cv2.putText(
                            masked_image,
                            "***",
                            (x, y + h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                        
                        self.logger.info(
                            f"Замаскированы числовые персональные данные: {data_type} "
                            f"в позиции ({x}, {y})"
                        )
            
            return masked_image
            
        except Exception as e:
            self.logger.error(f"Ошибка при маскировании данных: {str(e)}")
            return image  # Возвращаем исходное изображение в случае ошибки 