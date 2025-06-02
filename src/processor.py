"""
Основной модуль для обработки документов
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pytesseract
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc
)
from . import utils
from .database import Database

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentProcessor:
    """Класс для обработки медицинских документов"""
    
    def __init__(self, db: Optional[Database] = None):
        """
        Инициализация процессора
        
        Args:
            db: экземпляр базы данных (опционально)
        """
        self.db = db
        
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)
        
        # Паттерны для поиска персональных данных (кроме имен)
        self.patterns = {
            'address': r'(?:Адрес пациента|Адрес)[:\s]+(.+?(?=\s*(?:Наименование исследования|$)))',
            'birth_date': r'(?:Дата рождения|Рождён)[:\s]+(\d{2}[/\.]\d{2}[/\.]\d{4}|\d{6,8})',
            'passport': r'(?:Паспорт|Серия и номер)[:\s]*(\d{4}\s?\d{6})',
            'oms': r'(?:Полис ОМС|Страховой полис)[:\s]*(\d{16})'
        }
        
        # Паттерны для медицинских данных (не для маскирования)
        self.medical_patterns = {
            'birth_date': r'дата рождения:.*?(\d{2}\.\d{2}\.\d{4})',
            'gender': r'пол:.*?([мж])'
        }
    
    def process_document(self, 
                        file_path: str,
                        clinic_name: str,
                        output_dir: str) -> Tuple[str, Dict]:
        """
        Обработка документа
        
        Args:
            file_path: путь к файлу
            clinic_name: название клиники
            output_dir: директория для сохранения результатов
            
        Returns:
            Tuple[str, Dict]: (путь к обработанному файлу, извлеченные данные)
        """
        # Генерация ID документа
        document_id = utils.generate_document_id()
        
        # Загрузка и предобработка изображения
        image, file_type = utils.load_image(file_path)
        processed_image = utils.preprocess_image(image)
        
        # Распознавание текста
        text_data = self._recognize_text(processed_image)
        
        # Логируем распознанный текст
        full_text = ' '.join(word['text'] for word in text_data)
        print(f"\n[Распознанный текст для файла {file_path}]:\n{full_text}\n")
        
        # Извлечение данных
        extracted_data = self._extract_data(text_data)
        
        # Логируем деперсонализируемые данные
        if extracted_data['sensitive_regions']:
            print(f"[Данные для деперсонализации в файле {file_path}]:")
            for region in extracted_data['sensitive_regions']:
                print(f"  - Тип: {region['type']}, Текст: {region['text']}, Координаты: (x={region['left']}, y={region['top']}, w={region['width']}, h={region['height']})")
        else:
            print(f"[Нет данных для деперсонализации в файле {file_path}]")
        
        # Маскирование чувствительных данных
        masked_image = self._mask_sensitive_data(image, extracted_data['sensitive_regions'])
        
        # Сохранение результатов
        output_filename = f"analysis_{document_id}.jpg"
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
            image: изображение
            
        Returns:
            List[Dict]: список распознанных слов с координатами
        """
        # Получаем данные от Tesseract
        data = pytesseract.image_to_data(image, lang='rus', output_type=pytesseract.Output.DICT)
        
        # Формируем список слов с координатами
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                words.append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
        
        return words
    
    def _extract_data(self, text_data: List[Dict]) -> Dict:
        """
        Извлечение данных из распознанного текста
        
        Args:
            text_data: список распознанных слов
            
        Returns:
            Dict: извлеченные данные
        """
        # Объединяем все слова в один текст
        full_text = ' '.join(word['text'] for word in text_data)
        
        # Создаем документ для Natasha
        doc = Doc(full_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)
        
        # Извлекаем именованные сущности
        for span in doc.spans:
            span.normalize(self.morph_vocab)
        
        # Ищем персональные данные
        sensitive_regions = []
        medical_data = {}

        # Извлечение имен с помощью Natasha
        matches = list(self.names_extractor(full_text))
        for match in matches:
            # Получаем начало и конец имени в тексте
            start_pos = match.start
            end_pos = match.stop
            
            # Ищем слова, которые попадают в этот диапазон
            for word in text_data:
                word_start = full_text.find(word['text'], start_pos)
                if word_start >= start_pos and word_start <= end_pos:
                    # Формируем нормализованное имя
                    name_parts = []
                    if match.fact.first:
                        name_parts.append(match.fact.first)
                    if match.fact.last:
                        name_parts.append(match.fact.last)
                    if match.fact.middle:
                        name_parts.append(match.fact.middle)
                    
                    normalized_name = ' '.join(name_parts)
                    
                    sensitive_regions.append({
                        'type': 'name',
                        'text': normalized_name,
                        'left': word['left'],
                        'top': word['top'],
                        'width': word['width'],
                        'height': word['height']
                    })     
        # Поиск по остальным паттернам
        for data_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                start_pos = match.start()
                end_pos = match.end()
                
                for word in text_data:
                    word_start = full_text.find(word['text'], start_pos)
                    if word_start >= start_pos and word_start <= end_pos:
                        sensitive_regions.append({
                            'type': data_type,
                            'text': match.group(1),
                            'left': word['left'],
                            'top': word['top'],
                            'width': word['width'],
                            'height': word['height']
                        })
        
        # Поиск медицинских данных (не для маскирования)
        for data_type, pattern in self.medical_patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                medical_data[data_type] = match.group(1)
        
        return {
            'sensitive_regions': sensitive_regions,
            'medical_data': medical_data
        }
    
    def _mask_sensitive_data(self, 
                           image: np.ndarray,
                           sensitive_regions: List[Dict]) -> np.ndarray:
        """
        Маскирование чувствительных данных на изображении
        
        Args:
            image: исходное изображение
            sensitive_regions: список регионов для маскирования
            
        Returns:
            np.ndarray: изображение с замаскированными данными
        """
        masked_image = image.copy()
        
        for region in sensitive_regions:
            # Получаем координаты
            x = region['left']
            y = region['top']
            w = region['width']
            h = region['height']
            
            # Закрашиваем область черным цветом
            cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        return masked_image
    
    def _save_to_database(self,
                         document_id: str,
                         original_filename: str,
                         processed_filename: str,
                         clinic_name: str,
                         extracted_data: Dict) -> None:
        """
        Сохранение данных в базу
        
        Args:
            document_id: ID документа
            original_filename: путь к оригинальному файлу
            processed_filename: путь к обработанному файлу
            clinic_name: название клиники
            extracted_data: извлеченные данные
        """
        if self.db is not None:
            self.db.save_document(
                document_id=document_id,
                original_filename=original_filename,
                processed_filename=processed_filename,
                clinic_name=clinic_name,
                extracted_data=extracted_data
            )