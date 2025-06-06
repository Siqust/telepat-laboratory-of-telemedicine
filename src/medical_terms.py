"""
Модуль для работы с медицинскими терминами
"""

import json
import os
from typing import Set, Dict, List
import logging

class MedicalTermsManager:
    def __init__(self):
        """Инициализация менеджера медицинских терминов"""
        self.logger = logging.getLogger(__name__)
        self.terms: Set[str] = set()
        self.load_terms()
        
    def load_terms(self) -> None:
        """Загрузка медицинских терминов из JSON файлов"""
        data_dir = os.path.join('src', 'data')
        
        # Список файлов для загрузки
        json_files = [
            'russian_medical_terms_symptoms.json',
            'russian_medical_terms_diagnosis.json',
            'russian_medical_terms_anatomy.json',
            'russian_medical_terms_drugs.json',
            'russian_medical_terms_procedures.json',
            'russian_medical_terms_1.json',
            'english_stats_terms.json',
            'russian_medical_terms_whitelist.json',
            'russian_stats_terms.json',
            'english_medical_terms.json'
        ]
        
        for file_name in json_files:
            try:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Извлекаем термины в зависимости от структуры JSON
                        if isinstance(data, dict):
                            # Если это словарь, берем все значения
                            terms = self._extract_terms_from_dict(data)
                        elif isinstance(data, list):
                            # Если это список, берем все элементы
                            terms = self._extract_terms_from_list(data)
                        else:
                            terms = set()
                            
                        self.terms.update(terms)
                        self.logger.info(f"Загружено {len(terms)} терминов из {file_name}")
                        
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке {file_name}: {str(e)}")
                
        self.logger.info(f"Всего загружено {len(self.terms)} уникальных медицинских терминов")
        
    def _extract_terms_from_dict(self, data: Dict) -> Set[str]:
        """Извлечение терминов из словаря"""
        terms = set()
        
        def extract_from_dict(d: Dict, current_path: List[str] = None):
            if current_path is None:
                current_path = []
                
            for key, value in d.items():
                if isinstance(value, dict):
                    extract_from_dict(value, current_path + [key])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            terms.add(item.lower())
                        elif isinstance(item, dict):
                            extract_from_dict(item, current_path + [key])
                elif isinstance(value, str):
                    terms.add(value.lower())
                    
        extract_from_dict(data)
        return terms
        
    def _extract_terms_from_list(self, data: List) -> Set[str]:
        """Извлечение терминов из списка"""
        terms = set()
        
        def extract_from_list(lst: List):
            for item in lst:
                if isinstance(item, str):
                    terms.add(item.lower())
                elif isinstance(item, dict):
                    for value in item.values():
                        if isinstance(value, str):
                            terms.add(value.lower())
                        elif isinstance(value, list):
                            extract_from_list(value)
                            
        extract_from_list(data)
        return terms
        
    def is_medical_term(self, text: str) -> bool:
        """
        Проверка, является ли текст медицинским термином
        
        Args:
            text: проверяемый текст
            
        Returns:
            bool: является ли текст медицинским термином
        """
        return text.lower() in self.terms
        
    def get_all_terms(self) -> Set[str]:
        """
        Получение всех медицинских терминов
        
        Returns:
            Set[str]: множество всех медицинских терминов
        """
        return self.terms.copy() 