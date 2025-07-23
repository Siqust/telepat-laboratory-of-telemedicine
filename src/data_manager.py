"""
Модуль для работы с данными через pandas DataFrame
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger
from pathlib import Path

class DataManager:
    """Класс для работы с данными через pandas DataFrame"""
    
    def __init__(self, output_dir: str):
        """
        Инициализация менеджера данных
        
        Args:
            output_dir: директория для сохранения данных
        """
        self.output_dir = Path(output_dir)
        self.documents_df = pd.DataFrame(columns=[
            'document_id', 'original_filename', 'processed_filename',
            'processing_date', 'clinic_name', 'document_type',
            'birth_date', 'gender'
        ])
        
        self.sensitive_data_df = pd.DataFrame(columns=[
            'id', 'document_id', 'data_type', 'original_value', 'coordinates'
        ])
        
        self.patient_analyses_df = pd.DataFrame(columns=[
            'id', 'document_id', 'surname', 'name', 'analysis_id', 'created_at'
        ])
        
        # Создаем директорию если её нет
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_data(self):
        """Сохраняет все данные в CSV файлы"""
        try:
            # Сохраняем документы
            self.documents_df.to_csv(
                self.output_dir / 'documents.csv',
                index=False,
                encoding='utf-8'
            )
            
            # Сохраняем персональные данные
            self.sensitive_data_df.to_csv(
                self.output_dir / 'sensitive_data.csv',
                index=False,
                encoding='utf-8'
            )
            
            # Сохраняем связи пациентов с анализами
            self.patient_analyses_df.to_csv(
                self.output_dir / 'patient_analyses.csv',
                index=False,
                encoding='utf-8'
            )
            
            logger.info("Данные успешно сохранены в CSV файлы")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {str(e)}")
            raise
            
    def load_data(self):
        """Загружает данные из CSV файлов если они существуют"""
        try:
            # Загружаем документы
            docs_path = self.output_dir / 'documents.csv'
            if docs_path.exists():
                self.documents_df = pd.read_csv(docs_path, encoding='utf-8')
                
            # Загружаем персональные данные
            sens_path = self.output_dir / 'sensitive_data.csv'
            if sens_path.exists():
                self.sensitive_data_df = pd.read_csv(sens_path, encoding='utf-8')
                
            # Загружаем связи пациентов с анализами
            anal_path = self.output_dir / 'patient_analyses.csv'
            if anal_path.exists():
                self.patient_analyses_df = pd.read_csv(anal_path, encoding='utf-8')
                
            logger.info("Данные успешно загружены из CSV файлов")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise 