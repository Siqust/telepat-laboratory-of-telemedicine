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
        
    def store_document(self, 
                      document_id: str,
                      original_filename: str,
                      processed_filename: str,
                      clinic_name: str,
                      document_type: Optional[str] = None,
                      birth_date: Optional[str] = None,
                      gender: Optional[str] = None) -> None:
        """
        Сохраняет информацию о документе
        
        Args:
            document_id: уникальный идентификатор документа
            original_filename: имя оригинального файла
            processed_filename: имя обработанного файла
            clinic_name: название клиники
            document_type: тип документа (опционально)
            birth_date: дата рождения (опционально)
            gender: пол (опционально)
        """
        new_row = pd.DataFrame([{
            'document_id': document_id,
            'original_filename': original_filename,
            'processed_filename': processed_filename,
            'processing_date': datetime.utcnow(),
            'clinic_name': clinic_name,
            'document_type': document_type,
            'birth_date': birth_date,
            'gender': gender
        }])
        
        self.documents_df = pd.concat([self.documents_df, new_row], ignore_index=True)
        self._save_data()
        
    def store_sensitive_data(self,
                           document_id: str,
                           data_type: str,
                           original_value: str,
                           coordinates: Dict) -> None:
        """
        Сохраняет информацию о персональных данных
        
        Args:
            document_id: идентификатор документа
            data_type: тип данных
            original_value: оригинальное значение
            coordinates: координаты на изображении
        """
        new_row = pd.DataFrame([{
            'id': f"{document_id}_{data_type}",
            'document_id': document_id,
            'data_type': data_type,
            'original_value': original_value,
            'coordinates': str(coordinates)  # Сохраняем как строку
        }])
        
        self.sensitive_data_df = pd.concat([self.sensitive_data_df, new_row], ignore_index=True)
        self._save_data()
        
    def add_patient_analysis(self, document_id: str, surname: str, name: str, analysis_id: str) -> None:
        """
        Добавляет запись о связи пациента с анализом
        
        Args:
            document_id: ID документа
            surname: Фамилия пациента
            name: Имя пациента
            analysis_id: ID анализа
        """
        new_row = pd.DataFrame([{
            'id': len(self.patient_analyses_df) + 1,  # Простой автоинкремент
            'document_id': document_id,
            'surname': surname.lower(),
            'name': name.lower(),
            'analysis_id': analysis_id,
            'created_at': datetime.utcnow()
        }])
        
        self.patient_analyses_df = pd.concat([self.patient_analyses_df, new_row], ignore_index=True)
        self._save_data()
        
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