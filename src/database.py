"""
Модуль для работы с базой данных
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Создаем базовый класс для моделей
Base = declarative_base()

class Document(Base):
    """Модель для хранения информации о документах"""
    __tablename__ = 'documents'
    
    document_id = Column(String, primary_key=True)
    original_filename = Column(String, nullable=False)
    processed_filename = Column(String, nullable=False)
    processing_date = Column(DateTime, default=datetime.utcnow)
    clinic_name = Column(String, nullable=False)
    document_type = Column(String)
    birth_date = Column(String)  # Сохраняем как строку для гибкости
    gender = Column(String)
    
    # Связь с персональными данными
    sensitive_data = relationship("SensitiveData", back_populates="document")

class SensitiveData(Base):
    """Модель для хранения персональных данных"""
    __tablename__ = 'sensitive_data'
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.document_id'))
    data_type = Column(String, nullable=False)  # тип данных (паспорт, адрес, ФИО)
    original_value = Column(String, nullable=False)
    coordinates = Column(JSON)  # координаты на изображении
    
    # Связь с документом
    document = relationship("Document", back_populates="sensitive_data")

class Database:
    """Класс для работы с базой данных"""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Инициализация подключения к базе данных
        
        Args:
            db_url: URL подключения к базе данных (если None, берется из переменных окружения)
        """
        if db_url is None:
            db_url = os.getenv('DATABASE_URL')
            if db_url is None:
                raise ValueError("Не указан URL базы данных")
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Создаем таблицы если их нет
        Base.metadata.create_all(self.engine)
    
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
        session = self.Session()
        try:
            document = Document(
                document_id=document_id,
                original_filename=original_filename,
                processed_filename=processed_filename,
                clinic_name=clinic_name,
                document_type=document_type,
                birth_date=birth_date,
                gender=gender
            )
            session.add(document)
            session.commit()
        finally:
            session.close()
    
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
        session = self.Session()
        try:
            sensitive_data = SensitiveData(
                id=f"{document_id}_{data_type}",
                document_id=document_id,
                data_type=data_type,
                original_value=original_value,
                coordinates=coordinates
            )
            session.add(sensitive_data)
            session.commit()
        finally:
            session.close()
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Получает информацию о документе
        
        Args:
            document_id: идентификатор документа
            
        Returns:
            Optional[Document]: информация о документе или None
        """
        session = self.Session()
        try:
            return session.query(Document).filter_by(document_id=document_id).first()
        finally:
            session.close()
    
    def get_sensitive_data(self, document_id: str) -> List[SensitiveData]:
        """
        Получает персональные данные документа
        
        Args:
            document_id: идентификатор документа
            
        Returns:
            List[SensitiveData]: список персональных данных
        """
        session = self.Session()
        try:
            return session.query(SensitiveData).filter_by(document_id=document_id).all()
        finally:
            session.close() 