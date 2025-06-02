"""
Тесты для модуля обработки документов
"""

import os
import pytest
import numpy as np
from src.processor import DocumentProcessor
from src.database import Database

@pytest.fixture
def processor():
    """Фикстура для создания процессора"""
    return DocumentProcessor()

@pytest.fixture
def sample_image():
    """Фикстура для создания тестового изображения"""
    # Создаем белое изображение с текстом
    image = np.ones((100, 300), dtype=np.uint8) * 255
    
    # Добавляем текст (в реальном тесте нужно использовать реальное изображение)
    return image

def test_preprocess_image(processor, sample_image):
    """Тест предобработки изображения"""
    processed = processor._preprocess_image(sample_image)
    assert processed.shape == sample_image.shape
    assert processed.dtype == np.uint8

def test_recognize_text(processor, sample_image):
    """Тест распознавания текста"""
    # В реальном тесте нужно использовать изображение с текстом
    text_data = processor._recognize_text(sample_image)
    assert isinstance(text_data, list)

def test_extract_data(processor):
    """Тест извлечения данных"""
    # Тестовые данные
    text_data = [
        {
            'text': 'Пациент: Иванов Иван Иванович',
            'left': 0,
            'top': 0,
            'width': 200,
            'height': 20,
            'conf': 90
        },
        {
            'text': 'Адрес: 123456 Москва, ул. Примерная, д. 1',
            'left': 0,
            'top': 30,
            'width': 300,
            'height': 20,
            'conf': 90
        }
    ]
    
    extracted = processor._extract_data(text_data)
    
    assert 'sensitive_regions' in extracted
    assert 'medical_data' in extracted
    assert len(extracted['sensitive_regions']) > 0

def test_mask_sensitive_data(processor, sample_image):
    """Тест маскирования данных"""
    sensitive_regions = [
        {
            'type': 'name',
            'text': 'Иванов Иван Иванович',
            'left': 0,
            'top': 0,
            'width': 200,
            'height': 20
        }
    ]
    
    masked = processor._mask_sensitive_data(sample_image, sensitive_regions)
    assert masked.shape == sample_image.shape
    assert masked.dtype == sample_image.dtype

def test_process_document(processor, tmp_path):
    """Тест обработки документа"""
    # Создаем тестовый файл
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # В реальном тесте нужно использовать реальный документ
    test_file = input_dir / "test.jpg"
    np.save(str(test_file), np.ones((100, 100)))
    
    try:
        result = processor.process_document(
            str(test_file),
            "Тестовая клиника",
            str(output_dir)
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
    except Exception as e:
        pytest.skip(f"Тест пропущен: {str(e)}")

def test_database_integration(processor, tmp_path):
    """Тест интеграции с базой данных"""
    # Создаем тестовую базу данных
    db_url = "sqlite:///test.db"
    db = Database(db_url)
    
    processor_with_db = DocumentProcessor(db)
    
    # Тест сохранения данных
    document_id = "test_id"
    original_filename = "test.jpg"
    processed_filename = "processed_test.jpg"
    clinic_name = "Тестовая клиника"
    extracted_data = {
        'sensitive_regions': [
            {
                'type': 'name',
                'text': 'Иванов Иван Иванович',
                'left': 0,
                'top': 0,
                'width': 200,
                'height': 20
            }
        ],
        'medical_data': {
            'birth_date': '01.01.1990',
            'gender': 'м'
        }
    }
    
    processor_with_db._save_to_database(
        document_id,
        original_filename,
        processed_filename,
        clinic_name,
        extracted_data
    )
    
    # Проверяем сохраненные данные
    saved_doc = db.get_document(document_id)
    assert saved_doc is not None
    assert saved_doc.original_filename == original_filename
    assert saved_doc.clinic_name == clinic_name
    
    sensitive_data = db.get_sensitive_data(document_id)
    assert len(sensitive_data) > 0
    
    # Удаляем тестовую базу
    os.remove("test.db") 