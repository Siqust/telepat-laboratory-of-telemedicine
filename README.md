# Система деперсонализации медицинских документов

Система для автоматической деперсонализации медицинских документов с сохранением связи между оригиналом и обработанной версией.

## Требования

- Python 3.8+
- Tesseract OCR
- PostgreSQL (опционально)

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите Tesseract OCR:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

4. Создайте файл .env в корневой директории проекта:
```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

## Структура проекта

```
project/
├── src/
│   ├── __init__.py
│   ├── processor.py        # Основной класс обработки
│   ├── batch.py           # Пакетная обработка
│   ├── database.py        # Работа с БД
│   └── utils.py           # Вспомогательные функции
├── tests/
│   └── test_processor.py
├── input_documents/       # Входная директория
├── processed_documents/   # Выходная директория
└── requirements.txt
```

## Использование

1. Поместите документы для обработки в директорию `input_documents/`
2. Запустите обработку:
```bash
python -m src.batch --input input_documents --output processed_documents --clinic "Название клиники"
```

## Лицензия

MIT