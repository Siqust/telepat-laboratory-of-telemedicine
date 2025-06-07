# Лаборатория Телемедицины - Система обработки медицинских документов

Система для автоматической обработки медицинских документов с использованием искусственного интеллекта. Проект предназначен для деперсонализации медицинских документов и их последующего анализа с помощью моделей DeepSeek, GigaChat и ChatGPT.

## Основные возможности

- Деперсонализация медицинских документов (маскирование персональных данных)
- Распознавание текста с медицинских документов (OCR)
- Анализ медицинских документов с помощью нескольких AI моделей:
  - DeepSeek AI
  - GigaChat
  - ChatGPT
- Автоматическое определение отклонений от референсных значений в медицинских анализах
- Сохранение результатов анализа в структурированном виде (JSON и TXT)
- Поддержка различных форматов документов (PDF, JPG, PNG, TIFF, BMP)
- Автоматическое архивирование обработанных документов
- Подробное логирование всех операций

## Структура проекта

```
project/
├── src/                    # Исходный код
│   ├── __init__.py
│   ├── main.py            # Основной модуль запуска
│   ├── processor.py       # Обработка документов
│   ├── deepseek_client.py # Клиент DeepSeek API
│   ├── gigachat_client.py # Клиент GigaChat API
│   ├── chatgpt_client.py  # Клиент ChatGPT API
│   ├── data_manager.py    # Управление данными
│   └── utils.py           # Вспомогательные функции
├── input/                 # Входная директория для документов
│   └── archive/          # Архив обработанных документов
├── output/               # Директория для деперсонализированных документов
├── ai-result/           # Результаты анализа AI
│   ├── deepseek/        # Результаты DeepSeek (JSON)
│   ├── gigachat/        # Результаты GigaChat (JSON)
│   └── chatgpt/         # Результаты ChatGPT (TXT)
├── data/                # Данные для обработки
│   ├── medical_terms/   # Медицинские термины
│   └── russian-words/   # Словари для русского языка
├── logs/                # Логи работы системы
└── tests/              # Тесты
```

## Требования

- Python 3.8+
- Tesseract OCR
- OpenCV (cv2)
- Доступ к API:
  - DeepSeek
  - GigaChat
  - ChatGPT (OpenAI API)

## Установка

1. Клонируйте репозиторий:
```bash
git clone [url-репозитория]
cd telepat-laboratory-of-telemedicine
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Установите Tesseract OCR:

Для Windows:
- Скачайте установщик с https://github.com/UB-Mannheim/tesseract/wiki
- Установите, следуя инструкциям установщика
- Добавьте путь к Tesseract в переменную PATH

Для Linux:

Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus  # для русского языка
```

Fedora:
```bash
sudo dnf install tesseract
sudo dnf install tesseract-langpack-rus  # для русского языка
```

Arch Linux:
```bash
sudo pacman -S tesseract
sudo pacman -S tesseract-data-rus  # для русского языка
```

openSUSE:
```bash
sudo zypper install tesseract-ocr
sudo zypper install tesseract-ocr-langpack-rus  # для русского языка
```

Для macOS:
```bash
brew install tesseract
brew install tesseract-lang  # включает русский язык
```

5. Настройте доступ к API:
- Создайте файл `gigachat_config.txt` с токеном GigaChat
- Настройте доступ к DeepSeek API (инструкции в документации API)
- Создайте файл с API ключом OpenAI для ChatGPT

## Использование

1. Поместите медицинские документы в директорию `input/`

2. Запустите обработку:
```bash
python -m src.main
```

Система выполнит следующие шаги:
1. Деперсонализация документов (маскирование персональных данных)
2. Сохранение деперсонализированных документов в `output/`
3. Отправка документов на анализ в:
   - DeepSeek (структурированный JSON)
   - GigaChat (структурированный JSON)
   - ChatGPT (текстовый анализ отклонений)
4. Сохранение результатов анализа в соответствующих поддиректориях `ai-result/`
5. Перемещение обработанных документов в `input/archive/`

## Результаты

- Деперсонализированные документы: `output/`
- Результаты анализа:
  - DeepSeek: `ai-result/deepseek/` (JSON)
  - GigaChat: `ai-result/gigachat/` (JSON)
  - ChatGPT: `ai-result/chatgpt/` (TXT)
- Логи работы: `logs/` (ротация каждые 24 часа, хранение 7 дней)
- Архив обработанных документов: `input/archive/`

## Безопасность

- Все персональные данные маскируются перед анализом
- Деперсонализированные документы не содержат персональной информации
- Доступ к API защищен токенами
- Все операции логируются для аудита
- Автоматическое архивирование исходных документов
- Ротация логов с ограниченным сроком хранения

## Поддержка

При возникновении проблем:
1. Проверьте логи в директории `logs/`
2. Убедитесь, что все API ключи настроены правильно
3. Проверьте наличие и доступность Tesseract OCR
4. Убедитесь, что входные документы в поддерживаемых форматах
