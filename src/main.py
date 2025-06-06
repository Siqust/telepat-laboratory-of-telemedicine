import os
import asyncio
from pathlib import Path
from loguru import logger
from processor import DocumentProcessor
from data_manager import DataManager

async def process_documents(clinic_name: str, data_manager: DataManager) -> None:
    """
    Обрабатывает все документы в директории input
    
    Args:
        clinic_name: название клиники
        data_manager: менеджер данных
    """
    processor = DocumentProcessor(data_manager=data_manager)
    
    # Определяем статические пути
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Создаем выходную директорию, если она не существует
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех файлов в директории
    input_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')]
    
    if not input_files:
        logger.warning("В директории input не найдено изображений")
        return
        
    logger.info(f"Найдено {len(input_files)} файлов для обработки")
    
    # Обрабатываем каждый файл последовательно
    for file_path in input_files:
        logger.info(f"Обработка файла: {file_path.name}")
        
        try:
            output_path, extracted_data = await processor.process_document(
                file_path=str(file_path),
                clinic_name=clinic_name,
                output_dir=str(output_dir)
            )
            logger.info(f"Файл успешно обработан: {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path.name}: {str(e)}")
            continue

def main():
    # Настраиваем логирование
    logger.remove()  # Удаляем стандартный обработчик
    logger.add(
        "logs/processor_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg), level="INFO")  # Добавляем вывод в консоль
    
    # Создаем директорию для логов, если она не существует
    Path("logs").mkdir(exist_ok=True)
    
    # Проверяем существование входной директории
    if not Path("input").exists():
        logger.error("Директория input не существует")
        return
    
    # Инициализируем менеджер данных
    data_manager = DataManager(output_dir="output")
    
    # Загружаем существующие данные, если они есть
    data_manager.load_data()
    
    try:
        # Запускаем обработку документов
        asyncio.run(process_documents(
            clinic_name="Default Clinic",
            data_manager=data_manager
        ))
        
        # Сохраняем все данные перед завершением
        data_manager._save_data()
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise
    finally:
        logger.info("Обработка завершена")

if __name__ == "__main__":
    main() 