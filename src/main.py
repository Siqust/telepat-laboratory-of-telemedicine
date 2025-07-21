# -*- coding: utf-8 -*-

import os
import asyncio
import logging
import shutil
from pathlib import Path
from loguru import logger
from processor import DocumentProcessor
from data_manager import DataManager
import cv2
import uuid
from deepseek_client import DeepSeekClient
from gigachat_client import GigaChatClient
from chatgpt_client import ChatGPTClient
import json

def setup_logging():
    """Настройка логирования для всего приложения"""
    logger.remove()  # Удаляем стандартный обработчик
    logger.add(
        "logs/processor_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg), level="DEBUG")
    
    # Создаем директорию для логов, если она не существует
    Path("logs").mkdir(exist_ok=True)

# Настраиваем логирование при импорте модуля
setup_logging()

async def process_documents(clinic_name: str, data_manager: DataManager) -> None:
    """
    Обрабатывает все документы в директории input
    
    Args:
        clinic_name: название клиники
        data_manager: менеджер данных
    """
    processor = await DocumentProcessor.create(data_manager=data_manager)
    
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
    
    # Создаем клиенты для API
    deepseek = DeepSeekClient()
    gigachat = GigaChatClient()
    chatgpt = ChatGPTClient()
    
    try:
        # Обрабатываем каждый файл последовательно
        for file_path in input_files:
            logger.info(f"Обработка файла: {file_path.name}")
            logger.debug(f"Полный путь к входному файлу: {file_path.absolute()}")
            
            try:
                # Сначала обрабатываем документ через DocumentProcessor
                output_path, extracted_data = await processor.process_document(
                    file_path=str(file_path),
                    clinic_name=clinic_name,
                    output_dir=str(output_dir)
                )
                
                if output_path and Path(output_path).exists():
                    logger.info(f"Файл успешно обработан и сохранен: {output_path}")
                    logger.debug(f"Полный путь к деперсонализированному файлу: {Path(output_path).absolute()}")
                    
                    # Проверяем размер файла
                    file_size = Path(output_path).stat().st_size
                    logger.debug(f"Размер деперсонализированного файла: {file_size} байт")
                    
                    # Затем отправляем деперсонализированный файл на анализ через все доступные API
                    logger.info(f"Отправка деперсонализированного файла на анализ: {output_path}")
                    
                    # Создаем директории для результатов
                    for dir_name in ['ai-result/deepseek', 'ai-result/gigachat', 'ai-result/chatgpt']:
                        Path(dir_name).mkdir(parents=True, exist_ok=True)
                    
                    # Получаем базовое имя файла без расширения
                    base_name = Path(output_path).stem
                    
                    # Пробуем анализ через DeepSeek
                    try:
                        logger.info(f"Начинаем анализ через DeepSeek: {output_path}")
                        deepseek_result = await deepseek.analyze_medical_report(output_path)
                        if deepseek_result:
                            output_file = Path(f"ai-result/deepseek/{base_name}_deepseek.json")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(deepseek_result, f, ensure_ascii=False, indent=2)
                            logger.info(f"Результат DeepSeek сохранен в {output_file}")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через DeepSeek: {str(e)}")
                    
                    # Пробуем анализ через GigaChat
                    try:
                        logger.info(f"Начинаем анализ через GigaChat: {output_path}")
                        gigachat_result = await gigachat.analyze_medical_report(output_path)
                        if gigachat_result:
                            output_file = Path(f"ai-result/gigachat/{base_name}_gigachat.json")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(gigachat_result, f, ensure_ascii=False, indent=2)
                            logger.info(f"Результат GigaChat сохранен в {output_file}")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через GigaChat: {str(e)}")
                    
                    # Пробуем анализ через ChatGPT
                    try:
                        logger.info(f"Начинаем анализ через ChatGPT: {output_path}")
                        chatgpt_result = await chatgpt.analyze_medical_report(
                            output_path,
                            prompt="Тебе будут передаваться файлы, на которых деперсонализированы персональные данные.людей путем закрашивания их черными квадратами. Считай за персональные данные только: адрес, ИНЗ, ИНН, ОМС, серия и номер паспорта, номер телефона, адрес электронной почты, имя, фамилия, отчество. Твоя задача определить, все ли персональные данные были деперсонализированы, или ты все еще можешь распознать какие-то личные данные. Если ты не можешь найти личных данных - ответь коротко: 'Личных данных нет.' Если можешь распознать - определи, что это за вид персональных данных и напиши: 'Распознано: (и тут перечисли какие виды персональных данных ты обнаружил. Не пиши Медведев Николай Владимирович, напиши просто: Фамилия, Имя, Отчество. Не указывай что именно ты распознал, укажи название Персональных данных.)'"
                        )
                        if chatgpt_result:
                            output_file = Path(f"ai-result/chatgpt/{base_name}_chatgpt.txt")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(chatgpt_result)
                            logger.info(f"Результат ChatGPT сохранен в {output_file}")
                            logger.debug(f"Содержимое ответа ChatGPT: {chatgpt_result[:200]}...")
                    except Exception as e:
                        logger.error(f"Ошибка при анализе через ChatGPT: {str(e)}")
                        logger.exception("Полный стек ошибки:")
                    
                    # Перемещаем обработанный файл в архив
                    archive_dir = input_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    shutil.move(file_path, archive_dir / file_path.name)
                    logger.info(f"Файл {file_path.name} перемещен в архив")
                else:
                    logger.error(f"Не удалось получить или найти деперсонализированный файл для {file_path.name}")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path.name}: {str(e)}")
                logger.exception("Полный стек ошибки:")
                continue
    finally:
        # Закрываем клиенты API, которые поддерживают закрытие
        try:
            if hasattr(gigachat, 'close'):
                await gigachat.close()
            if hasattr(chatgpt, 'close'):
                await chatgpt.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии клиентов API: {str(e)}")

async def process_medical_report(image_path: str):
    """Обрабатывает медицинский отчет через DeepSeek и GigaChat"""
    logger.debug(f"Начало process_medical_report с файлом: {image_path}")
    logger.debug(f"Полный путь к файлу: {Path(image_path).absolute()}")
    
    # Проверяем существование файла
    if not Path(image_path).exists():
        logger.error(f"Файл не найден: {image_path}")
        return {'deepseek': None, 'gigachat': None}
        
    # Проверяем размер файла
    file_size = Path(image_path).stat().st_size
    logger.debug(f"Размер файла для анализа: {file_size} байт")
    
    # Инициализируем клиенты
    deepseek = DeepSeekClient()
    gigachat = GigaChatClient()
    
    # Создаем директории для результатов
    for dir_name in ['ai-result/deepseek', 'ai-result/gigachat']:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Получаем базовое имя файла без расширения
    base_name = Path(image_path).stem
    logger.debug(f"Базовое имя файла: {base_name}")
    
    # Анализ через DeepSeek
    logger.info(f"Начинаем анализ через DeepSeek: {image_path}")
    deepseek_result = await deepseek.analyze_medical_report(image_path)
    
    if deepseek_result:
        # Сохраняем результат DeepSeek
        output_file = Path(f"ai-result/deepseek/{base_name}_deepseek.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deepseek_result, f, ensure_ascii=False, indent=2)
        logger.info(f"Результат DeepSeek сохранен в {output_file}")
    else:
        logger.error("Не удалось получить результат от DeepSeek")
    
    # Анализ через GigaChat
    logger.info(f"Начинаем анализ через GigaChat: {image_path}")
    logger.debug(f"Отправляем в GigaChat файл: {image_path}")
    gigachat_result = await gigachat.analyze_medical_report(image_path)
    
    if gigachat_result:
        # Результат GigaChat уже сохранен в методе analyze_medical_report
        logger.info("Анализ GigaChat завершен успешно")
    else:
        logger.error("Не удалось получить результат от GigaChat")
    
    return {
        'deepseek': deepseek_result,
        'gigachat': gigachat_result
    }

async def main():
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
        await process_documents(
            clinic_name="Default Clinic",
            data_manager=data_manager
        )
        
        # Сохраняем все данные перед завершением
        data_manager._save_data()
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении: {str(e)}")
        logger.exception("Полный стек ошибки:")
    finally:
        logger.info("Обработка завершена")

if __name__ == "__main__":
    asyncio.run(main()) 