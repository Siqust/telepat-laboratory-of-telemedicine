import os
import json
import aiohttp
import asyncio
from loguru import logger
from typing import Optional, Union
from dotenv import load_dotenv
from pathlib import Path
import ssl
from PIL import Image
import io
import base64

class DeepSeekClient:
    """Клиент для взаимодействия с DeepSeek API"""
    
    def __init__(self):
        """Инициализация клиента DeepSeek"""
        load_dotenv()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        # Обновленный URL API
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
        if not self.api_key:
            logger.error("API ключ DeepSeek не найден в переменных окружения (DEEPSEEK_API_KEY)")
            self.is_available = False
            return
            
        # Проверяем доступность API при инициализации
        self.is_available = self._check_api_availability()
        
    async def _check_api_availability(self) -> bool:
        """Проверяет доступность API и валидность ключа"""
        try:
            # Настройки SSL и таймаута для aiohttp
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                # Простой запрос для проверки доступности API
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    if response.status == 200:
                        logger.info("DeepSeek API доступен и ключ валиден")
                        return True
                    elif response.status == 401:
                        logger.error("Неверный API ключ DeepSeek")
                        return False
                    else:
                        logger.error(f"DeepSeek API недоступен. Статус: {response.status}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при проверке доступности DeepSeek API: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке DeepSeek API: {str(e)}")
            return False

    async def analyze_medical_report(self, image_path: str) -> Union[str, None]:
        """
        Отправляет медицинский отчет (изображение) на анализ в DeepSeek
        
        Args:
            image_path: путь к изображению медицинского отчета
            
        Returns:
            Union[str, None]: результат анализа в виде текста или None в случае ошибки
        """
        if not self.is_available:
            logger.error("DeepSeek API недоступен. Проверьте API ключ и доступность сервиса.")
            return None

        if not Path(image_path).exists():
            logger.error(f"Файл изображения для анализа не найден: {image_path}")
            return None

        try:
            logger.info(f"Начинаем подготовку изображения для DeepSeek: {image_path}")
            # Проверяем размер файла
            file_size = Path(image_path).stat().st_size
            logger.info(f"Размер файла: {file_size / 1024:.2f} KB")
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning("Файл слишком большой для отправки в DeepSeek API (>10MB)")
                return None

            # Настройки SSL и таймаута для aiohttp
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=60)  # Увеличиваем таймаут до 60 секунд
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                try:
                    # Подготавливаем изображение в правильном формате
                    logger.info("Подготовка изображения для API...")
                    with Image.open(image_path) as img:
                        # Уменьшаем размер изображения если оно слишком большое
                        max_size = 1024  # Максимальный размер стороны
                        if max(img.size) > max_size:
                            ratio = max_size / max(img.size)
                            new_size = tuple(int(dim * ratio) for dim in img.size)
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Конвертируем в JPEG с оптимальным качеством
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85, optimize=True)
                        image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                    
                    logger.info(f"Изображение подготовлено (размер base64: {len(base64_image)} символов)")

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }

                    # Упрощенный формат запроса
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Проанализируй этот медицинский отчет. Извлеки ключевую информацию: диагноз, назначенные лекарства, процедуры, рекомендации. Предоставь структурированный ответ в формате JSON: {\"diagnosis\": [...], \"medications\": [...], \"procedures\": [...], \"recommendations\": [...]}",
                                "images": [f"data:image/jpeg;base64,{base64_image}"]
                            }
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    }

                    logger.info("Отправляем запрос в DeepSeek API...")
                    logger.debug(f"URL запроса: {self.base_url}")
                    logger.debug(f"Используемая модель: {payload['model']}")
                    logger.debug(f"Размер payload: {len(str(payload))} байт")
                    
                    # Проверяем размер base64 изображения
                    base64_size = len(base64_image)
                    if base64_size > 10 * 1024 * 1024:  # 10MB
                        logger.error(f"Размер base64 изображения слишком большой: {base64_size / 1024 / 1024:.2f} MB")
                        return None
                    
                    async with session.post(self.base_url, headers=headers, json=payload) as response:
                        response_text = await response.text()
                        logger.info(f"Получен ответ от DeepSeek. Статус: {response.status}, Причина: {response.reason}")
                        
                        if response.status != 200:
                            logger.error(f"Тело ответа с ошибкой: {response_text}")
                            # Пробуем получить более подробную информацию об ошибке
                            try:
                                error_data = json.loads(response_text)
                                if 'error' in error_data:
                                    logger.error(f"Детали ошибки API: {error_data['error']}")
                            except:
                                pass
                            return None

                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка HTTP запроса к DeepSeek API: {str(e)}")
                    return None
                except asyncio.TimeoutError:
                    logger.error("Превышено время ожидания ответа от DeepSeek API (30 секунд)")
                    return None
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при работе с DeepSeek API: {str(e)}")
                    return None

        except FileNotFoundError:
            logger.error(f"Файл не найден при чтении для base64 кодирования: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Критическая ошибка при анализе DeepSeek: {str(e)}")
            return None

    async def analyze_multiple_medical_reports(self, image_paths: list) -> Union[dict, None]:
        """
        Отправляет несколько медицинских отчетов (изображений) на анализ в DeepSeek одним запросом
        
        Args:
            image_paths: список путей к изображениям медицинских отчетов
            
        Returns:
            Union[dict, None]: результат анализа в виде словаря или None в случае ошибки
        """
        if not self.is_available:
            logger.error("DeepSeek API недоступен. Проверьте API ключ и доступность сервиса.")
            return None

        if not image_paths:
            logger.error("Список путей к изображениям пуст")
            return None

        try:
            logger.info(f"Начинаем подготовку {len(image_paths)} изображений для DeepSeek")
            
            # Настройки SSL и таймаута для aiohttp
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=120)  # Увеличиваем таймаут для множественных файлов
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                try:
                    # Подготавливаем все изображения
                    base64_images = []
                    total_size = 0
                    
                    for i, image_path in enumerate(image_paths):
                        if not Path(image_path).exists():
                            logger.warning(f"Файл изображения не найден: {image_path}")
                            continue
                            
                        # Проверяем размер файла
                        file_size = Path(image_path).stat().st_size
                        if file_size > 5 * 1024 * 1024:  # 5MB на файл
                            logger.warning(f"Файл слишком большой: {image_path} ({file_size / 1024:.2f} KB)")
                            continue

                        # Подготавливаем изображение
                        with Image.open(image_path) as img:
                            # Уменьшаем размер изображения если оно слишком большое
                            max_size = 800  # Уменьшаем максимальный размер для множественных файлов
                            if max(img.size) > max_size:
                                ratio = max_size / max(img.size)
                                new_size = tuple(int(dim * ratio) for dim in img.size)
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # Конвертируем в JPEG с оптимальным качеством
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=80, optimize=True)
                            image_data = buffer.getvalue()
                            base64_image = base64.b64encode(image_data).decode("utf-8")
                            
                            base64_images.append(f"data:image/jpeg;base64,{base64_image}")
                            total_size += len(base64_image)
                            
                            logger.info(f"Изображение {i+1} подготовлено (размер base64: {len(base64_image)} символов)")

                    if not base64_images:
                        logger.error("Не удалось подготовить ни одного изображения")
                        return None

                    # Проверяем общий размер
                    if total_size > 20 * 1024 * 1024:  # 20MB общий лимит
                        logger.error(f"Общий размер изображений слишком большой: {total_size / 1024 / 1024:.2f} MB")
                        return None

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }

                    # Формируем контент с несколькими изображениями
                    content_parts = [
                        {
                            "type": "text",
                            "text": "Проанализируй эти медицинские отчеты. Извлеки ключевую информацию из всех страниц: диагнозы, назначенные лекарства, процедуры, рекомендации. Предоставь структурированный ответ в формате JSON: {\"diagnosis\": [...], \"medications\": [...], \"procedures\": [...], \"recommendations\": [...], \"summary\": \"общая сводка по всем страницам\"}"
                        }
                    ]
                    
                    # Добавляем все изображения
                    for base64_image in base64_images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        })

                    payload = {
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "user",
                                "content": content_parts
                            }
                        ],
                        "max_tokens": 3000,  # Увеличиваем лимит токенов для множественных файлов
                        "temperature": 0.7
                    }

                    logger.info(f"Отправляем запрос в DeepSeek API с {len(base64_images)} изображениями...")
                    logger.debug(f"URL запроса: {self.base_url}")
                    logger.debug(f"Используемая модель: {payload['model']}")
                    logger.debug(f"Общий размер payload: {len(str(payload))} байт")
                    
                    async with session.post(self.base_url, headers=headers, json=payload) as response:
                        response_text = await response.text()
                        logger.info(f"Получен ответ от DeepSeek. Статус: {response.status}, Причина: {response.reason}")
                        
                        if response.status != 200:
                            logger.error(f"Тело ответа с ошибкой: {response_text}")
                            try:
                                error_data = json.loads(response_text)
                                if 'error' in error_data:
                                    logger.error(f"Детали ошибки API: {error_data['error']}")
                            except:
                                pass
                            return None

                        # Парсим ответ
                        try:
                            response_data = json.loads(response_text)
                            if 'choices' in response_data and len(response_data['choices']) > 0:
                                content = response_data['choices'][0]['message']['content']
                                
                                # Пытаемся извлечь JSON из ответа
                                try:
                                    # Ищем JSON в ответе
                                    start_idx = content.find('{')
                                    end_idx = content.rfind('}') + 1
                                    if start_idx != -1 and end_idx > start_idx:
                                        json_str = content[start_idx:end_idx]
                                        result = json.loads(json_str)
                                        logger.info("Успешно извлечен JSON из ответа DeepSeek")
                                        return result
                                    else:
                                        # Если JSON не найден, возвращаем весь текст
                                        logger.warning("JSON не найден в ответе, возвращаем текст")
                                        return {"raw_response": content}
                                except json.JSONDecodeError:
                                    logger.warning("Не удалось распарсить JSON из ответа, возвращаем текст")
                                    return {"raw_response": content}
                            else:
                                logger.error("Неожиданная структура ответа от DeepSeek API")
                                return None
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка при парсинге JSON ответа: {str(e)}")
                            return None

                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка HTTP запроса к DeepSeek API: {str(e)}")
                    return None
                except asyncio.TimeoutError:
                    logger.error("Превышено время ожидания ответа от DeepSeek API (120 секунд)")
                    return None
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при работе с DeepSeek API: {str(e)}")
                    return None

        except Exception as e:
            logger.error(f"Критическая ошибка при анализе множественных отчетов DeepSeek: {str(e)}")
            return None 