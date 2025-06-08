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

class GptClient:
    """Клиент для взаимодействия с GPT API"""
    
    def __init__(self):
        """Инициализация клиента GPT"""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY') # Используем OPENAI_API_KEY для GPT
        self.base_url = "https://api.openai.com/v1/chat/completions" # URL для OpenAI API
        
        if not self.api_key:
            logger.error("API ключ OpenAI не найден в переменных окружения (OPENAI_API_KEY)")
            self.is_available = False
            return
            
        # Проверка будет выполняться только при первом запросе, чтобы избежать ошибок с asyncio.run в __init__
        self.is_available = True # Временно считаем доступным до первой проверки
        
    async def _check_api_availability(self) -> bool:
        """Проверяет доступность API и валидность ключа"""
        try:
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
                async with session.get("https://api.openai.com/v1/models", headers=headers) as response: # URL для списка моделей OpenAI
                    if response.status == 200:
                        logger.info("GPT (OpenAI) API доступен и ключ валиден")
                        return True
                    elif response.status == 401:
                        logger.error("Неверный API ключ OpenAI")
                        return False
                    else:
                        logger.error(f"GPT (OpenAI) API недоступен. Статус: {response.status}")
                        return False
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при проверке доступности GPT (OpenAI) API: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке GPT (OpenAI) API: {str(e)}")
            return False

    async def analyze_medical_report(self, image_path: str) -> Union[str, None]:
        """
        Отправляет медицинский отчет (изображение) на анализ в GPT (OpenAI)
        
        Args:
            image_path: путь к изображению медицинского отчета
            
        Returns:
            Union[str, None]: результат анализа в виде текста или None в случае ошибки
        """
        # Проверяем доступность API при первом запросе
        if not self.is_available:
            self.is_available = await self._check_api_availability()
            if not self.is_available:
                logger.error("GPT (OpenAI) API недоступен. Проверьте API ключ и доступность сервиса.")
                return None

        if not Path(image_path).exists():
            logger.error(f"Файл изображения для анализа не найден: {image_path}")
            return None

        try:
            logger.info(f"Начинаем подготовку изображения для GPT (OpenAI): {image_path}")
            file_size = Path(image_path).stat().st_size
            logger.info(f"Размер файла: {file_size / 1024:.2f} KB")
            
            if file_size > 20 * 1024 * 1024:  # GPT Vision может принимать до 20MB, но лучше быть осторожным
                logger.warning("Файл слишком большой для отправки в GPT (OpenAI) API (>20MB)")
                return None

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=90)  # Увеличиваем таймаут
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                try:
                    logger.info("Подготовка изображения для API...")
                    with Image.open(image_path) as img:
                        # GPT Vision лучше работает с изображениями до 2048x2048
                        max_dim = 2048
                        if max(img.size) > max_dim:
                            ratio = max_dim / max(img.size)
                            new_size = tuple(int(dim * ratio) for dim in img.size)
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85, optimize=True)
                        image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                    
                    logger.info(f"Изображение подготовлено (размер base64: {len(base64_image)} символов)")

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }

                    # Используем необработанную строковую литералу с тройными кавычками для JSON-промпта
                    json_prompt_text = r"""Проанализируй этот медицинский отчет. Извлеки ключевую информацию: диагноз, назначенные лекарства, процедуры, рекомендации. Предоставь структурированный ответ в формате JSON: {"diagnosis": [...], "medications": [...], "procedures": [...], "recommendations": []}"""

                    payload = {
                        "model": "gpt-4o", # Используем gpt-4o, последнюю модель с хорошим зрением
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json_prompt_text
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    }

                    logger.info("Отправляем запрос в GPT (OpenAI) API...")
                    logger.debug(f"URL запроса: {self.base_url}")
                    logger.debug(f"Используемая модель: {payload['model']}")
                    logger.debug(f"Размер payload: {len(str(payload))} байт")
                    
                    base64_size = len(base64_image)
                    if base64_size > 20 * 1024 * 1024:
                        logger.error(f"Размер base64 изображения слишком большой: {base64_size / 1024 / 1024:.2f} MB")
                        return None
                    
                    async with session.post(self.base_url, headers=headers, json=payload) as response:
                        response_text = await response.text()
                        logger.info(f"Получен ответ от GPT (OpenAI). Статус: {response.status}, Причина: {response.reason}")
                        
                        if response.status != 200:
                            logger.error(f"Тело ответа с ошибкой: {response_text}")
                            try:
                                error_data = json.loads(response_text)
                                if 'error' in error_data:
                                    logger.error(f"Детали ошибки API: {error_data['error']}")
                            except:
                                pass
                            return None

                        try:
                            response_json = json.loads(response_text)
                            # Извлекаем контент из ответа
                            if 'choices' in response_json and len(response_json['choices']) > 0:
                                message_content = response_json['choices'][0]['message']['content']
                                # OpenAI может возвращать JSON в виде строки внутри markdown блока
                                # Попробуем извлечь JSON из строки
                                if '```json' in message_content:
                                    json_start = message_content.find('```json') + len('```json')
                                    json_end = message_content.find('```', json_start)
                                    if json_start != -1 and json_end != -1:
                                        json_str = message_content[json_start:json_end].strip()
                                        return json_str
                                else:
                                    # Если нет markdown, пробуем распарсить напрямую
                                    return message_content
                            return None
                        except json.JSONDecodeError:
                            logger.error(f"Ошибка при парсинге JSON ответа от GPT (OpenAI): {response_text}")
                            return None

                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка HTTP запроса к GPT (OpenAI) API: {str(e)}")
                    return None
                except asyncio.TimeoutError:
                    logger.error("Превышено время ожидания ответа от GPT (OpenAI) API")
                    return None
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при работе с GPT (OpenAI) API: {str(e)}")
                    return None

        except FileNotFoundError:
            logger.error(f"Файл не найден при чтении для base64 кодирования: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Критическая ошибка при анализе GPT (OpenAI): {str(e)}")
            return None

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def send_request(self):
        # Implementation of send_request method
        pass

    def receive_response(self):
        # Implementation of receive_response method
        pass 