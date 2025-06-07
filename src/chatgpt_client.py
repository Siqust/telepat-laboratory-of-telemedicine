import os
import base64
import aiohttp
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
import aiohttp_socks

class ChatGPTClient:
    """Клиент для работы с ChatGPT API"""
    
    def __init__(self):
        """Инициализация клиента ChatGPT"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY не найден в переменных окружения")
        else:
            logger.info("OPENAI_API_KEY успешно загружен")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Настройки SOCKS5 прокси (порт по умолчанию для Hiddify)
        self.proxy_url = "socks5://127.0.0.1:12334"
        logger.info(f"Используется SOCKS5 прокси: {self.proxy_url}")
        logger.debug("ChatGPTClient инициализирован")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение или создание сессии aiohttp"""
        if self._session is None or self._session.closed:
            logger.debug("Создание новой сессии aiohttp с SOCKS5 прокси")
            connector = aiohttp_socks.ProxyConnector.from_url(self.proxy_url)
            self._session = aiohttp.ClientSession(connector=connector)
            logger.debug(f"Создана сессия с прокси: {self.proxy_url}")
        return self._session

    async def close(self):
        """Закрытие сессии"""
        if self._session and not self._session.closed:
            logger.debug("Закрытие сессии aiohttp")
            await self._session.close()

    def _encode_image(self, image_path: str) -> str:
        """Кодирование изображения в base64"""
        try:
            logger.debug(f"Начало кодирования изображения: {image_path}")
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                logger.debug(f"Размер изображения: {len(image_data)} байт")
                encoded = base64.b64encode(image_data).decode('utf-8')
                logger.debug(f"Изображение успешно закодировано в base64 (длина: {len(encoded)})")
                return encoded
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            raise

    async def analyze_medical_report(self, image_path: str, prompt: Optional[str] = None) -> Optional[str]:
        """
        Анализ медицинского отчета с помощью ChatGPT
        
        Args:
            image_path: путь к изображению
            prompt: опциональный промпт для анализа
            
        Returns:
            str: результат анализа или None в случае ошибки
        """
        logger.info(f"Начало анализа медицинского отчета: {image_path}")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY не настроен")
            return None

        try:
            # Проверяем существование файла
            if not Path(image_path).exists():
                logger.error(f"Файл не найден: {image_path}")
                return None
            logger.debug(f"Файл существует: {image_path}")

            # Кодируем изображение
            logger.debug("Начало кодирования изображения в base64")
            encoded_image = self._encode_image(image_path)
            logger.debug("Изображение успешно закодировано")
            
            # Формируем заголовки
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            logger.debug("Заголовки запроса сформированы")

            # Формируем промпт
            default_prompt = "К сообщению приложены изображения с результатами анализов. Выходят ли какие то параметры за реферсные значения? Ответь на первой строке да или нет, далее приведи список отклонений по одному на строке."
            prompt = prompt or default_prompt
            logger.debug(f"Используемый промпт: {prompt}")

            # Формируем payload
            payload = {
                "model": "gpt-4o",  # Обновленная модель для работы с изображениями
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            logger.debug("Payload сформирован")

            # Отправляем запрос
            logger.info("Отправка запроса к ChatGPT API через SOCKS5 прокси")
            session = await self._get_session()
            
            async with session.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=30  # Увеличиваем таймаут для работы через прокси
            ) as response:
                logger.debug(f"Получен ответ от API (статус: {response.status})")
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Успешно получен ответ от ChatGPT")
                    logger.debug(f"Содержимое ответа: {content[:100]}...")
                    return content
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка API ChatGPT (статус {response.status}): {error_text}")
                    logger.error(f"Заголовки ответа: {dict(response.headers)}")
                    if response.status == 403 and "unsupported_country_region_territory" in error_text:
                        logger.error("Доступ к API заблокирован. Проверьте работу SOCKS5 прокси.")
                    return None

        except Exception as e:
            logger.error(f"Ошибка при анализе через ChatGPT: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return None

    async def __aenter__(self):
        """Поддержка контекстного менеджера"""
        logger.debug("Вход в контекстный менеджер ChatGPTClient")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие сессии при выходе из контекста"""
        logger.debug("Выход из контекстного менеджера ChatGPTClient")
        await self.close() 