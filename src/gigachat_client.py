import os
import json
import requests
import asyncio
from loguru import logger
from typing import Optional, Union, List
from dotenv import load_dotenv
from pathlib import Path
import uuid
import ssl
import urllib3
import base64

class GigaChatClient:
    """Клиент для взаимодействия с GigaChat API"""
    
    def __init__(self):
        """Инициализация клиента GigaChat"""
        # Читаем конфигурацию напрямую из файла
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gigachat_config.txt')
        logger.debug(f"Путь к файлу конфигурации: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            
            # Устанавливаем значения напрямую
            self.auth_key = config.get('GIGACHAT_AUTH_KEY')
            if not self.auth_key:
                raise ValueError("GIGACHAT_AUTH_KEY не найден в файле конфигурации")
                
            self.scope = config.get('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
            
            # Отладочный вывод
            logger.debug(f"Загруженный ключ GigaChat: {self.auth_key}")
            logger.debug(f"Загруженный scope: {self.scope}")
            
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при чтении файла конфигурации: {str(e)}")
            raise
            
        self.base_url = "https://gigachat-preview.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        # Инициализируем сессию
        self.session = requests.Session()
        # Отключаем проверку SSL для GigaChat, так как используется самоподписанный сертификат
        self.session.verify = False
        # Отключаем предупреждения о небезопасном SSL
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Проверяем доступность API при инициализации
        self.is_available = False
        self._access_token = None
        
        # Пробуем получить токен
        try:
            token = self._get_access_token()
            if token:
                self._access_token = token
                self.is_available = True
                logger.info("GigaChat API доступен и ключ валиден")
            else:
                logger.error("Не удалось получить токен доступа GigaChat")
        except Exception as e:
            logger.error(f"Ошибка при инициализации GigaChat API: {str(e)}")
            
    def _get_access_token(self) -> Optional[str]:
        """Получает токен доступа к API"""
        try:
            # Отладочный вывод
            logger.debug(f"Используем ключ для авторизации: {self.auth_key}")
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': str(uuid.uuid4()),
                'Authorization': f'Basic {self.auth_key}'
            }
            
            payload = {'scope': self.scope}
            
            logger.debug(f"Отправляем запрос на получение токена GigaChat. Scope: {self.scope}")
            logger.debug(f"Заголовки запроса: {headers}")
            
            response = self.session.post(
                self.auth_url,
                headers=headers,
                data=payload
            )
            
            if response.status_code == 200:
                token_data = response.json()
                logger.debug("Успешно получен токен GigaChat")
                return token_data['access_token']
            else:
                logger.error(f"Ошибка получения токена GigaChat. Статус: {response.status_code}, Ответ: {response.text}")
                if response.status_code == 401:
                    logger.error("Проверьте правильность API ключа в файле конфигурации")
                    logger.error(f"Текущий ключ: {self.auth_key}")
                elif response.status_code == 400:
                    logger.error("Проверьте формат API ключа в файле конфигурации")
                return None
                
        except requests.exceptions.SSLError as e:
            logger.error(f"Ошибка SSL при подключении к GigaChat: {str(e)}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ошибка подключения к GigaChat: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении токена GigaChat: {str(e)}")
            return None
            
    def _upload_file(self, image_path: str) -> Optional[str]:
        """Загружает файл в GigaChat"""
        try:
            if not self._access_token:
                self._access_token = self._get_access_token()
                if not self._access_token:
                    return None
                    
            url = f"{self.base_url}/files"
            name = Path(image_path).name
            
            # Проверяем существование файла
            if not os.path.exists(image_path):
                logger.error(f"Файл не найден: {image_path}")
                return None
                
            # Проверяем размер файла
            file_size = os.path.getsize(image_path)
            logger.debug(f"Размер файла {name}: {file_size} байт")
            
            # Проверяем тип файла
            mime_type = 'image/jpeg'  # По умолчанию
            if name.lower().endswith('.png'):
                mime_type = 'image/png'
            elif name.lower().endswith('.gif'):
                mime_type = 'image/gif'
            logger.debug(f"Тип файла {name}: {mime_type}")
            
            headers = {
                'Authorization': f'Bearer {self._access_token}',
                'Accept': 'application/json'
            }
            
            payload = {'purpose': 'general'}
            files = [
                ('file', (name, open(image_path, 'rb'), mime_type))
            ]
            
            logger.debug(f"Отправляем запрос на загрузку файла {name}")
            logger.debug(f"URL: {url}")
            logger.debug(f"Заголовки: {headers}")
            logger.debug(f"Payload: {payload}")
            
            response = self.session.post(
                url,
                headers=headers,
                data=payload,
                files=files
            )
            
            logger.debug(f"Статус ответа: {response.status_code}")
            logger.debug(f"Заголовки ответа: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Ответ сервера: {result}")
                return result['id']
            else:
                logger.error(f"Ошибка загрузки файла в GigaChat. Статус: {response.status_code}")
                logger.error(f"Ответ сервера: {response.text}")
                if response.status_code == 413:
                    logger.error("Файл слишком большой")
                elif response.status_code == 415:
                    logger.error("Неподдерживаемый тип файла")
                elif response.status_code == 401:
                    logger.error("Ошибка авторизации при загрузке файла")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла в GigaChat: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return None
            
    def _remove_file(self, file_id: str) -> bool:
        """Удаляет файл из GigaChat"""
        try:
            if not self._access_token:
                self._access_token = self._get_access_token()
                if not self._access_token:
                    return False
                    
            url = f"{self.base_url}/files/{file_id}/delete"
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self._access_token}'
            }
            
            response = self.session.post(
                url,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка удаления файла из GigaChat. Статус: {response.status_code}, Ответ: {response.text}")
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Ошибка при удалении файла из GigaChat: {str(e)}")
            return False
            
    async def analyze_medical_report(self, image_path: str) -> Union[str, None]:
        """
        Отправляет медицинский отчет (изображение) на анализ в GigaChat
        
        Args:
            image_path: путь к изображению медицинского отчета
            
        Returns:
            Union[str, None]: результат анализа в виде текста или None в случае ошибки
        """
        logger.debug(f"Начало анализа медицинского отчета: {image_path}")
        
        if not self.is_available:
            logger.error("GigaChat API недоступен. Проверьте API ключ и доступность сервиса.")
            return None

        if not Path(image_path).exists():
            logger.error(f"Файл изображения для анализа не найден: {image_path}")
            return None

        try:
            logger.debug("Начинаем загрузку файла в GigaChat")
            # Загружаем файл
            file_id = self._upload_file(image_path)
            if not file_id:
                logger.error("Не удалось загрузить файл в GigaChat")
                return None
                
            logger.debug(f"Файл успешно загружен, получен ID: {file_id}")
                
            try:
                # Отправляем запрос на анализ
                url = f"{self.base_url}/chat/completions"
                
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self._access_token}'
                }
                
                payload = {
                    "model": "GigaChat-Pro-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Проанализируй этот медицинский отчет. Извлеки ключевую информацию: диагноз, назначенные лекарства, процедуры, рекомендации. Предоставь структурированный ответ в формате JSON: {\"diagnosis\": [...], \"medications\": [...], \"procedures\": [...], \"recommendations\": [...]}",
                            "attachments": [file_id]
                        }
                    ],
                    "stream": False,
                    "update_interval": 0
                }
                
                logger.debug(f"Отправляем запрос на анализ в GigaChat")
                logger.debug(f"URL: {url}")
                logger.debug(f"Заголовки: {headers}")
                logger.debug(f"Payload: {payload}")
                
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                logger.debug(f"Получен ответ от GigaChat. Статус: {response.status_code}")
                logger.debug(f"Заголовки ответа: {dict(response.headers)}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Ответ сервера: {result}")
                    
                    # Сохраняем результат в файл
                    output_dir = Path("ai-result/gigachat")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{Path(image_path).stem}_gigachat.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"Результат анализа GigaChat сохранен в {output_file}")
                    return result
                else:
                    logger.error(f"Ошибка анализа в GigaChat. Статус: {response.status_code}")
                    logger.error(f"Ответ сервера: {response.text}")
                    return None
                    
            finally:
                # Удаляем файл в любом случае
                logger.debug(f"Удаляем файл из GigaChat: {file_id}")
                self._remove_file(file_id)
                
        except Exception as e:
            logger.error(f"Критическая ошибка при анализе GigaChat: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return None 