import os
import json
import aiohttp
import asyncio
from loguru import logger
from typing import Optional
from dotenv import load_dotenv

class DeepSeekClient:
    """Класс для работы с DeepSeek API"""
    
    def __init__(self):
        """Инициализация клиента DeepSeek"""
        load_dotenv()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY не найден в переменных окружения")
            
        self.api_url = "https://api.deepseek.com/v1/chat/completions"  # Замените на реальный URL API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    async def analyze_medical_report(self, image_path: str) -> Optional[str]:
        """
        Анализирует медицинский отчет с помощью DeepSeek API
        
        Args:
            image_path: путь к изображению для анализа
            
        Returns:
            str: результат анализа или None в случае ошибки
        """
        try:
            # Читаем изображение и конвертируем его в base64
            with open(image_path, 'rb') as image_file:
                import base64
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Формируем промпт для API
            prompt = "Проанализируй анализы пациента и сделай выводы о его здоровье, если он болен - насколько велико отклонение от нормы"
            
            # Формируем данные для запроса
            payload = {
                "model": "deepseek-chat",  # Замените на актуальную модель
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API DeepSeek: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Ошибка при анализе отчета: {str(e)}")
            return None
            
    def save_analysis_result(self, image_path: str, analysis_result: str) -> bool:
        """
        Сохраняет результат анализа в текстовый файл
        
        Args:
            image_path: путь к исходному изображению
            analysis_result: результат анализа
            
        Returns:
            bool: успешно ли сохранен результат
        """
        try:
            # Получаем имя файла без расширения
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Формируем путь для сохранения результата
            result_dir = os.path.join('ai-result', 'deepseek')
            os.makedirs(result_dir, exist_ok=True)
            
            result_path = os.path.join(result_dir, f"{base_name}.txt")
            
            # Сохраняем результат
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
                
            logger.info(f"Результат анализа сохранен в {result_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении результата анализа: {str(e)}")
            return False 