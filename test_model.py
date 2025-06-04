import logging
import sys
import transformers
import torch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_model_loading():
    logger.info("Начало тестирования модели...")
    
    # Проверяем доступность GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")
    
    try:
        # Загружаем токенизатор
        logger.info("Загрузка токенизатора...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sberbank-ai/sbert_large_nlu_ru",
            local_files_only=False,
            cache_dir="./models"
        )
        logger.info("Токенизатор успешно загружен")
        
        # Загружаем модель
        logger.info("Загрузка модели...")
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            "sberbank-ai/sbert_large_nlu_ru",
            local_files_only=False,
            cache_dir="./models"
        )
        logger.info("Модель успешно загружена")
        
        # Инициализируем pipeline
        logger.info("Инициализация pipeline...")
        ner_pipeline = transformers.pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("Pipeline успешно инициализирован")
        
        # Тестируем модель
        logger.info("Тестирование модели на примере...")
        test_text = "Иванов работает врачом в больнице Петрова"
        result = ner_pipeline(test_text)
        logger.info(f"Результат тестирования: {result}")
        
        if not result:
            raise Exception("Модель не вернула результат на тестовом примере")
            
        logger.info("Тест успешно завершен")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании модели: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if not success:
        logger.error("Тест завершился с ошибкой")
        sys.exit(1)
    logger.info("Тест успешно пройден") 