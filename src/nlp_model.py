import logging
import torch
from typing import List, Dict

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from ctransformers import AutoModelForCausalLM as CTransformersModel
except ImportError:
    CTransformersModel = None

logger = logging.getLogger(__name__)

class NERModel:
    def __init__(self, model_path: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        """
        Инициализация модели для распознавания именованных сущностей
        Args:
            model_path: путь к локальной модели или название модели в HuggingFace
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # logger.info(f"Используется устройство: {self.device}")
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        # Сначала пробуем ctransformers (для CPU и некоторых quantized моделей)
        if CTransformersModel is not None:
            try:
                logger.info(f"Пробуем загрузить модель через ctransformers: {self.model_path}")
                self.model = CTransformersModel.from_pretrained(self.model_path, model_type="llama")
                logger.info("Модель успешно загружена через ctransformers")
                return
            except Exception as e:
                logger.warning(f"Не удалось загрузить через ctransformers: {e}")
        # Если не получилось, пробуем transformers
        if AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                logger.info(f"Пробуем загрузить модель через transformers: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                logger.info("Модель успешно загружена через transformers")
                return
            except Exception as e:
                logger.error(f"Не удалось загрузить модель через transformers: {e}")
        raise RuntimeError("Не удалось загрузить языковую модель. Проверьте путь и зависимости.")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Извлечение именованных сущностей из текста (через промпт)
        Args:
            text: входной текст
        Returns:
            Dict[str, List[str]]: словарь с найденными сущностями по категориям
        """
        # logger.info("Извлечение сущностей из текста через языковую модель...")
        prompt = (
            "Выдели из текста имена людей (ФИО) и выведи их списком. "
            "Текст: " + text + "\n\nФормат ответа: Имена: [список через запятую]"
        )
        response = self._generate(prompt)
        # Примитивный парсер ответа
        names = []
        if "Имена:" in response:
            part = response.split("Имена:", 1)[1]
            names = [n.strip() for n in part.split(",") if n.strip()]
        # logger.info(f"Найдено имён: {len(names)}")
        return {"names": names}

    def is_person_name(self, text: str) -> bool:
        """
        Проверяет, является ли текст именем человека (через промпт)
        Args:
            text: текст для проверки
        Returns:
            bool: True если текст похож на имя человека
        """
        prompt = (
            f"Является ли '{text}' именем человека? Ответь только Да или Нет."
        )
        response = self._generate(prompt)
        return "да" in response.lower()

    def _generate(self, prompt: str) -> str:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована!")
        if self.tokenizer is None and hasattr(self.model, "__call__"):
            # ctransformers
            return self.model(prompt, max_new_tokens=128)
        elif self.tokenizer is not None:
            # transformers
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise RuntimeError("Не удалось сгенерировать ответ: не найдена модель/токенизатор.") 