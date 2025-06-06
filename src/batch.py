"""
Модуль для пакетной обработки документов
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from src.processor import DocumentProcessor
from src.database import Database
import re
import numpy as np
from src.nlp_model import NERModel
from src.medical_terms import MedicalTermsManager


def setup_logging(log_file: str = 'deidentification.log') -> None:
    """
    Настройка логирования
    
    Args:
        log_file: путь к файлу лога
    """
    # Создаем директорию для логов если её нет
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Добавляем файловый обработчик
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Добавляем консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

class BatchProcessor:
    """Класс для пакетной обработки документов"""
    
    def __init__(self, db: Optional[Database] = None):
        """
        Инициализация процессора
        
        Args:
            db: экземпляр базы данных (опционально)
        """
        self.processor = DocumentProcessor(db)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация BatchProcessor")
        
        # Белый список медицинских терминов
        self.medical_whitelist = {
            'препарат', 'лекарство', 'диагноз', 'симптом',
            'анализ', 'исследование', 'процедура', 'терапия',
            'синдром', 'патология', 'заболевание', 'состояние',
            'медицинский', 'клинический', 'лабораторный', 'диагностический',
            'лечебный', 'профилактический', 'реабилитационный', 'паллиативный',
            'амбулаторный', 'стационарный', 'экстренный', 'плановый',
            'консервативный', 'оперативный', 'хирургический', 'терапевтический',
            # Добавляем больше медицинских терминов
            'гемоглобин', 'лейкоциты', 'эритроциты', 'тромбоциты',
            'глюкоза', 'холестерин', 'билирубин', 'креатинин',
            'мочевина', 'белок', 'альбумин', 'глобулин',
            'фермент', 'гормон', 'витамин', 'минерал',
            'антибиотик', 'анальгетик', 'антисептик', 'антигистамин',
            'иммунитет', 'аллергия', 'воспаление', 'инфекция',
            'опухоль', 'метастаз', 'рецидив', 'ремиссия',
            'осложнение', 'побочный', 'противопоказание', 'показание'
        }
        
        # Единицы измерения
        self.measurement_units = {
            # Моль и производные
            'моль', 'моля', 'молей', 'молях', 'молем', 'молями',
            'ммоль', 'ммоля', 'ммолей', 'ммолях', 'ммолем', 'ммолями',
            'мкмоль', 'мкмоля', 'мкмолей', 'мкмолях', 'мкмолем', 'мкмолями',
            'нмоль', 'нмоля', 'нмолей', 'нмолях', 'нмолем', 'нмолями',
            'пмоль', 'пмоля', 'пмолей', 'пмолях', 'пмолем', 'пмолями',
            
            # Грамм и производные
            'г', 'гр', 'грамм', 'грамма', 'граммов', 'грамму', 'граммам',
            'мг', 'мгр', 'миллиграмм', 'миллиграмма', 'миллиграммов', 'миллиграмму', 'миллиграммам',
            'мкг', 'микрограмм', 'микрограмма', 'микрограммов', 'микрограмму', 'микрограммам',
            'нг', 'нанограмм', 'нанограмма', 'нанограммов', 'нанограмму', 'нанограммам',
            
            # Литр и производные
            'л', 'литр', 'литра', 'литров', 'литру', 'литрам',
            'мл', 'миллилитр', 'миллилитра', 'миллилитров', 'миллилитру', 'миллилитрам',
            'мкл', 'микролитр', 'микролитра', 'микролитров', 'микролитру', 'микролитрам',
            'дл', 'децилитр', 'децилитра', 'децилитров', 'децилитру', 'децилитрам',
            
            # Метр и производные
            'м', 'метр', 'метра', 'метров', 'метру', 'метрам',
            'см', 'сантиметр', 'сантиметра', 'сантиметров', 'сантиметру', 'сантиметрам',
            'мм', 'миллиметр', 'миллиметра', 'миллиметров', 'миллиметру', 'миллиметрам',
            'мкм', 'микрометр', 'микрометра', 'микрометров', 'микрометру', 'микрометрам',
            'нм', 'нанометр', 'нанометра', 'нанометров', 'нанометру', 'нанометрам',
            
            # Комбинированные единицы
            'мг/мл', 'мкг/мл', 'ммоль/л', 'мкмоль/л', 'г/л', 'мг/дл', 'мкг/дл',
            'мм/ч', 'мм рт.ст.', 'мм рт. ст.', 'кПа', 'кпа', 'атм', 'бар',
            'мл/мин', 'л/мин', 'мл/кг', 'мл/кг/мин',
            
            # Время
            'сут', 'сутки', 'дн', 'день', 'нед', 'неделя', 'мес', 'месяц',
            'год', 'гг', 'года', 'мин', 'минута', 'час', 'часы', 'сек', 'секунда',
            'мс', 'миллисекунда', 'мкс', 'микросекунда'
        }
        
        # Медицинские сокращения
        self.medical_abbreviations = {
            'АД', 'ад', 'АД', 'ад',  # артериальное давление
            'ЧСС', 'чсс',  # частота сердечных сокращений
            'ЧД', 'чд',  # частота дыхания
            'ОАК', 'оак',  # общий анализ крови
            'ОАМ', 'оам',  # общий анализ мочи
            'БАК', 'бак',  # биохимический анализ крови
            'ЭКГ', 'экг',  # электрокардиограмма
            'УЗИ', 'узи',  # ультразвуковое исследование
            'КТ', 'кт',  # компьютерная томография
            'МРТ', 'мрт',  # магнитно-резонансная томография
            'РЭГ', 'рег',  # реоэнцефалография
            'ЭЭГ', 'ээг',  # электроэнцефалография
            'ФГДС', 'фгдс',  # фиброгастродуоденоскопия
            'ФКС', 'фкс',  # фиброколоноскопия
            'ИФА', 'ифа',  # иммуноферментный анализ
            'ПЦР', 'пцр',  # полимеразная цепная реакция
            'СОЭ', 'соэ',  # скорость оседания эритроцитов
            'ЛДГ', 'лдг',  # лактатдегидрогеназа
            'АЛТ', 'алт',  # аланинаминотрансфераза
            'АСТ', 'аст',  # аспартатаминотрансфераза
            'ГГТ', 'ггт',  # гамма-глутамилтрансфераза
            'ЩФ', 'щф',  # щелочная фосфатаза
            'СРБ', 'срб',  # С-реактивный белок
            'ПТИ', 'пти',  # протромбиновый индекс
            'МНО', 'мно',  # международное нормализованное отношение
            'АЧТВ', 'ачтв',  # активированное частичное тромбопластиновое время
            'Фибриноген', 'фибриноген',
            'D-димер', 'd-димер',
            'ТТГ', 'ттг',  # тиреотропный гормон
            'Т3', 'т3',  # трийодтиронин
            'Т4', 'т4',  # тироксин
            'Пролактин', 'пролактин',
            'Кортизол', 'кортизол',
            'Инсулин', 'инсулин',
            'Глюкоза', 'глюкоза',
            'Гликированный гемоглобин', 'гликированный гемоглобин',
            'HbA1c', 'hba1c',
            'Холестерин', 'холестерин',
            'ЛПВП', 'лпвп',  # липопротеины высокой плотности
            'ЛПНП', 'лпнп',  # липопротеины низкой плотности
            'Триглицериды', 'триглицериды',
            'Креатинин', 'креатинин',
            'Мочевина', 'мочевина',
            'Мочевая кислота', 'мочевая кислота',
            'Билирубин', 'билирубин',
            'Альбумин', 'альбумин',
            'Общий белок', 'общий белок',
            'Кальций', 'кальций',
            'Фосфор', 'фосфор',
            'Магний', 'магний',
            'Натрий', 'натрий',
            'Калий', 'калий',
            'Хлор', 'хлор',
            'Железо', 'железо',
            'Ферритин', 'ферритин',
            'Трансферрин', 'трансферрин',
            'Витамин B12', 'витамин b12',
            'Фолиевая кислота', 'фолиевая кислота',
            'Витамин D', 'витамин d',
            'ПСА', 'пса',  # простат-специфический антиген
            'СА-125', 'са-125',  # онкомаркер
            'СА-19-9', 'са-19-9',  # онкомаркер
            'РЭА', 'рэа',  # раково-эмбриональный антиген
            'АФП', 'афп',  # альфа-фетопротеин
            'ХГЧ', 'хгч'  # хорионический гонадотропин человека
        }
        
        self.logger.info("Инициализация завершена")
        self.logger.info(f"Загружено {len(self.medical_whitelist)} медицинских терминов")
        self.logger.info(f"Загружено {len(self.measurement_units)} единиц измерения")
        self.logger.info(f"Загружено {len(self.medical_abbreviations)} медицинских сокращений")
        
        # Добавляем паттерны для единиц измерения
        self.measurement_patterns = [
            r'\b\d+\s*(?:мкмоль|ммоль|моль|нмоль|пмоль)\b',
            r'\b(?:мкмоль|ммоль|моль|нмоль|пмоль)\s*/\s*(?:л|мл|дл)\b',
            r'\b\d+\s*(?:г|мг|мкг|нг)\b',
            r'\b(?:г|мг|мкг|нг)\s*/\s*(?:л|мл|дл)\b',
            r'\b\d+\s*(?:л|мл|мкл|дл)\b',
            r'\b(?:л|мл|мкл|дл)\s*/\s*(?:мин|час|сут|день)\b',
            r'\b\d+\s*(?:м|см|мм|мкм|нм)\b',
            r'\b(?:м|см|мм|мкм|нм)\s*/\s*(?:ч|мин|сек)\b'
        ]
        
        # Добавляем регулярные выражения для медицинских терминов
        self.medical_patterns = [
            r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:болезнь|синдром|симптом|патология|состояние)\b',
            r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:терапия|лечение|процедура|манипуляция)\b',
            r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:отделение|кабинет|палата|центр)\b',
            r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:анализ|исследование|тест|проба)\b',
            r'\b(?:[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*)\s+(?:препарат|лекарство|средство|медикамент)\b'
        ]
        
        # Контексты, где персональные данные допустимы
        self.allowed_contexts = {
            'doctor_signature': True,
            'medical_staff': True,
            'department_name': True
        }
        
        # Добавляем медицинские контекстные маркеры
        self.medical_context_markers = {
            'patient_markers': {
                'пациент', 'больной', 'господин', 'госпожа', 'гражданин', 'гражданка',
                'ф.и.о.', 'фио', 'фамилия', 'имя', 'отчество', 'дата рождения',
                'возраст', 'пол', 'адрес', 'телефон', 'страховой полис', 'снилс'
            },
            'medical_markers': {
                'диагноз', 'жалобы', 'анамнез', 'обследование', 'результаты',
                'анализ', 'исследование', 'процедура', 'операция', 'лечение',
                'назначения', 'рекомендации', 'прогноз', 'осложнения',
                'аллергия', 'непереносимость', 'хронические заболевания',
                'принимаемые препараты', 'группа крови', 'резус-фактор'
            },
            'doctor_markers': {
                'врач', 'доктор', 'фельдшер', 'медсестра', 'заведующий',
                'профессор', 'доцент', 'ассистент', 'ординатор', 'интерн',
                'подпись', 'печать', 'штамп', 'отделение', 'кафедра',
                'клиника', 'больница', 'поликлиника', 'диспансер'
            }
        }
        
        # Добавляем паттерны медицинских документов
        self.document_patterns = {
            'medical_card': {
                'header': r'МЕДИЦИНСКАЯ КАРТА\s+№\s*\d+',
                'sections': [
                    r'ДАННЫЕ ПАЦИЕНТА',
                    r'ЖАЛОБЫ',
                    r'АНАМНЕЗ',
                    r'ОБЪЕКТИВНОЕ ОБСЛЕДОВАНИЕ',
                    r'ДИАГНОЗ',
                    r'ПЛАН ОБСЛЕДОВАНИЯ',
                    r'ЛЕЧЕНИЕ',
                    r'НАЗНАЧЕНИЯ'
                ]
            },
            'discharge_summary': {
                'header': r'ВЫПИСКА\s+ИЗ\s+ИСТОРИИ\s+БОЛЕЗНИ',
                'sections': [
                    r'ПАЦИЕНТ',
                    r'ДИАГНОЗ ПРИ ПОСТУПЛЕНИИ',
                    r'ДИАГНОЗ КЛИНИЧЕСКИЙ',
                    r'ПРОВЕДЕННОЕ ЛЕЧЕНИЕ',
                    r'РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ',
                    r'РЕКОМЕНДАЦИИ'
                ]
            },
            'operation_report': {
                'header': r'ПРОТОКОЛ\s+ОПЕРАЦИИ',
                'sections': [
                    r'ПАЦИЕНТ',
                    r'ДИАГНОЗ',
                    r'ВИД ОПЕРАЦИИ',
                    r'ХОД ОПЕРАЦИИ',
                    r'ОСЛОЖНЕНИЯ',
                    r'ЗАКЛЮЧЕНИЕ'
                ]
            }
        }
        
        # Добавляем медицинские классификации
        self.medical_classifications = {
            'icd10': set(),  # Будет заполнено из базы данных
            'icd11': set(),  # Будет заполнено из базы данных
            'medical_protocols': set()  # Будет заполнено из базы данных
        }
        
        # Инициализация медицинских классификаций
        if db:
            self._init_medical_classifications(db)
            
        # Пороги уверенности для разных типов данных
        self.confidence_thresholds = {
            'name': 0.85,  # Повышенный порог для имен
            'diagnosis': 0.90,  # Очень высокий порог для диагнозов
            'procedure': 0.85,  # Высокий порог для процедур
            'medication': 0.80,  # Высокий порог для лекарств
            'measurement': 0.75  # Средний порог для измерений
        }
        
        # Система верификации
        self.verification_rules = {
            'name': [
                self._verify_name_context,
                self._verify_name_format,
                self._verify_name_frequency
            ],
            'diagnosis': [
                self._verify_diagnosis_icd,
                self._verify_diagnosis_context,
                self._verify_diagnosis_consistency
            ],
            'procedure': [
                self._verify_procedure_protocol,
                self._verify_procedure_context,
                self._verify_procedure_consistency
            ]
        }
        
        self.medical_terms = MedicalTermsManager()
    
    def _init_medical_classifications(self, db: Database):
        """Инициализация медицинских классификаций из базы данных"""
        # Эта функция теперь не нужна, так как классификации не используются для валидации
        self.logger.info("Пропущена инициализация медицинских классификаций, т.к. валидация отключена")

    def _verify_name_context(self, text: str, start: int, end: int) -> bool:
        """Проверка контекста имени"""
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()
        
        # Проверяем наличие маркеров пациента
        has_patient_marker = any(marker in context for marker in self.medical_context_markers['patient_markers'])
        
        # Проверяем отсутствие маркеров врача
        has_doctor_marker = any(marker in context for marker in self.medical_context_markers['doctor_markers'])
        
        return has_patient_marker and not has_doctor_marker

    def _verify_name_format(self, name: str) -> bool:
        """Проверка формата имени"""
        # Проверка на типичные форматы имен
        name_patterns = [
            r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+$',  # ФИО
            r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+$',  # ФИ
            r'^[А-ЯЁ][а-яё]+$'  # Ф
        ]
        return any(re.match(pattern, name) for pattern in name_patterns)

    def _verify_name_frequency(self, name: str, text: str) -> bool:
        """Проверка частоты появления имени в тексте"""
        # Имя должно встречаться несколько раз в документе
        frequency = text.lower().count(name.lower())
        return 1 <= frequency <= 10  # Реалистичное количество упоминаний

    def _verify_diagnosis_icd(self, diagnosis: str) -> bool:
        """Проверка диагноза по МКБ"""
        diagnosis_lower = diagnosis.lower()
        return any(
            diagnosis_lower in name.lower() 
            for _, name in self.medical_classifications['icd10'] | self.medical_classifications['icd11']
        )

    def _verify_diagnosis_context(self, text: str, start: int, end: int) -> bool:
        """Проверка контекста диагноза"""
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()
        
        return any(marker in context for marker in {
            'диагноз', 'нозология', 'заболевание', 'патология',
            'синдром', 'осложнение', 'сопутствующий'
        })

    def _verify_diagnosis_consistency(self, diagnosis: str, text: str) -> bool:
        """Проверка согласованности диагноза"""
        # Проверяем, что диагноз упоминается в контексте других медицинских терминов
        diagnosis_lower = diagnosis.lower()
        medical_terms = {
            'симптом', 'признак', 'жалоба', 'анамнез', 'обследование',
            'анализ', 'исследование', 'лечение', 'терапия', 'прогноз'
        }
        
        # Ищем контекст вокруг диагноза
        start = text.lower().find(diagnosis_lower)
        if start == -1:
            return False
            
        context_start = max(0, start - 200)
        context_end = min(len(text), start + len(diagnosis) + 200)
        context = text[context_start:context_end].lower()
        
        # Проверяем наличие медицинских терминов в контексте
        return any(term in context for term in medical_terms)

    def _verify_procedure_protocol(self, procedure: str) -> bool:
        """Проверка процедуры по медицинским протоколам"""
        procedure_lower = procedure.lower()
        return any(
            procedure_lower in name.lower() or procedure_lower in desc.lower()
            for name, desc in self.medical_classifications['medical_protocols']
        )

    def _verify_procedure_context(self, text: str, start: int, end: int) -> bool:
        """Проверка контекста процедуры"""
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()
        
        return any(marker in context for marker in {
            'процедура', 'манипуляция', 'операция', 'вмешательство',
            'метод', 'техника', 'способ', 'прием'
        })

    def _verify_procedure_consistency(self, procedure: str, text: str) -> bool:
        """Проверка согласованности процедуры"""
        procedure_lower = procedure.lower()
        medical_terms = {
            'показание', 'противопоказание', 'подготовка', 'техника',
            'осложнение', 'результат', 'эффективность', 'безопасность'
        }
        
        start = text.lower().find(procedure_lower)
        if start == -1:
            return False
            
        context_start = max(0, start - 200)
        context_end = min(len(text), start + len(procedure) + 200)
        context = text[context_start:context_end].lower()
        
        return any(term in context for term in medical_terms)

    def _validate_personal_data(self, data_type: str, value: str, context: str) -> bool:
        """
        Расширенная валидация персональных данных (отключена)

        Всегда возвращает False, т.к. маскирование отключено.
        """
        self.logger.debug(f"Валидация данных отключена. Проверка: тип={data_type}, значение='{value}', контекст='{context}'")
        return False # Всегда возвращаем False, т.к. маскирование отключено

    def _calculate_confidence(self, data_type: str, value: str, context: str) -> float:
        """
        Расчет уверенности в правильности определения данных (отключен)

        Всегда возвращает 0.0, т.к. маскирование отключено.
        """
        self.logger.debug("Расчет уверенности отключен")
        return 0.0 # Всегда возвращаем 0.0, т.к. маскирование отключено

    def _get_context_text(self, value: str, context: str) -> str:
        """
        Получение контекстного текста для проверки (отключено)
        """
        # Эта функция теперь не используется для валидации
        return context

    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         clinic_name: str) -> List[Dict]:
        """
        Обработка всех документов в директории
        
        Args:
            input_dir: путь к директории с исходными документами
            output_dir: путь для сохранения обработанных документов
            clinic_name: название клиники
            
        Returns:
            List[Dict]: результаты обработки
        """
        # Создаем выходную директорию если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем список всех файлов
        files = self._get_document_files(input_dir)
        
        results = []
        for file_path in tqdm(files, desc="Обработка документов"):
            try:
                # Обработка одного документа
                processed_path, extracted_data = self.processor.process_document(
                    file_path=file_path,
                    clinic_name=clinic_name,
                    output_dir=output_dir
                )
                
                results.append({
                    'original': file_path,
                    'processed': processed_path,
                    'status': 'success',
                    'extracted_data': extracted_data
                })
                
                self.logger.info(f"Успешно обработан файл: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке {file_path}: {str(e)}")
                results.append({
                    'original': file_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _get_document_files(self, directory: str) -> List[str]:
        """
        Получение списка файлов для обработки
        
        Args:
            directory: путь к директории
            
        Returns:
            List[str]: список путей к файлам
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
        
        files = []
        for filename in os.listdir(directory):  
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                files.append(os.path.join(directory, filename))
        
        return files

    def _get_word_context(self, match, sections: Dict[str, Tuple[int, int]]) -> str:
        """
        Определяет контекст слова на основе его позиции в документе (отключено)

        Всегда возвращает 'unknown', т.к. контекст не используется для маскирования.
        """
        return 'unknown' # Всегда возвращаем unknown, т.к. контекст не используется для маскирования

    def _is_sensitive_context(self, context: str) -> bool:
        """
        Проверяет, является ли контекст чувствительным для маскирования (отключено)

        Всегда возвращает False, т.к. маскирование отключено.
        """
        return False # Всегда возвращаем False, т.к. маскирование отключено

    def _extract_data(self, text_data: List[Dict]) -> Dict:
        """
        Извлечение данных из распознанного текста (отключено для маскирования)

        Возвращает пустой список чувствительных регионов, т.к. маскирование отключено.
        """
        # Возвращаем пустой список регионов - маскирование отключено
        self.logger.info("Извлечение данных: маскирование отключено, возвращаем пустой список регионов.")
        return {
            'sensitive_regions': [], # Возвращаем пустой список
            'medical_data': {}
        }

    def _identify_document_sections(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        Определение секций в медицинском документе
        
        Args:
            text: текст документа
            
        Returns:
            Dict[str, Tuple[int, int]]: словарь секций и их границ
        """
        sections = {
            'header': (0, 0),
            'patient_info': (0, 0),
            'medical_data': (0, 0),
            'doctor_info': (0, 0),
            'footer': (0, 0)
        }
        
        # Определяем границы секций по ключевым словам
        section_markers = {
            'header': r'^(?:МЕДИЦИНСКАЯ КАРТА|ИСТОРИЯ БОЛЕЗНИ)',
            'patient_info': r'(?:ДАННЫЕ ПАЦИЕНТА|ПАЦИЕНТ)',
            'medical_data': r'(?:ДИАГНОЗ|НАЗНАЧЕНИЯ)',
            'doctor_info': r'(?:ВРАЧ|ПОДПИСЬ)',
            'footer': r'(?:ДАТА|ПЕЧАТЬ)'
        }
        
        # ... определение границ секций ...
        
        return sections

    def _mask_sensitive_data(self, image: np.ndarray, sensitive_regions: List[Dict]) -> np.ndarray:
        """
        Маскирование чувствительных данных на изображении
        
        Args:
            image: исходное изображение
            sensitive_regions: список регионов для маскирования
            
        Returns:
            np.ndarray: изображение с замазанными данными
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info("НАЧАЛО ПРОЦЕССА МАСКИРОВАНИЯ")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Размер изображения: {image.shape}")
        self.logger.info(f"Количество регионов для маскирования: {len(sensitive_regions)}")
        
        # В текущей реализации маскирование отключено, просто возвращаем исходное изображение
        if not sensitive_regions:
            self.logger.info("Нет регионов для маскирования. Возвращаем исходное изображение.")
            return image

        # Создаем копию изображения
        masked_image = image.copy()

        # Добавляем проверку на пересечение регионов
        def is_overlapping(region1: Dict, region2: Dict) -> bool:
            return not (region1['left'] + region1['width'] < region2['left'] or
                       region2['left'] + region2['width'] < region1['left'] or
                       region1['top'] + region1['height'] < region2['top'] or
                       region2['top'] + region2['height'] < region1['top'])

        # Объединяем пересекающиеся регионы
        merged_regions = []
        for region in sensitive_regions:
            self.logger.info(f"\n{'-'*30}")
            self.logger.info(f"Обработка региона: {region['text']} (тип: {region['type']})")
            self.logger.info(f"Координаты: left={region['left']}, top={region['top']}, width={region['width']}, height={region['height']}")
            self.logger.info(f"Уверенность модели: {region.get('confidence', 'N/A')}")
            self.logger.info(f"Использованные модели: {region.get('models', ['N/A'])}")
            
            merged = False
            for existing in merged_regions:
                if is_overlapping(region, existing):
                    self.logger.info(f"Обнаружено пересечение с регионом: {existing['text']}")
                    # Объединяем регионы
                    existing['left'] = min(existing['left'], region['left'])
                    existing['top'] = min(existing['top'], region['top'])
                    existing['width'] = max(existing['left'] + existing['width'],
                                          region['left'] + region['width']) - existing['left']
                    existing['height'] = max(existing['top'] + existing['height'],
                                           region['top'] + region['height']) - existing['top']
                    self.logger.info(f"Объединенный регион: {existing['text']}")
                    self.logger.info(f"Новые координаты: left={existing['left']}, top={existing['top']}, width={existing['width']}, height={existing['height']}")
                    merged = True
                    break
            if not merged:
                merged_regions.append(region)
                self.logger.info(f"Добавлен новый регион для маскирования: {region['text']}")
        
        # Маскируем каждый регион
        for region in merged_regions:
            try:
                # Получаем координаты региона
                left = int(region['left'])
                top = int(region['top'])
                width = int(region['width'])
                height = int(region['height'])
                
                # Проверяем валидность координат
                if (left < 0 or top < 0 or width <= 0 or height <= 0 or
                    left + width > masked_image.shape[1] or
                    top + height > masked_image.shape[0]):
                    self.logger.warning(f"Некорректные координаты региона: {region['text']}")
                    self.logger.warning(f"Координаты: left={left}, top={top}, width={width}, height={height}")
                    self.logger.warning(f"Размеры изображения: {masked_image.shape}")
                    continue
                
                # Маскируем регион
                masked_image[top:top+height, left:left+width] = 255  # Белый цвет
                
                self.logger.info(f"Успешно замазан регион: {region['text']}")
                self.logger.info(f"Финальные координаты маскирования: left={left}, top={top}, width={width}, height={height}")
                
            except Exception as e:
                self.logger.error(f"Ошибка при маскировании региона {region['text']}: {str(e)}")
                self.logger.error(f"Координаты региона: left={region['left']}, top={region['top']}, width={region['width']}, height={region['height']}")
                continue
        
        self.logger.info(f"Процесс маскирования завершен. Замазано регионов: {len(merged_regions)}")
        return masked_image

    def _log_extraction_details(self, text: str, found_data: Dict, context: str):
        """
        Подробное логирование процесса извлечения данных
        
        Args:
            text: исходный текст
            found_data: найденные данные
            context: контекст извлечения
        """
        logging.info(f"Контекст: {context}")
        logging.info(f"Найденные данные: {found_data}")
        logging.info("Решение о маскировании: Отключено") # Изменено сообщение

    def _is_medical_term(self, text: str) -> bool:
        """
        Проверяет, является ли текст медицинским термином
        
        Args:
            text: проверяемый текст
            
        Returns:
            bool: является ли текст медицинским термином
        """
        text = text.lower().strip('., ')
        is_term = self.medical_terms.is_medical_term(text)
        self.logger.debug(f"Проверка текста на медицинский термин: '{text}' -> {is_term}")
        return is_term

def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Обработка медицинских документов')
    parser.add_argument('--input', help='Входная директория')
    parser.add_argument('--output', help='Выходная директория')
    parser.add_argument('--clinic', help='Название клиники')
    parser.add_argument('--log', default='deidentification.log', help='Файл лога')
    parser.add_argument('--db-url', help='URL базы данных (опционально)')
    args = parser.parse_args()

    # Значения по умолчанию
    input_dir = args.input or 'input'
    output_dir = args.output or 'output'
    clinic_name = args.clinic or 'DefaultClinic'

    # Настройка логирования
    setup_logging(args.log)

    # Создание базы данных если указан URL
    db = None
    if args.db_url:
        db = Database(args.db_url)

    # Создание процессора
    processor = BatchProcessor(db)

    # Обработка директории
    results = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        clinic_name=clinic_name
    )

    # Вывод результатов
    success_count = len([r for r in results if r['status'] == 'success'])
    error_count = len([r for r in results if r['status'] == 'error'])

    print(f"\nОбработка завершена:")
    print(f"Всего документов: {len(results)}")
    print(f"Успешно обработано: {success_count}")
    print(f"Ошибок: {error_count}")

    if error_count > 0:
        print("\nСписок ошибок:")
        for result in results:
            if result['status'] == 'error':
                print(f"- {result['original']}: {result['error']}")

if __name__ == '__main__':
    main() 