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

def setup_logging(log_file: str = 'deidentification.log') -> None:
    """
    Настройка логирования
    
    Args:
        log_file: путь к файлу лога
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
        
        # Инициализация моделей
        try:
            from deeppavlov import build_model
            self.deep_pavlov_model = build_model('ner_ontonotes_bert_mult', download=True)
            self.use_deep_pavlov = True
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить DeepPavlov: {e}")
            self.use_deep_pavlov = False
            
        try:
            from yargy import Parser
            from yargy.predicates import dictionary
            from yargy.interpretation import fact, attribute, FactExtractor
            
            # Определяем факт для имени
            Name = fact('Name', ['first', 'last', 'middle'])
            self.yargy_parser = Parser()
            self.use_yargy = True
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить Yargy: {e}")
            self.use_yargy = False
            
        # Белый список медицинских терминов
        self.medical_whitelist = {
            'препарат', 'лекарство', 'диагноз', 'симптом',
            'анализ', 'исследование', 'процедура', 'терапия',
            'синдром', 'патология', 'заболевание', 'состояние'
        }
        
        # Белый список медицинских сокращений
        self.medical_abbreviations = {
            # Единицы измерения
            'г', 'гр', 'грамм', 'мг', 'миллиграмм', 'мкг', 'микрограмм', 'нг', 'нанограмм',
            'мл', 'миллилитр', 'л', 'литр', 'мкл', 'микролитр', 'дл', 'децилитр',
            'мм', 'миллиметр', 'см', 'сантиметр', 'м', 'метр', 'км', 'километр',
            'ммоль', 'миллимоль', 'мкмоль', 'микромоль', 'моль', 'нмоль', 'наномоль',
            'ед', 'единица', 'ме', 'международная единица', 'мке', 'микроме',
            'мг/мл', 'мкг/мл', 'ммоль/л', 'мкмоль/л', 'г/л', 'мг/дл', 'мкг/дл',
            'мм/ч', 'мм рт.ст.', 'мм рт. ст.', 'кПа', 'кпа', 'атм', 'бар',
            'мл/мин', 'л/мин', 'мл/кг', 'мл/кг/мин',
            'мкм', 'микрометр', 'нм', 'нанометр', 'пм', 'пикометр',
            'ккал', 'кал', 'дж', 'джоуль', 'вт', 'ватт',
            'гц', 'герц', 'кгц', 'килогерц', 'мгц', 'мегагерц',
            
            # Временные интервалы
            'сут', 'сутки', 'дн', 'день', 'нед', 'неделя', 'мес', 'месяц',
            'год', 'гг', 'года', 'мин', 'минута', 'час', 'часы', 'сек', 'секунда',
            'мс', 'миллисекунда', 'мкс', 'микросекунда',
            'сут.', 'дн.', 'нед.', 'мес.', 'год.', 'мин.', 'час.', 'сек.',
            
            # Частоты и дозировки
            'р/д', 'раз в день', 'р/нед', 'раз в неделю', 'р/мес', 'раз в месяц',
            'р/сут', 'раз в сутки', 'р/час', 'раз в час', 'р/мин', 'раз в минуту',
            'еж/д', 'ежедневно', 'еж/нед', 'еженедельно', 'еж/мес', 'ежемесячно',
            'через', 'каждые', 'каж.', 'каж.',
            
            # Медицинские сокращения
            'в/в', 'внутривенно', 'в/м', 'внутримышечно', 'п/к', 'подкожно',
            'п/о', 'перорально', 'в/б', 'внутрибрюшинно', 'в/а', 'внутриартериально',
            'и/м', 'интрамускулярно', 'и/в', 'интравенозно', 'в/п', 'внутриплеврально',
            'в/с', 'внутрисуставно', 'в/к', 'внутрикожно', 'в/л', 'внутрилингвально',
            'в/т', 'внутритрахеально', 'в/ц', 'внутрицеребрально',
            'п/я', 'подъязычно', 'п/к', 'подкожно', 'п/б', 'подбуккально',
            'р/р', 'раствор', 'сусп.', 'суспензия', 'эмульс.', 'эмульсия',
            'таб.', 'таблетка', 'капс.', 'капсула', 'амп.', 'ампула',
            'фл.', 'флакон', 'уп.', 'упаковка', 'бл.', 'блистер',
            
            # Общие медицинские сокращения
            'д.м.н.', 'доктор медицинских наук', 'к.м.н.', 'кандидат медицинских наук',
            'проф.', 'профессор', 'доц.', 'доцент', 'асс.', 'ассистент',
            'ст.', 'старший', 'мл.', 'младший', 'гл.', 'главный',
            'зав.', 'заведующий', 'отд.', 'отделение', 'каф.', 'кафедра',
            'поликл.', 'поликлиника', 'стац.', 'стационар', 'амб.', 'амбулатория',
            'дисп.', 'диспансер', 'лабор.', 'лаборатория', 'пров.', 'провизор',
            'фарм.', 'фармацевт', 'мед.', 'медицинский', 'сан.', 'санитарный',
            'эпид.', 'эпидемиологический', 'гиг.', 'гигиенический',
            'врач', 'доктор', 'фельдшер', 'медсестра', 'акушерка',
            'рец.', 'рецепт', 'назн.', 'назначение', 'диагн.', 'диагноз',
            'жал.', 'жалобы', 'анамн.', 'анамнез', 'обсл.', 'обследование',
            'конс.', 'консультация', 'набл.', 'наблюдение', 'леч.', 'лечение',
            'проф.', 'профилактика', 'реаб.', 'реабилитация',
            
            # Анатомические сокращения
            'пр.м.', 'правый', 'л.м.', 'левый', 'об.м.', 'общий',
            'в.п.', 'верхний', 'н.п.', 'нижний', 'ср.п.', 'средний',
            'перед.', 'передний', 'зад.', 'задний', 'бок.', 'боковой',
            'внутр.', 'внутренний', 'внеш.', 'внешний',
            'пр.', 'правый', 'л.', 'левый', 'об.', 'общий',
            'в.', 'верхний', 'н.', 'нижний', 'ср.', 'средний',
            'перед.', 'передний', 'зад.', 'задний', 'бок.', 'боковой',
            'внутр.', 'внутренний', 'внеш.', 'внешний',
            'вентр.', 'вентральный', 'дорс.', 'дорсальный', 'латер.', 'латеральный',
            'медиал.', 'медиальный', 'прокс.', 'проксимальный', 'дист.', 'дистальный',
            'краниал.', 'краниальный', 'каудал.', 'каудальный',
            
            # Лабораторные показатели
            'гемоглобин', 'гб', 'hgb', 'hct', 'гематокрит', 'эритроциты', 'эр',
            'лейкоциты', 'лейк', 'тромбоциты', 'тромб', 'соэ', 'скорость оседания',
            'глюкоза', 'глюк', 'сахар', 'холестерин', 'холест', 'триглицериды', 'тригл',
            'билирубин', 'билир', 'алт', 'аст', 'ггт', 'щелочная фосфатаза', 'щф',
            'креатинин', 'креат', 'мочевина', 'мочев', 'мочевая кислота', 'моч.кисл',
            'натрий', 'na', 'калий', 'k', 'хлор', 'cl', 'кальций', 'ca',
            'магний', 'mg', 'фосфор', 'p', 'железо', 'fe', 'ферритин', 'ферр',
            'трансферрин', 'трансф', 'вит.д', 'витамин д', 'вит.в12', 'витамин в12',
            'фолиевая кислота', 'фол.кисл', 'гомоцистеин', 'гомоцист',
            'ттг', 'т4', 'т3', 'ат-тпо', 'ат-тг', 'пролактин', 'прол',
            'эстрадиол', 'эстр', 'прогестерон', 'прог', 'тестостерон', 'тест',
            'кортизол', 'кортиз', 'актг', 'инсулин', 'инс', 'с-пептид', 'с-пет',
            'гликированный гемоглобин', 'гликир.гб', 'hba1c',
            
            # Диагностические сокращения
            'экг', 'эхокг', 'узи', 'кт', 'мрт', 'рентген', 'рентг',
            'фгдс', 'колоноскопия', 'колон', 'ирригоскопия', 'ирриг',
            'маммография', 'мамм', 'денситометрия', 'денсит',
            'спирография', 'спирогр', 'пневмотахометрия', 'пневмот',
            'ээг', 'реоэнцефалография', 'реоэг', 'эмг', 'электронейромиография',
            'холтер', 'суточное мониторирование', 'смэкг', 'см ад',
            
            # Фармакологические сокращения
            'мг/кг', 'мкг/кг', 'ед/кг', 'мл/кг', 'кап/кг', 'таб/кг',
            'мг/м2', 'мкг/м2', 'ед/м2', 'мл/м2',
            'р-р', 'раствор', 'сусп.', 'суспензия', 'эмульс.', 'эмульсия',
            'таб.', 'таблетка', 'капс.', 'капсула', 'амп.', 'ампула',
            'фл.', 'флакон', 'уп.', 'упаковка', 'бл.', 'блистер',
            'мазь', 'крем', 'гель', 'линимент', 'паста', 'порошок', 'пор',
            'капли', 'кап', 'спрей', 'аэрозоль', 'аэроз', 'ингалятор', 'ингал',
            'свечи', 'св', 'суппозитории', 'супп', 'микроклизма', 'микрокл',
            
            # Статусы и состояния
            'сост.', 'состояние', 'статус', 'стат', 'стабил.', 'стабильное',
            'критич.', 'критическое', 'тяж.', 'тяжелое', 'сред.', 'средней тяжести',
            'легк.', 'легкое', 'удовл.', 'удовлетворительное', 'неудовл.', 'неудовлетворительное',
            'комп.', 'компенсированное', 'субкомп.', 'субкомпенсированное', 'декомп.', 'декомпенсированное',
            'ремиссия', 'ремисс', 'обострение', 'обостр', 'рецидив', 'рецид',
            'хрон.', 'хроническое', 'ост.', 'острое', 'подост.', 'подострое',
            
            # Процедуры и манипуляции
            'инъекция', 'инъекц', 'в/в', 'в/м', 'п/к', 'п/о',
            'капельница', 'капельн', 'инфузия', 'инфуз', 'ингаляция', 'ингал',
            'промывание', 'промыв', 'спринцевание', 'спринц', 'клизма', 'кл',
            'перевязка', 'перевяз', 'обработка', 'обраб', 'массаж', 'масс',
            'физиотерапия', 'физио', 'лфк', 'лечебная физкультура',
            'дыхательная гимнастика', 'дых.гимн', 'механотерапия', 'механо',
            
            # Документация
            'история болезни', 'и.б.', 'амбулаторная карта', 'амб.карта',
            'медицинская карта', 'мед.карта', 'выписка', 'вып', 'справка', 'справ',
            'направление', 'направл', 'результат', 'рез-т', 'заключение', 'заключ',
            'протокол', 'проток', 'акт', 'отчет', 'отч', 'журнал', 'журн',
            'регистрация', 'регистр', 'учет', 'уч', 'статистика', 'стат',
            
            # Организационные
            'отделение', 'отд', 'палата', 'пал', 'кабинет', 'каб',
            'приемный покой', 'прием.пок', 'операционная', 'опер',
            'реанимация', 'реаним', 'физиотерапия', 'физио',
            'лаборатория', 'лаб', 'рентген', 'рентг', 'узи', 'узи',
            'консультация', 'конс', 'обход', 'обх', 'дежурство', 'деж',
            'смена', 'см', 'вахта', 'вахт', 'график', 'граф',
            
            # Дополнительные медицинские термины
            'аллергия', 'аллерг', 'анафилаксия', 'анафил', 'шок', 'шок',
            'инфекция', 'инф', 'воспаление', 'воспал', 'отек', 'отек',
            'кровотечение', 'кровот', 'геморрагия', 'геморр', 'тромбоз', 'тромб',
            'эмболия', 'эмбол', 'ишемия', 'иш', 'некроз', 'некр',
            'рубцевание', 'рубц', 'грануляция', 'гранул', 'эпителизация', 'эпител',
            'нагноение', 'нагн', 'сепсис', 'сепс', 'абсцесс', 'абсц',
            'флегмона', 'флегм', 'гангрена', 'гангр', 'язва', 'язв',
            'эрозия', 'эрозия', 'полип', 'полип', 'опухоль', 'опух',
            'метастаз', 'метаст', 'рецидив', 'рецид', 'ремиссия', 'ремисс'
        }
        
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
    
    def _init_medical_classifications(self, db: Database):
        """Инициализация медицинских классификаций из базы данных"""
        try:
            # Загрузка МКБ-10
            icd10_codes = db.execute("SELECT code, name FROM icd10_codes")
            self.medical_classifications['icd10'] = {
                (code, name.lower()) for code, name in icd10_codes
            }
            
            # Загрузка МКБ-11
            icd11_codes = db.execute("SELECT code, name FROM icd11_codes")
            self.medical_classifications['icd11'] = {
                (code, name.lower()) for code, name in icd11_codes
            }
            
            # Загрузка медицинских протоколов
            protocols = db.execute("SELECT name, description FROM medical_protocols")
            self.medical_classifications['medical_protocols'] = {
                (name.lower(), desc.lower()) for name, desc in protocols
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке медицинских классификаций: {e}")
            raise

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
        Расширенная валидация персональных данных
        
        Args:
            data_type: тип данных (name, address, etc.)
            value: найденное значение
            context: контекст, в котором найдено значение
            
        Returns:
            bool: является ли данное значение персональными данными
        """
        # Базовая проверка контекста
        if context in self.allowed_contexts:
            return False
            
        # Проверка на медицинские сокращения
        value_lower = value.lower().strip('., ')
        if value_lower in self.medical_abbreviations:
            return False
            
        # Получаем текст для проверки контекста
        text = self._get_context_text(value, context)
        
        # Применяем специфичные правила верификации
        if data_type in self.verification_rules:
            verification_results = []
            for rule in self.verification_rules[data_type]:
                try:
                    if data_type == 'name':
                        result = rule(text, text.find(value), text.find(value) + len(value))
                    else:
                        result = rule(value, text)
                    verification_results.append(result)
                except Exception as e:
                    self.logger.error(f"Ошибка при верификации {data_type}: {e}")
                    verification_results.append(False)
            
            # Требуем подтверждения всех правил
            if not all(verification_results):
                return False
        
        # Проверка порога уверенности
        if data_type in self.confidence_thresholds:
            confidence = self._calculate_confidence(data_type, value, context)
            if confidence < self.confidence_thresholds[data_type]:
                return False
        
        # Дополнительные проверки в зависимости от типа данных
        if data_type == 'name':
            # Проверка на медицинские термины
            if any(term in value_lower for term in self.medical_whitelist):
                return False
                
            # Проверка формата имени
            if not self._verify_name_format(value):
                return False
                
            # Проверка контекста
            if not self._verify_name_context(text, text.find(value), text.find(value) + len(value)):
                return False
                
            # Проверка частоты
            if not self._verify_name_frequency(value, text):
                return False
                
        elif data_type == 'diagnosis':
            # Проверка по МКБ
            if not self._verify_diagnosis_icd(value):
                return False
                
            # Проверка контекста
            if not self._verify_diagnosis_context(text, text.find(value), text.find(value) + len(value)):
                return False
                
            # Проверка согласованности
            if not self._verify_diagnosis_consistency(value, text):
                return False
                
        elif data_type == 'procedure':
            # Проверка по протоколам
            if not self._verify_procedure_protocol(value):
                return False
                
            # Проверка контекста
            if not self._verify_procedure_context(text, text.find(value), text.find(value) + len(value)):
                return False
                
            # Проверка согласованности
            if not self._verify_procedure_consistency(value, text):
                return False
        
        return True

    def _calculate_confidence(self, data_type: str, value: str, context: str) -> float:
        """
        Расчет уверенности в правильности определения данных
        
        Args:
            data_type: тип данных
            value: значение
            context: контекст
            
        Returns:
            float: значение уверенности от 0 до 1
        """
        confidence = 0.0
        weights = {
            'model_confidence': 0.4,
            'context_match': 0.3,
            'format_match': 0.2,
            'frequency_match': 0.1
        }
        
        # Уверенность модели
        if hasattr(self, f'_{data_type}_model_confidence'):
            confidence += weights['model_confidence'] * getattr(self, f'_{data_type}_model_confidence')(value)
        
        # Соответствие контексту
        if data_type in self.verification_rules:
            context_matches = sum(1 for rule in self.verification_rules[data_type] 
                                if rule(value, context))
            confidence += weights['context_match'] * (context_matches / len(self.verification_rules[data_type]))
        
        # Соответствие формату
        if hasattr(self, f'_verify_{data_type}_format'):
            confidence += weights['format_match'] * float(getattr(self, f'_verify_{data_type}_format')(value))
        
        # Соответствие частоте
        if hasattr(self, f'_verify_{data_type}_frequency'):
            confidence += weights['frequency_match'] * float(getattr(self, f'_verify_{data_type}_frequency')(value, context))
        
        return confidence

    def _get_context_text(self, value: str, context: str) -> str:
        """
        Получение контекстного текста для проверки
        
        Args:
            value: искомое значение
            context: контекст
            
        Returns:
            str: текст для проверки
        """
        # Здесь должна быть реализация получения текста из документа
        # В текущей реализации возвращаем контекст как есть
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
        Определяет контекст слова на основе его позиции в документе
        
        Args:
            match: найденное совпадение
            sections: словарь секций документа
            
        Returns:
            str: контекст слова
        """
        position = match.start
        for section_name, (start, end) in sections.items():
            if start <= position <= end:
                return section_name
        return 'unknown'

    def _is_sensitive_context(self, context: str) -> bool:
        """
        Проверяет, является ли контекст чувствительным для маскирования
        
        Args:
            context: контекст слова
            
        Returns:
            bool: нужно ли маскировать данные в этом контексте
        """
        sensitive_sections = {'patient_info', 'medical_data'}
        return context in sensitive_sections

    def _extract_names_with_natasha(self, text: str) -> List[Dict]:
        """Извлечение имен с помощью Natasha"""
        matches = list(self.names_extractor(text))
        results = []
        for match in matches:
            if match.fact:
                results.append({
                    'text': match.text,
                    'start': match.start,
                    'end': match.stop,
                    'confidence': 1.0,  # Natasha не предоставляет уверенность
                    'model': 'natasha',
                    'fact': {
                        'first': match.fact.first,
                        'last': match.fact.last,
                        'middle': match.fact.middle
                    }
                })
        return results
        
    def _extract_names_with_deep_pavlov(self, text: str) -> List[Dict]:
        """Извлечение имен с помощью DeepPavlov"""
        if not self.use_deep_pavlov:
            return []
            
        results = []
        tokens, tags, probs = self.deep_pavlov_model([text])
        
        current_name = []
        current_start = 0
        current_confidence = 0.0
        
        for token, tag, prob in zip(tokens[0], tags[0], probs[0]):
            if tag == 'B-PER' or (tag == 'I-PER' and current_name):
                if tag == 'B-PER' and current_name:
                    # Сохраняем предыдущее имя
                    results.append({
                        'text': ' '.join(current_name),
                        'start': current_start,
                        'end': current_start + len(' '.join(current_name)),
                        'confidence': current_confidence / len(current_name),
                        'model': 'deep_pavlov',
                        'fact': {'first': None, 'last': None, 'middle': None}
                    })
                    current_name = []
                
                if tag == 'B-PER':
                    current_start = text.find(token)
                    current_confidence = prob
                current_name.append(token)
                
        # Добавляем последнее имя
        if current_name:
            results.append({
                'text': ' '.join(current_name),
                'start': current_start,
                'end': current_start + len(' '.join(current_name)),
                'confidence': current_confidence / len(current_name),
                'model': 'deep_pavlov',
                'fact': {'first': None, 'last': None, 'middle': None}
            })
            
        return results
        
    def _extract_names_with_yargy(self, text: str) -> List[Dict]:
        """Извлечение имен с помощью Yargy"""
        if not self.use_yargy:
            return []
            
        results = []
        matches = self.yargy_parser.find(text)
        
        for match in matches:
            if match.fact:
                results.append({
                    'text': match.text,
                    'start': match.span[0],
                    'end': match.span[1],
                    'confidence': 1.0,  # Yargy не предоставляет уверенность
                    'model': 'yargy',
                    'fact': {
                        'first': match.fact.first,
                        'last': match.fact.last,
                        'middle': match.fact.middle
                    }
                })
        return results
        
    def _merge_model_results(self, results: List[List[Dict]]) -> List[Dict]:
        """
        Объединение результатов разных моделей с помощью голосования
        
        Args:
            results: список результатов от разных моделей
            
        Returns:
            List[Dict]: объединенные результаты
        """
        # Создаем словарь для подсчета голосов
        votes = {}
        
        # Подсчитываем голоса для каждого найденного имени
        for model_results in results:
            for result in model_results:
                key = (result['start'], result['end'])
                if key not in votes:
                    votes[key] = {
                        'text': result['text'],
                        'count': 0,
                        'confidence_sum': 0,
                        'models': set(),
                        'fact': result['fact']
                    }
                votes[key]['count'] += 1
                votes[key]['confidence_sum'] += result['confidence']
                votes[key]['models'].add(result['model'])
        
        # Фильтруем результаты
        merged_results = []
        for (start, end), vote in votes.items():
            # Проверяем, что имя найдено как минимум двумя моделями
            # или одной моделью с высокой уверенностью
            if (vote['count'] >= 2 or 
                (vote['count'] == 1 and vote['confidence_sum'] > self.confidence_threshold)):
                merged_results.append({
                    'text': vote['text'],
                    'start': start,
                    'end': end,
                    'confidence': vote['confidence_sum'] / vote['count'],
                    'models': list(vote['models']),
                    'fact': vote['fact']
                })
        
        return merged_results

    def _extract_data(self, text_data: List[Dict]) -> Dict:
        """
        Извлечение данных из распознанного текста
        
        Args:
            text_data: список распознанных слов
            
        Returns:
            Dict: извлеченные данные
        """
        # Объединяем все слова в один текст
        full_text = ' '.join(word['text'] for word in text_data)
        
        # Определяем секции документа
        sections = self._identify_document_sections(full_text)
        
        # Извлекаем имена с помощью разных моделей
        model_results = [
            self._extract_names_with_natasha(full_text),
            self._extract_names_with_deep_pavlov(full_text),
            self._extract_names_with_yargy(full_text)
        ]
        
        # Объединяем результаты
        merged_names = self._merge_model_results(model_results)
        
        sensitive_regions = []
        medical_data = {}
        
        # Обрабатываем объединенные результаты
        for name in merged_names:
            context = self._get_word_context(name, sections)
            
            # Проверяем, нужно ли маскировать это имя
            if self._is_sensitive_context(context) and self._validate_personal_data('name', name['text'], context):
                # Ищем слова, которые попадают в диапазон имени
                for word in text_data:
                    word_start = full_text.find(word['text'], name['start'])
                    if word_start >= name['start'] and word_start <= name['end']:
                        # Проверяем, не является ли это частью медицинского термина
                        if not any(term in word['text'].lower() for term in self.medical_whitelist):
                            sensitive_regions.append({
                                'type': 'name',
                                'text': word['text'],
                                'left': word['left'],
                                'top': word['top'],
                                'width': word['width'],
                                'height': word['height'],
                                'confidence': name['confidence'],
                                'models': name['models']
                            })
                            
                            # Логируем найденное имя
                            self._log_extraction_details(
                                text=full_text,
                                found_data={
                                    'data_type': 'name',
                                    'value': word['text'],
                                    'context': context,
                                    'confidence': name['confidence'],
                                    'models': name['models']
                                },
                                context=context
                            )
        
        return {
            'sensitive_regions': sensitive_regions,
            'medical_data': medical_data
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
            # ... существующий код ...
        
        # Добавляем проверку на пересечение регионов
        def is_overlapping(region1: Dict, region2: Dict) -> bool:
            return not (region1['left'] + region1['width'] < region2['left'] or
                       region2['left'] + region2['width'] < region1['left'] or
                       region1['top'] + region1['height'] < region2['top'] or
                       region2['top'] + region2['height'] < region1['top'])
        
        # Объединяем пересекающиеся регионы
        merged_regions = []
        for region in sensitive_regions:
            merged = False
            for existing in merged_regions:
                if is_overlapping(region, existing):
                    # Объединяем регионы
                    existing['left'] = min(existing['left'], region['left'])
                    existing['top'] = min(existing['top'], region['top'])
                    existing['width'] = max(existing['left'] + existing['width'],
                                          region['left'] + region['width']) - existing['left']
                    existing['height'] = max(existing['top'] + existing['height'],
                                           region['top'] + region['height']) - existing['top']
                    merged = True
                    break
            if not merged:
                merged_regions.append(region)

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
        logging.info(f"Принято решение о маскировании: {self._validate_personal_data(**found_data)}")

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