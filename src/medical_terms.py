"""
Модуль для работы с медицинскими терминами
"""

import json
import os
from typing import Set, Dict, List
import logging

class MedicalTermsManager:
    def __init__(self):
        """Инициализация менеджера медицинских терминов"""
        self.logger = logging.getLogger(__name__)
        self.terms: Set[str] = set()
        self.load_terms()
        
    def load_terms(self) -> None:
        """Загрузка медицинских терминов из JSON файлов"""
        data_dir = os.path.join('src', 'data')
        
        # Список файлов для загрузки
        json_files = [
            'russian_medical_terms_symptoms.json',
            'russian_medical_terms_diagnosis.json',
            'russian_medical_terms_anatomy.json',
            'russian_medical_terms_drugs.json',
            'russian_medical_terms_procedures.json',
            'russian_medical_terms_1.json',
            'english_stats_terms.json',
            'russian_medical_terms_whitelist.json',
            'russian_stats_terms.json',
            'english_medical_terms.json',
            # Добавляем новые файлы из project2
            'medical_terms_whitelist.json',
            'medical_abbreviations.json',
            'medical_units.json',
            'medical_procedures.json',
            'medical_diagnoses.json',
            'medical_symptoms.json',
            'medical_anatomy.json',
            'medical_drugs.json'
        ]
        
        # Добавляем базовые медицинские термины, если файлы не найдены
        basic_terms = {
            # Анатомические термины
            'голова', 'шея', 'грудь', 'спина', 'живот', 'таз', 'рука', 'нога',
            'кисть', 'стопа', 'палец', 'глаз', 'ухо', 'нос', 'рот', 'зуб',
            'язык', 'горло', 'пищевод', 'желудок', 'кишка', 'печень', 'почка',
            'сердце', 'легкое', 'мозг', 'позвоночник', 'кость', 'мышца', 'связка',
            'сустав', 'кровь', 'лимфа', 'нерв', 'сосуд', 'артерия', 'вена',
            
            # Медицинские процедуры
            'осмотр', 'пальпация', 'перкуссия', 'аускультация', 'рентген',
            'узи', 'кт', 'мрт', 'экг', 'эндоскопия', 'биопсия', 'операция',
            'перевязка', 'инъекция', 'капельница', 'массаж', 'физиотерапия',
            
            # Симптомы и состояния
            'боль', 'температура', 'кашель', 'насморк', 'тошнота', 'рвота',
            'диарея', 'запор', 'головокружение', 'слабость', 'усталость',
            'сонливость', 'бессонница', 'тревога', 'депрессия', 'стресс',
            
            # Диагнозы
            'грипп', 'орви', 'пневмония', 'бронхит', 'гастрит', 'язва',
            'гипертония', 'гипотония', 'диабет', 'артрит', 'остеохондроз',
            'сколиоз', 'мигрень', 'инсульт', 'инфаркт',
            
            # Лекарства и препараты
            'антибиотик', 'анальгетик', 'антисептик', 'витамин', 'гормон',
            'иммуномодулятор', 'пробиотик', 'фермент', 'антигистамин',
            
            # Единицы измерения
            'миллиметр', 'сантиметр', 'метр', 'миллилитр', 'литр',
            'миллиграмм', 'грамм', 'килограмм', 'миллимоль', 'моль',
            'миллиграмм-процент', 'миллиграмм-децилитр', 'миллиграмм-литр',
            'микрограмм-литр', 'наномоль-литр', 'миллимоль-литр',
            'единица-литр', 'международная-единица', 'миллиединица',
            'килоединица', 'миллиединица-миллилитр', 'грамм-литр',
            'пикограмм-миллилитр', 'фемтограмм', 'микрометр', 'нанометр',
            'паскаль', 'килопаскаль', 'миллиметр-ртутного-столба',
            'секунда', 'минута', 'час', 'сутки', 'неделя', 'месяц', 'год',
            'удар-в-минуту', 'дыхание-в-минуту', 'миллиметр-в-час',
            'грамм-в-сутки', 'миллиграмм-в-сутки', 'миллиграмм-на-килограмм',
            'микрограмм-на-килограмм', 'миллиграмм-процент',
            'микрограмм-миллилитр', 'наномоль-литр', 'микромоль-литр',
            'миллимоль-литр', 'моль-литр', 'единица-литр', 'международная-единица-литр',
            'килоединица-литр', 'миллиединица-литр', 'относительная-единица',
            'международное-нормализованное-отношение', 'протромбиновый-индекс',
            'активированное-частичное-тромбопластиновое-время', 'фибриноген',
            'тромбиновое-время', 'д-димер', 'растворимые-фибрин-мономерные-комплексы',
            'протромбиновое-время', 'активность-протромбина',
            
            # Аббревиатуры
            'оак', 'оам', 'бак', 'экг', 'узи', 'мрт', 'кт', 'рег', 'ээг',
            'фгдс', 'фкс', 'ифа', 'пцр', 'соэ', 'лдг', 'алт', 'аст', 'ггт',
            'щф', 'хс', 'лпнп', 'лпвп', 'тг', 'гп', 'сд', 'ад', 'чсс', 'чдд',
            'spo2', 'sao2', 'fio2', 'pao2', 'paco2', 'ph', 'be', 'hb', 'rbc',
            'wbc', 'plt', 'hct', 'mcv', 'mch', 'mchc', 'rdw', 'rdw-sd', 'rdw-cv',
            'pct', 'mpv', 'pdw', 'esr', 'crp', 'ast', 'alt', 'ggt', 'alp', 'tbil',
            'dbil', 'ibil', 'tp', 'alb', 'glob', 'urea', 'crea', 'glu', 'chol',
            'tg', 'hdl', 'ldl', 'vldl', 'ck', 'ck-mb', 'ldh', 'amyl', 'lipase',
            'na', 'k', 'cl', 'ca', 'p', 'mg', 'fe', 'ferr', 'tibc', 'uibc',
            'transferrin', 'crp', 'procalcitonin', 'lactate', 'crp', 'esr', 'rf',
            'aslo', 'ana', 'anca', 'dsdna', 'ena', 'ccp', 'apla', 'coags', 'pt',
            'inr', 'aptt', 'tt', 'fibrinogen', 'd-dimer', 'fdp', 'atiii', 'pc',
            'ps', 'ua', 'le', 'rbc', 'wbc', 'epi', 'cyl', 'crystals', 'bacteria',
            'yeast', 'ph', 'sg', 'prot', 'glu', 'ket', 'bil', 'uro', 'nit', 'le',
            'rbc', 'wbc', 'epi', 'cyl', 'csf', 'prot', 'glu', 'cells', 'cl',
            'lactate', 'ecg', 'eeg', 'emg', 'enmg', 'echocg', 'us', 'ct', 'mri',
            'x-ray', 'pet-ct', 'spect', 'fgds', 'fcs', 'colono', 'broncho',
            'gastro', 'laparos', 'thoracos', 'arthros', 'ecmo', 'ivl', 'cpap',
            'bipap', 'niv', 'hfnc', 'o2', 'iv', 'im', 'sc', 'po', 'pr', 'pv',
            'sl', 'td', 'ih', 'neb', 'et', 'io', 'it', 'cpr', 'aed', 'acls',
            'bls', 'pals', 'nrp', 'icd-10', 'icd-11', 'who', 'fda', 'ema', 'cdc',
            'nih', 'nice', 'sign', 'uptodate', 'medlineplus', 'pubmed', 'elibrary',
            'cyberleninka', 'google-scholar', 'web-of-science', 'scopus', 'ринц'
        }
        
        # Добавляем базовые термины
        self.terms.update(basic_terms)
        
        # Загружаем термины из файлов
        for file_name in json_files:
            try:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Извлекаем термины в зависимости от структуры JSON
                        if isinstance(data, dict):
                            # Если это словарь, берем все значения
                            terms = self._extract_terms_from_dict(data)
                        elif isinstance(data, list):
                            # Если это список, берем все элементы
                            terms = self._extract_terms_from_list(data)
                        else:
                            terms = set()
                            
                        self.terms.update(terms)
                        self.logger.info(f"Загружено {len(terms)} терминов из {file_name}")
                        
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке {file_name}: {str(e)}")
                
        self.logger.info(f"Всего загружено {len(self.terms)} уникальных медицинских терминов")
        
    def _extract_terms_from_dict(self, data: Dict) -> Set[str]:
        """Извлечение терминов из словаря"""
        terms = set()
        
        def extract_from_dict(d: Dict, current_path: List[str] = None):
            if current_path is None:
                current_path = []
                
            for key, value in d.items():
                if isinstance(value, dict):
                    extract_from_dict(value, current_path + [key])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            terms.add(item.lower())
                        elif isinstance(item, dict):
                            extract_from_dict(item, current_path + [key])
                elif isinstance(value, str):
                    terms.add(value.lower())
                    
        extract_from_dict(data)
        return terms
        
    def _extract_terms_from_list(self, data: List) -> Set[str]:
        """Извлечение терминов из списка"""
        terms = set()
        
        def extract_from_list(lst: List):
            for item in lst:
                if isinstance(item, str):
                    terms.add(item.lower())
                elif isinstance(item, dict):
                    for value in item.values():
                        if isinstance(value, str):
                            terms.add(value.lower())
                        elif isinstance(value, list):
                            extract_from_list(value)
                            
        extract_from_list(data)
        return terms
        
    def is_medical_term(self, text: str) -> bool:
        """
        Проверка, является ли текст медицинским термином
        
        Args:
            text: проверяемый текст
            
        Returns:
            bool: является ли текст медицинским термином
        """
        return text.lower() in self.terms
        
    def get_all_terms(self) -> Set[str]:
        """
        Получение всех медицинских терминов
        
        Returns:
            Set[str]: множество всех медицинских терминов
        """
        return self.terms.copy() 