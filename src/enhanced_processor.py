from .medical_terms import MedicalTermsManager

class EnhancedDocumentProcessor:
    def __init__(self):
        # ... existing code ...
        self.medical_terms = MedicalTermsManager()
        # ... existing code ...

    def _should_mask_text(self, text: str) -> bool:
        """
        Проверяет, нужно ли маскировать текст
        
        Args:
            text: проверяемый текст
            
        Returns:
            bool: нужно ли маскировать текст
        """
        # Не маскируем медицинские термины
        if self.medical_terms.is_medical_term(text):
            return False
            
        # Остальная логика маскирования остается без изменений
        # ... existing code ... 