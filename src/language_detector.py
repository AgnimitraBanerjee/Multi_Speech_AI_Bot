"""
LanguageDetector: Detects language from text and translates bot prompts.

Since SpeechRecognizer now handles the hard language detection work,
this module's detect() is only called on native-script text (which
langdetect handles perfectly) or as a secondary check.
"""

from langdetect import detect_langs, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
DetectorFactory.seed = 42


class LanguageDetector:

    LANGUAGE_NAMES = {
        "en": "English", "hi": "Hindi", "bn": "Bengali", "es": "Spanish",
        "fr": "French", "de": "German", "zh-cn": "Chinese (Simplified)",
        "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
        "ja": "Japanese", "ko": "Korean", "it": "Italian",
        "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "ur": "Urdu",
        "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam",
    }

    def __init__(self):
        self._translation_cache: dict = {}
        logger.info("LanguageDetector initialized.")

    def detect(self, text: str) -> str:
        """
        Detect language from text.
        Most reliable on native (non-ASCII) script.
        """
        if not text or len(text.strip()) < 2:
            return "en"

        try:
            results = detect_langs(text)
            top = results[0]
            if top.prob >= 0.80:
                name = self.LANGUAGE_NAMES.get(top.lang, top.lang)
                logger.info(f"Detected: {top.lang} ({name}) conf={top.prob:.3f}")
                return top.lang
        except LangDetectException as exc:
            logger.warning(f"LangDetect failed: {exc}")

        return "en"

    def translate(self, text: str, target_lang: str) -> str:
        """Translate text from English to target_lang."""
        if not text:
            return text

        target_lang = self._normalise(target_lang)

        if target_lang in ("en", "english"):
            return text

        cache_key = (text, target_lang)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        try:
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            if translated:
                self._translation_cache[cache_key] = translated
                logger.debug(f"Translated [{target_lang}]: '{text}' -> '{translated}'")
                return translated
        except Exception as exc:
            logger.warning(f"Translation failed [{target_lang}]: {exc}")

        return text

    @staticmethod
    def _normalise(code: str) -> str:
        mapping = {"zh-cn": "zh-CN", "zh-tw": "zh-TW", "auto": "en"}
        return mapping.get(code.lower(), code)