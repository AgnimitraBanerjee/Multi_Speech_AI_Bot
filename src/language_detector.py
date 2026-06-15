"""
LanguageDetector: Detects language from native-script text and translates
bot prompts into that language.

Since SpeechRecognizer now returns native script directly (via multi-language
probing), detection here is simple and reliable — non-ASCII text is trivially
identified by langdetect with very high accuracy.
"""

from langdetect import detect_langs, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DetectorFactory.seed = 42

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "es": "Spanish",
    "fr": "French", "de": "German", "zh-cn": "Chinese (Simplified)",
    "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "it": "Italian",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "ur": "Urdu",
    "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam",
}

_ENGLISH_MARKERS = {
    "my", "name", "is", "i", "am", "the", "a", "an", "to", "of",
    "and", "in", "it", "you", "he", "she", "we", "they", "this",
    "that", "what", "who", "where", "from", "here", "want", "meet",
    "purpose", "visit", "hello", "hi", "yes", "no", "please", "thank",
    "coming", "going", "speak", "again", "sorry", "are", "for", "with",
}


def _is_native_script(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


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

        If text contains native script → langdetect is highly accurate.
        If text is pure ASCII → apply English marker check first.
        """
        if not text or len(text.strip()) < 2:
            return "en"

        # Native script present → run langdetect directly (very reliable)
        if _is_native_script(text):
            try:
                lang_probs = detect_langs(text)
                top = lang_probs[0]
                logger.info(f"Native script detection: {lang_probs}")
                if top.prob >= 0.80:
                    name = LANGUAGE_NAMES.get(top.lang, top.lang)
                    logger.info(f"Detected: {top.lang} ({name}) conf={top.prob:.2f}")
                    return top.lang
            except LangDetectException as exc:
                logger.warning(f"LangDetect failed on native script: {exc}")
            return "en"

        # Pure ASCII text — use English marker check
        words = text.lower().split()
        if _ENGLISH_MARKERS.intersection(set(words)):
            logger.info(f"English markers found -> 'en'")
            return "en"

        # ASCII but no English markers — try langdetect with higher threshold
        try:
            lang_probs = detect_langs(text)
            top = lang_probs[0]
            logger.info(f"ASCII langdetect: {lang_probs}")
            if top.prob >= 0.95 and top.lang != "en":
                logger.info(f"High confidence ASCII detection: {top.lang} ({top.prob:.2f})")
                return top.lang
        except LangDetectException:
            pass

        return "en"

    def translate(self, text: str, target_lang: str) -> str:
        if not text:
            return text
        target_lang = self._normalise_lang_code(target_lang)
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
            logger.warning(f"Translation failed for '{target_lang}': {exc}. Using original.")
        return text

    @staticmethod
    def _normalise_lang_code(code: str) -> str:
        mapping = {"zh-cn": "zh-CN", "zh-tw": "zh-TW", "auto": "en"}
        return mapping.get(code.lower(), code)