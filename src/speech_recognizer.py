"""
SpeechRecognizer — Definitive Multilingual Architecture
========================================================

Detection order (CRITICAL — PATH B must come before PATH A):

PATH B — Indian languages (checked FIRST)
  Hindi, Bengali, Urdu, Tamil, etc. use native scripts but are often
  spoken in Romanized form. langdetect CANNOT detect these reliably.
  We detect them by matching en-US transcription against a curated
  list of Romanized Indian vocabulary. On match, we probe hi-IN/bn-IN
  to get native-script transcription.

PATH A — European/other Roman-script languages (checked SECOND)
  French, German, Spanish, Russian etc. langdetect handles perfectly
  when text is already in Roman/Cyrillic script from en-US STT.
  BUT we only reach here if PATH B did not match, preventing
  langdetect from misidentifying Indian Romanized text as Dutch/Indonesian.

PATH C — English (default)
  No Indian markers, no high-confidence foreign language detected.
"""

import speech_recognition as sr
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

GOOGLE_LANG_MAP = {
    "hi": "hi-IN", "bn": "bn-IN", "mr": "mr-IN", "gu": "gu-IN",
    "ta": "ta-IN", "te": "te-IN", "ml": "ml-IN", "pa": "pa-IN",
    "ur": "ur-PK", "ar": "ar-SA", "fr": "fr-FR", "de": "de-DE",
    "es": "es-ES", "pt": "pt-BR", "ru": "ru-RU", "ja": "ja-JP",
    "ko": "ko-KR", "zh-cn": "zh-CN", "zh-tw": "zh-TW", "it": "it-IT",
    "en": "en-US",
}

# Languages that langdetect can detect reliably from Roman/Cyrillic script
# These are ONLY reached if no Indian markers are found first
ROMAN_DETECTABLE_LANGS = {
    "fr", "de", "es", "pt", "it", "ru", "nl", "pl", "sv", "da",
    "fi", "no", "cs", "sk", "ro", "hu", "tr", "vi",
}

# Indian language probing order (most common first)
INDIAN_LANG_PROBES = [
    ("hi", "hi-IN"),   # Hindi
    ("bn", "bn-IN"),   # Bengali
    ("ur", "ur-PK"),   # Urdu
    ("mr", "mr-IN"),   # Marathi
    ("gu", "gu-IN"),   # Gujarati
    ("ta", "ta-IN"),   # Tamil
    ("te", "te-IN"),   # Telugu
    ("ml", "ml-IN"),   # Malayalam
    ("pa", "pa-IN"),   # Punjabi
]

# ---------------------------------------------------------------------------
# Hindi Romanized vocabulary — reliable indicators of Hindi speech
# ---------------------------------------------------------------------------
HINDI_ROMAN_MARKERS = {
    "mera", "meri", "mere", "main", "mai", "hoon", "hun", "hai", "hain",
    "naam", "aap", "tum", "tu", "woh", "wo", "yeh", "ye", "kya", "kaun",
    "kahan", "kaise", "kyun", "nahi", "nahin", "haan", "han", "aur",
    "mujhe", "tumhe", "unhe", "milna", "milana", "chahta", "chahti",
    "chahte", "jana", "aana", "karna", "dena", "lena", "bolna", "chahiye",
    "bahut", "thoda", "accha", "achha", "theek", "shukriya", "dhanyavaad",
    "namaste", "bhai", "didi", "bhaiya", "apna", "apni", "unka", "unki",
    "humara", "tumhara", "iska", "uska", "idhar", "udhar", "yahan",
    "wahan", "abhi", "phir", "sirf", "bas", "lekin", "magar", "par",
    "tujhse", "usse", "inse", "unse", "saath", "baad", "pehle", "kaafi",
    "bohot", "zaroor", "kyunki", "isliye", "matlab", "samajh", "dekho",
    "suno", "bolo", "jao", "aao", "karo", "lao", "do", "raho", "chalte",
}

# ---------------------------------------------------------------------------
# Bengali Romanized vocabulary — reliable indicators of Bengali speech
# ---------------------------------------------------------------------------
BENGALI_ROMAN_MARKERS = {
    # Pronouns and basic words
    "amar", "hamar", "ami", "tumi", "apni", "aar", "ebong",
    "aache", "ache", "nei", "nai", "haan", "na",
    # Question words
    "ki", "ke", "keno", "kothay", "kothai", "kobe", "kemon", "kতটা",
    "kakhon", "kotota",
    # Verbs (present/future/past forms)
    "jabo", "ashbo", "korbo", "bolbo", "sunbo", "dekho", "khabo",
    "thakbo", "jaben", "asben", "korben", "bolben", "sunben", "deben",
    "jacchi", "aschhi", "korchi", "bolchi", "sunchi", "dekhchi",
    "thakchi", "khachhi", "porchi", "likhchi", "shunchi", "jachhi",
    "jete", "ashte", "korte", "bolte", "sunte", "dekhte", "khete",
    "thakte", "porte", "likhte",
    # Nouns and common words
    "naam", "bari", "barite", "bangla", "bhasha", "kotha", "katha",
    "manush", "lok", "chhele", "meye", "baba", "maa", "dada", "didi",
    "bondhu", "school", "college", "office", "kaj", "shomoy", "din",
    "raat", "shondha", "shokal",
    # Postpositions/particles
    "theke", "niye", "diye", "hoye", "giye", "eshe", "kore", "bole",
    "hobe", "hoyeche", "hoyechilo",
    # Adjectives
    "bhalo", "mondo", "manda", "boro", "chhoto", "noto", "shundor",
    "ektu", "onek", "sob", "keu", "tahole", "kintu",
    # Greetings/politeness
    "dhanyabad", "namaskar", "nomoshkar",
    # Connector words
    "tahole", "kintu", "tobe", "noyto", "nahole", "karon", "jehetu",
    # Object forms
    "amake", "tomake", "apnake", "tader", "amader", "tomader",
    "tar", "tার", "oder", "amar",
}

ALL_INDIAN_MARKERS = HINDI_ROMAN_MARKERS | BENGALI_ROMAN_MARKERS


def _is_native_script(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _check_indian_markers(text: str):
    """
    Check if text contains Indian Romanized vocabulary.
    Returns ('bn_first', hindi_hits, bengali_hits) or (None, set(), set())
    """
    words = set(text.lower().split())
    hindi_hits = words & HINDI_ROMAN_MARKERS
    bengali_hits = words & BENGALI_ROMAN_MARKERS

    if hindi_hits or bengali_hits:
        logger.info(
            f"Indian Roman markers — Hindi: {hindi_hits}, Bengali: {bengali_hits}"
        )
        # If more Bengali-specific hits, probe Bengali first
        bn_only = bengali_hits - HINDI_ROMAN_MARKERS  # words only in Bengali set
        if len(bn_only) >= 1 or len(bengali_hits) > len(hindi_hits):
            return "bn_first", hindi_hits, bengali_hits
        return "hi_first", hindi_hits, bengali_hits

    return None, set(), set()


class SpeechRecognizer:

    def __init__(
        self,
        timeout: int = 10,
        phrase_time_limit: int = 15,
        energy_threshold: int = 300,
        pause_threshold: float = 1.0,
    ):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self._vosk_model = None
        logger.info("SpeechRecognizer ready.")

    def listen(self, lang: str = "en") -> dict | None:
        """
        lang="en"  → unknown, run full detection
        lang="hi"  → confirmed Hindi, use hi-IN directly
        lang="bn"  → confirmed Bengali, use bn-IN directly
        etc.
        """
        audio = self._capture_audio()
        if audio is None:
            return None

        if lang != "en":
            return self._listen_known_language(audio, lang)

        return self._listen_unknown_language(audio)

    # ------------------------------------------------------------------
    # Known language — direct STT call
    # ------------------------------------------------------------------

    def _listen_known_language(self, audio: sr.AudioData, lang: str) -> dict | None:
        bcp47 = GOOGLE_LANG_MAP.get(lang, "en-US")
        result = self._google_call(audio, bcp47)
        if result:
            result["lang"] = lang
            return result
        # Fallback to English if native STT fails
        result = self._google_call(audio, "en-US")
        if result:
            result["lang"] = lang
            return result
        return self._vosk_recognise(audio)

    # ------------------------------------------------------------------
    # Unknown language — three-path detection
    # ------------------------------------------------------------------

    def _listen_unknown_language(self, audio: sr.AudioData) -> dict | None:
        # Step 1: Get English transcription (always first)
        en_result = self._google_call(audio, "en-US")
        en_text = en_result["text"] if en_result else ""
        logger.info(f"en-US result: '{en_text}'")

        if not en_text:
            return self._vosk_recognise(audio)

        # ---------------------------------------------------------------
        # PATH B — Indian languages (CHECKED FIRST — before langdetect)
        # langdetect cannot handle Romanized Indian text and would
        # misidentify it as Indonesian, Dutch, etc.
        # ---------------------------------------------------------------
        indian_hint, hindi_hits, bengali_hits = _check_indian_markers(en_text)
        if indian_hint:
            logger.info(f"PATH B: Indian markers found ({indian_hint}), probing native STT...")
            native_result = self._probe_indian_languages(audio, indian_hint)
            if native_result:
                return native_result
            # Probing got no native script — still mark as Indian if strong signal
            if len(hindi_hits) >= 2 and not bengali_hits:
                logger.info("PATH B fallback: strong Hindi markers, returning as hi")
                if en_result:
                    en_result["lang"] = "hi"
                    return en_result
            if len(bengali_hits) >= 2:
                logger.info("PATH B fallback: strong Bengali markers, returning as bn")
                if en_result:
                    en_result["lang"] = "bn"
                    return en_result

        # ---------------------------------------------------------------
        # PATH A — European/other Roman-script languages (CHECKED SECOND)
        # Only reached when no Indian markers found
        # ---------------------------------------------------------------
        roman_lang = self._detect_roman_language(en_text)
        if roman_lang and roman_lang in ROMAN_DETECTABLE_LANGS:
            logger.info(f"PATH A: Roman-script language detected: {roman_lang}")
            if en_result:
                en_result["lang"] = roman_lang
                return en_result

        # ---------------------------------------------------------------
        # PATH C — English (default)
        # ---------------------------------------------------------------
        logger.info("PATH C: No non-English markers found, staying English.")
        if en_result:
            en_result["lang"] = "en"
            return en_result

        return self._vosk_recognise(audio)

    def _detect_roman_language(self, text: str) -> str | None:
        """Run langdetect on Roman/Cyrillic text with high confidence threshold."""
        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 42
            results = detect_langs(text)
            top = results[0]
            if top.prob >= 0.92 and top.lang not in ("en", "id", "ms"):
                logger.info(f"langdetect Roman: {top.lang} conf={top.prob:.3f}")
                return top.lang
        except Exception as exc:
            logger.warning(f"langdetect failed: {exc}")
        return None

    def _probe_indian_languages(self, audio: sr.AudioData, hint: str) -> dict | None:
        """
        Probe Indian STT endpoints. Bengali-first or Hindi-first based on hint.
        Returns result with native script, or None.
        """
        if hint == "bn_first":
            ordered = [("bn", "bn-IN"), ("hi", "hi-IN")] + [
                p for p in INDIAN_LANG_PROBES if p[0] not in ("bn", "hi")
            ]
        else:
            ordered = INDIAN_LANG_PROBES  # Hindi first

        for iso, bcp47 in ordered:
            result = self._google_call(audio, bcp47)
            if result and _is_native_script(result["text"]):
                result["lang"] = iso
                logger.info(
                    f"Native script confirmed [{bcp47}]: '{result['text']}' -> {iso}"
                )
                return result

        logger.warning("Indian probe: no native script returned from any endpoint.")
        return None

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _capture_audio(self) -> sr.AudioData | None:
        with sr.Microphone() as source:
            try:
                logger.debug("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                logger.debug("Listening...")
                return self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                logger.warning("No speech detected within timeout.")
                return None
            except Exception as exc:
                logger.error(f"Microphone capture error: {exc}")
                return None

    def _google_call(self, audio: sr.AudioData, bcp47: str) -> dict | None:
        try:
            text = self.recognizer.recognize_google(audio, language=bcp47)
            if text and text.strip():
                logger.info(f"Google STT [{bcp47}]: '{text.strip()}'")
                return {"text": text.strip(), "lang": "en"}
            return None
        except sr.UnknownValueError:
            logger.debug(f"Google STT [{bcp47}]: could not understand.")
            return None
        except sr.RequestError as exc:
            logger.error(f"Google STT [{bcp47}] request error: {exc}")
            return None

    @staticmethod
    def _bcp47_to_iso(bcp47: str) -> str:
        reverse = {v: k for k, v in GOOGLE_LANG_MAP.items()}
        return reverse.get(bcp47, "en")

    def _vosk_recognise(self, audio: sr.AudioData) -> dict | None:
        try:
            from vosk import Model, KaldiRecognizer # type: ignore
            import json, wave, io
            if self._vosk_model is None:
                import os
                model_path = "models/vosk-model-small-en"
                if not os.path.exists(model_path):
                    logger.warning("Vosk model not found. Skipping.")
                    return None
                self._vosk_model = Model(model_path)
            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
            wf = wave.open(io.BytesIO(wav_data))
            rec = KaldiRecognizer(self._vosk_model, wf.getframerate())
            rec.AcceptWaveform(wf.readframes(wf.getnframes()))
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            if text:
                return {"text": text, "lang": "en"}
            return None
        except ImportError:
            logger.warning("Vosk not installed. Skipping.")
            return None
        except Exception as exc:
            logger.error(f"Vosk error: {exc}")
            return None