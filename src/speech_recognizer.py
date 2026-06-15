"""
SpeechRecognizer: Captures microphone audio and converts it to text.

Final Strategy
--------------
1. Call Google STT with en-US to get English transcription.
2. If language is unknown, also call with non-English codes to get native script.
3. CRITICAL VALIDATION: If a non-English STT returns native script, verify it is
   genuine by translating it back to English and comparing with the en-US result.
   If they are too similar (high overlap) → it was English phonetically
   transcribed, NOT a real non-English utterance. Reject it.
4. Only confirm a non-English language when the native-script result is
   semantically different from the English result.
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

PROBE_LANGUAGES = [
    "hi-IN", "bn-IN", "ur-PK", "mr-IN", "gu-IN",
    "ta-IN", "te-IN", "fr-FR", "de-DE", "es-ES",
    "ar-SA", "ru-RU", "ko-KR", "ja-JP",
]


def _is_native_script(text: str) -> bool:
    """True if text contains non-ASCII (native script) characters."""
    return any(ord(c) > 127 for c in text)


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Compute what fraction of words in text_a appear in text_b (case-insensitive).
    Used to detect if native-script result is just a phonetic echo of English.
    """
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    overlap = words_a.intersection(words_b)
    return len(overlap) / len(words_a)


def _translate_to_english(text: str, src_lang: str) -> str:
    """Translate native-script text back to English for validation."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=src_lang, target="en").translate(text)
        return result or ""
    except Exception as exc:
        logger.warning(f"Back-translation failed: {exc}")
        return ""


def _is_genuine_non_english(
    native_text: str,
    english_text: str,
    src_lang: str,
    overlap_threshold: float = 0.55,
) -> bool:
    """
    Validate that native_text is a genuine non-English utterance,
    NOT a phonetic transcription of English words.

    Method: translate native_text back to English, then compare with
    the original English STT result. High word overlap = phonetic echo = fake.

    Returns True only if the back-translated text is sufficiently different
    from the English transcription.
    """
    if not _is_native_script(native_text):
        return False

    back_translated = _translate_to_english(native_text, src_lang)
    logger.info(f"Back-translation [{src_lang}]: '{native_text}' -> '{back_translated}'")

    if not back_translated:
        # Can't validate — be conservative, reject
        return False

    overlap = _word_overlap_ratio(english_text.lower(), back_translated.lower())
    logger.info(
        f"Overlap between English STT and back-translation: {overlap:.2f} "
        f"(threshold={overlap_threshold})"
    )

    if overlap >= overlap_threshold:
        logger.info(
            f"HIGH OVERLAP ({overlap:.2f}) -> phonetic echo of English, rejecting as non-English"
        )
        return False

    logger.info(f"LOW OVERLAP ({overlap:.2f}) -> genuine non-English speech confirmed")
    return True


class SpeechRecognizer:
    """
    Wraps SpeechRecognition with validated multi-language probing.
    """

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
        Capture speech and return {"text": str, "lang": str} or None.

        lang="en"  → language unknown, probe + validate
        lang="hi"  → language known, call STT directly in that language
        """
        audio = self._capture_audio()
        if audio is None:
            return None

        if lang != "en":
            # Language confirmed — call STT directly
            bcp47 = GOOGLE_LANG_MAP.get(lang, "en-US")
            result = self._google_call(audio, bcp47)
            if result:
                result["lang"] = lang
                return result
            # Fallback to English if native STT fails
            result = self._google_call(audio, "en-US")
            if result:
                result["lang"] = "en"
                return result
            return self._vosk_recognise(audio)

        # Language unknown — probe with validation
        return self._probe_with_validation(audio)

    def _probe_with_validation(self, audio: sr.AudioData) -> dict | None:
        """
        Probe multiple languages and validate results to avoid
        mistaking phonetically-transcribed English for native speech.
        """
        # Step 1: Always get English transcription first
        en_result = self._google_call(audio, "en-US")
        english_text = en_result["text"] if en_result else ""
        logger.info(f"English STT: '{english_text}'")

        # Step 2: Probe each non-English language
        for bcp47 in PROBE_LANGUAGES:
            result = self._google_call(audio, bcp47)
            if not result:
                continue

            native_text = result["text"]

            if not _is_native_script(native_text):
                # Got ASCII back from non-English probe → not this language
                continue

            iso = self._bcp47_to_iso(bcp47)

            # Step 3: Validate — is this genuine or just phonetic echo?
            if _is_genuine_non_english(native_text, english_text, iso):
                logger.info(
                    f"GENUINE non-English confirmed: [{bcp47}] '{native_text}'"
                )
                return {"text": native_text, "lang": iso}
            else:
                logger.info(
                    f"Rejected [{bcp47}] '{native_text}' — phonetic echo of English"
                )
                # Continue probing other languages

        # No genuine non-English found → return English result
        if en_result:
            en_result["lang"] = "en"
            return en_result

        return self._vosk_recognise(audio)

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