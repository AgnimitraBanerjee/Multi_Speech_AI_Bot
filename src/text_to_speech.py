"""
TextToSpeech: Converts bot responses to spoken audio.

Primary engine : gTTS (Google Text-to-Speech) – multilingual, natural voice
Fallback engine: pyttsx3 – fully offline, English only
"""

import os
import tempfile
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextToSpeech:
    """
    Speaks text aloud using the best available TTS engine.

    Precedence
    ----------
    1. gTTS  (online, multilingual) → plays via pygame or playsound
    2. pyttsx3 (offline, English)   → speaks directly
    """

    def __init__(self):
        self._gtts_available = self._check_gtts()
        self._pyttsx3_engine = None
        if not self._gtts_available:
            self._pyttsx3_engine = self._init_pyttsx3()
        logger.info(
            f"TextToSpeech ready. "
            f"gTTS={'YES' if self._gtts_available else 'NO'}, "
            f"pyttsx3={'YES' if self._pyttsx3_engine else 'NO'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, lang: str = "en") -> None:
        """Convert *text* to speech in *lang* and play it."""
        if not text:
            return

        lang = self._normalise_lang(lang)

        if self._gtts_available:
            self._speak_gtts(text, lang)
        elif self._pyttsx3_engine:
            self._speak_pyttsx3(text)
        else:
            logger.warning("No TTS engine available. Printing text only.")
            print(f"  [TTS unavailable] {text}")

    # ------------------------------------------------------------------
    # Private: gTTS
    # ------------------------------------------------------------------

    @staticmethod
    def _check_gtts() -> bool:
        try:
            import gtts  # noqa: F401
            return True
        except ImportError:
            return False

    def _speak_gtts(self, text: str, lang: str) -> None:
        try:
            from gtts import gTTS

            # gTTS expects BCP-47 codes; normalise zh variants
            gtts_lang = self._gtts_lang_code(lang)
            tts = gTTS(text=text, lang=gtts_lang, slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                tmp_path = fp.name
            tts.save(tmp_path)

            self._play_audio(tmp_path)
            os.unlink(tmp_path)
        except Exception as exc:
            logger.error(f"gTTS error: {exc}")
            # Fall through to pyttsx3 if available
            if self._pyttsx3_engine:
                self._speak_pyttsx3(text)

    @staticmethod
    def _play_audio(path: str) -> None:
        """Play an audio file. Tries pygame → playsound → os default."""
        # 1. pygame
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            return
        except Exception:
            pass

        # 2. playsound
        try:
            from playsound import playsound
            playsound(path)
            return
        except Exception:
            pass

        # 3. OS default (Linux: aplay/mpg123, macOS: afplay, Windows: start)
        import platform
        system = platform.system()
        try:
            if system == "Darwin":
                os.system(f"afplay '{path}'")
            elif system == "Windows":
                os.system(f'start /wait "" "{path}"')
            else:
                # Try mpg123 then aplay (for mp3)
                ret = os.system(f"mpg123 -q '{path}' 2>/dev/null")
                if ret != 0:
                    os.system(f"aplay '{path}' 2>/dev/null")
        except Exception as exc:
            logger.error(f"Audio playback failed: {exc}")

    # ------------------------------------------------------------------
    # Private: pyttsx3
    # ------------------------------------------------------------------

    @staticmethod
    def _init_pyttsx3():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 0.9)
            return engine
        except Exception as exc:
            logger.warning(f"pyttsx3 init failed: {exc}")
            return None

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as exc:
            logger.error(f"pyttsx3 speak error: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_lang(code: str) -> str:
        mapping = {"auto": "en", "zh-cn": "zh-CN", "zh-tw": "zh-TW"}
        return mapping.get(code.lower(), code)

    @staticmethod
    def _gtts_lang_code(code: str) -> str:
        """gTTS uses two-letter codes; strip region suffix when needed."""
        # gTTS supports 'zh-TW', 'zh-CN', etc. so keep those.
        # For others, strip the region part (e.g. 'en-US' → 'en').
        if code.lower().startswith("zh"):
            return code
        return code.split("-")[0].split("_")[0]