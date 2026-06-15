"""
BotController: Orchestrates the full conversation lifecycle.

Language Detection Flow
-----------------------
Q1: SpeechRecognizer probes ALL languages simultaneously.
    If native script returned -> language confirmed immediately.
    If English returned -> continue in English, keep probing on Q2.

Q2+: If language confirmed -> listen directly in that language (fast, native).
     If still unknown -> keep probing until confirmed or session ends.
"""

import sys
import os

from src.speech_recognizer import SpeechRecognizer, PROBE_LANGUAGES, _is_native_script
from src.language_detector import LanguageDetector
from src.text_to_speech import TextToSpeech
from src.conversation_manager import ConversationManager
from src.database_manager import DatabaseManager
from src.utils.logger import setup_logger
from src.utils.keyboard_input import get_keypress

logger = setup_logger(__name__)


class BotController:
    def __init__(self):
        logger.info("Initializing BotController and all modules...")
        self.speech_recognizer = SpeechRecognizer()
        self.language_detector = LanguageDetector()
        self.tts = TextToSpeech()
        self.conversation_manager = ConversationManager()
        self.db_manager = DatabaseManager()
        self._running = True
        logger.info("BotController initialized successfully.")

    def run(self):
        while self._running:
            print("\n[WAITING] Press  S / 1  to Start  |  Q / 2  to Quit\n")
            key = get_keypress()
            if key in ("s", "1"):
                logger.info("User pressed START key.")
                self._start_session()
            elif key in ("q", "2"):
                logger.info("User pressed QUIT key.")
                print("\nGoodbye! Exiting the bot. Have a great day!\n")
                self._running = False
            else:
                print(f"  Unknown key '{key}'. Please press S/1 or Q/2.")

    def _start_session(self):
        session_id = self.db_manager.create_session()
        logger.info(f"Session started: {session_id}")

        detected_lang = "en"
        lang_confirmed = False

        # Greeting
        greeting = "Hello, welcome Here!"
        print(f"\n[BOT] {greeting}")
        self.tts.speak(greeting, lang="en")
        self.db_manager.log_event(session_id, "greeting", None, greeting, "en")

        questions_config = self.conversation_manager.get_questions()

        for q_index, q_data in enumerate(questions_config):
            question_key = q_data["key"]
            question_template = q_data["question"]

            # Ask question in current language
            translated_q = self.language_detector.translate(
                question_template, target_lang=detected_lang
            )
            print(f"\n[BOT] {translated_q}")
            self.tts.speak(translated_q, lang=detected_lang)

            # Listen: probe all languages if not confirmed, else use known lang
            stt_lang = detected_lang if lang_confirmed else "en"
            answer_text, stt_detected_lang = self._listen_with_retries(
                session_id, q_index, stt_lang
            )

            if answer_text is None:
                answer_text = "[No response detected]"
                stt_detected_lang = detected_lang

            # --- Language confirmation logic ---
            if not lang_confirmed:
                # Case 1: SpeechRecognizer found native script and identified language
                if stt_detected_lang and stt_detected_lang != "en":
                    detected_lang = stt_detected_lang
                    lang_confirmed = True
                    lang_name = self.language_detector.LANGUAGE_NAMES.get(
                        detected_lang, detected_lang
                    )
                    logger.info(f"Language confirmed via native script: {detected_lang} ({lang_name})")
                    print(f"\n[LANG CONFIRMED] {lang_name} ({detected_lang}) ✓")

                # Case 2: Check if returned text itself has native script
                elif _is_native_script(answer_text):
                    candidate = self.language_detector.detect(answer_text)
                    if candidate != "en":
                        detected_lang = candidate
                        lang_confirmed = True
                        lang_name = self.language_detector.LANGUAGE_NAMES.get(
                            detected_lang, detected_lang
                        )
                        logger.info(f"Language confirmed via text analysis: {detected_lang}")
                        print(f"\n[LANG CONFIRMED] {lang_name} ({detected_lang}) ✓")

                # Case 3: Still English
                else:
                    logger.info(f"Q{q_index+1}: Language unconfirmed, staying 'en'")

            print(f"[USER] {answer_text}  (lang: {detected_lang})")

            # Log answer
            self.db_manager.log_answer(
                session_id=session_id,
                question_key=question_key,
                question_text=translated_q,
                answer_text=answer_text,
                detected_lang=detected_lang,
            )

            # If language just confirmed on this question, re-translate and
            # re-ask the remaining questions (handled in next loop iteration)

        # Farewell
        farewell = "Thank you for coming here, welcome!"
        translated_farewell = self.language_detector.translate(
            farewell, target_lang=detected_lang
        )
        print(f"\n[BOT] {translated_farewell}\n")
        self.tts.speak(translated_farewell, lang=detected_lang)
        self.db_manager.log_event(session_id, "farewell", None, translated_farewell, detected_lang)
        self.db_manager.close_session(session_id, detected_lang)
        logger.info(f"Session {session_id} complete. lang={detected_lang}")
        print("-" * 60)

    def _listen_with_retries(
        self, session_id: str, q_index: int, lang: str = "en", max_retries: int = 3
    ):
        for attempt in range(1, max_retries + 1):
            print(f"  [Listening... attempt {attempt}/{max_retries}]")
            result = self.speech_recognizer.listen(lang=lang)
            if result:
                return result["text"], result.get("lang", lang)
            if attempt < max_retries:
                retry_msg = "Sorry, I didn't catch that. Please speak again."
                print(f"[BOT] {retry_msg}")
                self.tts.speak(retry_msg, lang="en")
        return None, None