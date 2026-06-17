"""
BotController: Orchestrates the full conversation lifecycle.

Session Language Flow
---------------------
- detected_lang starts as "en"
- lang_confirmed starts as False

Every question:
  - If lang_confirmed=False: listen with lang="en" (triggers full detection)
  - If lang_confirmed=True:  listen with lang=detected_lang (native STT)

Language confirmation:
  - SpeechRecognizer returns lang != "en" in result dict -> confirmed
  - Once confirmed, all subsequent questions use native STT
"""

from src.speech_recognizer import SpeechRecognizer
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
        """Main event loop."""
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
        """Run one complete visitor session."""
        session_id = self.db_manager.create_session()
        logger.info(f"Session started: {session_id}")

        detected_lang = "en"
        lang_confirmed = False

        # --- Greeting ---
        greeting = "Hello, welcome Here!"
        print(f"\n[BOT] {greeting}")
        self.tts.speak(greeting, lang="en")
        self.db_manager.log_event(session_id, "greeting", None, greeting, "en")

        questions = self.conversation_manager.get_questions()

        for q_index, q_data in enumerate(questions):
            question_key = q_data["key"]
            question_template = q_data["question"]

            # Translate question into detected language
            translated_q = self.language_detector.translate(
                question_template, target_lang=detected_lang
            )
            print(f"\n[BOT] {translated_q}")
            self.tts.speak(translated_q, lang=detected_lang)

            # Listen — pass "en" if language not yet confirmed (triggers full detection)
            # Pass detected_lang if confirmed (uses native STT directly)
            stt_lang = detected_lang if lang_confirmed else "en"
            answer_text, answer_lang = self._listen_with_retries(
                session_id, q_index, stt_lang
            )

            if answer_text is None:
                answer_text = "[No response detected]"
                answer_lang = detected_lang

            # --- Language confirmation ---
            if not lang_confirmed and answer_lang and answer_lang != "en":
                detected_lang = answer_lang
                lang_confirmed = True
                lang_name = self.language_detector.LANGUAGE_NAMES.get(
                    detected_lang, detected_lang
                )
                logger.info(
                    f"Language CONFIRMED: {detected_lang} ({lang_name}) at Q{q_index + 1}"
                )
                print(f"\n[LANG CONFIRMED] {lang_name} ({detected_lang}) ✓")

            print(f"[USER] {answer_text}  (lang: {detected_lang})")

            # Log to database
            self.db_manager.log_answer(
                session_id=session_id,
                question_key=question_key,
                question_text=translated_q,
                answer_text=answer_text,
                detected_lang=detected_lang,
            )

        # --- Farewell ---
        farewell = "Thank you for coming here, welcome!"
        translated_farewell = self.language_detector.translate(
            farewell, target_lang=detected_lang
        )
        print(f"\n[BOT] {translated_farewell}\n")
        self.tts.speak(translated_farewell, lang=detected_lang)
        self.db_manager.log_event(
            session_id, "farewell", None, translated_farewell, detected_lang
        )
        self.db_manager.close_session(session_id, detected_lang)
        logger.info(f"Session {session_id} complete. lang={detected_lang}")
        print("-" * 60)

    def _listen_with_retries(
        self,
        session_id: str,
        q_index: int,
        lang: str = "en",
        max_retries: int = 3,
    ):
        """Listen with up to max_retries attempts. Returns (text, lang)."""
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