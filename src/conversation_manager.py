"""
ConversationManager: Holds the ordered list of questions the bot asks.

Questions are defined in English; LanguageDetector translates them
before they are spoken/displayed.
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConversationManager:
    """
    Manages the structured conversation flow.

    Each question is a dict:
      {
        "key"      : str  – unique identifier (used as DB column key),
        "question" : str  – English text of the question,
      }
    """

    _QUESTIONS = [
        {
            "key": "full_name",
            "question": "May I know your full name, please?",
        },
        {
            "key": "person_to_meet",
            "question": "Whom would you like to meet today?",
        },
        {
            "key": "purpose",
            "question": "What is the purpose of your visit here?",
        },
        {
            "key": "origin",
            "question": "Where are you coming from?",
        },
    ]

    def __init__(self):
        logger.info(f"ConversationManager initialized with {len(self._QUESTIONS)} questions.")

    def get_questions(self) -> list[dict]:
        """Return the ordered list of question configs."""
        return list(self._QUESTIONS)

    def get_question_keys(self) -> list[str]:
        """Return just the question keys (useful for DB schema setup)."""
        return [q["key"] for q in self._QUESTIONS]