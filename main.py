"""
Multilingual AI Speech Recognition Bot
Entry point - orchestrates all modules
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bot_controller import BotController
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point for the Multilingual Speech Recognition Bot."""
    print("\n" + "="*60)
    print("   MULTILINGUAL AI SPEECH RECOGNITION BOT")
    print("="*60)
    print("  Press  S  or  1  to START a new conversation")
    print("  Press  Q  or  2  to QUIT the application")
    print("="*60 + "\n")

    controller = BotController()
    controller.run()


if __name__ == "__main__":
    main()