"""
Centralised logger factory.

Every module calls  setup_logger(__name__)  to get a consistently
configured logger that writes to:
  - Console (INFO and above)
  - logs/bot.log (DEBUG and above, rotating, max 5 MB × 3 backups)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring root handlers once."""
    global _configured

    if not _configured:
        os.makedirs(LOG_DIR, exist_ok=True)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # Console handler – INFO+
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(ch)

        # Rotating file handler – DEBUG+
        fh = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(fh)

        _configured = True

    return logging.getLogger(name)