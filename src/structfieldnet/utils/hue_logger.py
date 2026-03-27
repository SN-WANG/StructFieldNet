# A colorful logging utility with ANSI support
# Author: Shengning Wang

import sys
import logging

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    tqdm = None
    _HAS_TQDM = False


class HueLogger:
    """Create a tqdm-friendly logger with ANSI colors."""

    b = "\033[1;34m"
    c = "\033[1;36m"
    m = "\033[1;35m"
    y = "\033[1;33m"
    g = "\033[1;32m"
    r = "\033[1;31m"
    q = "\033[0m"

    def __init__(self, name: str = __name__, level: int = logging.INFO) -> None:
        """Initialize the logger instance.

        Args:
            name: Logger name.
            level: Logging level.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = self._build_handler()
        self.logger.addHandler(handler)

    def _build_handler(self) -> logging.StreamHandler:
        """Build a stdout stream handler.

        Returns:
            Configured logging handler.
        """
        log_format = f"\033[90m%(asctime)s{self.q} - {self.b}%(levelname)s{self.q} - %(message)s"
        formatter = logging.Formatter(log_format, "%H:%M:%S")
        handler = logging.StreamHandler(sys.stdout)

        if _HAS_TQDM:
            handler.emit = lambda record: tqdm.write(formatter.format(record))

        handler.setFormatter(formatter)
        return handler


hue = HueLogger()
logger = hue.logger
