import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format and level for all modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr
    )
