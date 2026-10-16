import os
import sys
from datetime import datetime

from loguru import logger as _logger

from app.config import PROJECT_ROOT


_print_level = "INFO"
_current_log_name: str | None = None
_LOG_NAME_ENV = "MAS_SYS_SHARED_LOG_NAME"


def _resolve_log_name(name: str | None = None) -> str:
    """Resolve the shared log file stem for this process tree.

    The first process to initialize logging chooses a timestamped name and
    stores it in an env var. Child processes inherit that env var and reuse
    the same file instead of creating their own per-process log files.
    """
    inherited_name = os.environ.get(_LOG_NAME_ENV)
    if inherited_name:
        return inherited_name

    formatted_date = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{formatted_date}" if name else formatted_date
    os.environ[_LOG_NAME_ENV] = log_name
    return log_name


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    global _current_log_name
    log_name = _resolve_log_name(name)
    _current_log_name = log_name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(
        PROJECT_ROOT / f"logs/{log_name}.log",
        level=logfile_level,
        enqueue=True,
    )
    return _logger


def get_current_log_name() -> str:
    """Return current log file stem without extension."""
    return _current_log_name or _resolve_log_name()


logger = define_log_level()


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
