import logging
import sys


def setup_logger(
    name: str = __name__, level: int = logging.INFO, log_format: str = None
) -> logging.Logger:
    """Create a professional logger instance."""

    if log_format is None:
        log_format = (
            "%(asctime)s │ %(levelname)-8s │ %(name)s:%(lineno)d\t │ %(message)s"
        )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers, but keep levels synchronized.
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            if log_format is not None:
                formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
                handler.setFormatter(formatter)
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
