import json
import logging
import os
import typing

_log_format = (
    '{"time":"%(asctime)s", "name": "%(name)s",'
    ' "file": "%(filename)s", "func": "%(funcName)s", "line": %('
    'lineno)d, "level": "%(levelname)s", "message": %(message)s}'
)
_datefmt = "%d-%b-%y %H:%M:%S"
logger_level: str = os.getenv("LOGGING_LEVEL", "INFO")
    
def _get_stream_handler() -> logging.StreamHandler:  # type: ignore
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger_level)
    stream_handler.setFormatter(logging.Formatter(_log_format, _datefmt))
    return stream_handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)
    logger.addHandler(_get_stream_handler())
    return logger


def json_msg(msg: typing.Any) -> str:
    return json.dumps(msg)
