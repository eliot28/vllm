# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""
import datetime
import logging
import os
import sys
import threading
from functools import partial
from typing import Optional

VLLM_CONFIGURE_LOGGING = int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("vllm")
_default_handler: Optional[logging.Handler] = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
if VLLM_CONFIGURE_LOGGING:
    _setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

    if VLLM_CONFIGURE_LOGGING:
        if _default_handler is None:
            raise ValueError(
                "_default_handler is not set up. This should never happen!"
                " Please open an issue on Github.")
        logger.addHandler(_default_handler)
        logger.propagate = False
    return logger


logger = init_logger(__name__)


def trace_calls(filename, frame, event, arg):
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        # Log every function call or return
        if event == 'call':
            logging.debug(f"{datetime.datetime.now()} Call to"
                          f" {func_name} in {filename}:{lineno}")
        else:
            logging.debug(f"{datetime.datetime.now()} Return from"
                          f" {func_name} in {filename}:{lineno}")
    return trace_calls


if int(os.getenv("VLLM_TRACE_FRAME", "0")):
    logger.warning(
        "VLLM_TRACE_FRAME is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang in distributed"
        " inference only.")
    temp_dir = os.environ.get('TMPDIR') or os.environ.get(
        'TEMP') or os.environ.get('TMP') or "/tmp/"
    log_path = os.path.join(temp_dir,
                            (f"vllm_trace_frame_for_process_{os.getpid()}"
                             f"_thread_{threading.get_ident()}_"
                             f"at_{datetime.datetime.now()}.log"))
    logger.info(f"Trace frame log is saved to {log_path}")
    sys.settrace(partial(trace_calls, log_path))
