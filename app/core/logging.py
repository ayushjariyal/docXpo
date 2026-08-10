"""Structured logging setup.

Goal: one log pipeline for the whole process. Our own ``structlog`` calls and
third-party libraries that use plain ``logging`` (uvicorn, sqlalchemy, alembic)
must come out of the same handler, in the same format, on stdout.

The mechanism that makes that work is ``ProcessorFormatter``:

    our code   -> structlog -> wrap_for_formatter -.
                                                    >-- ProcessorFormatter -> stdout
    uvicorn    -> logging   -> foreign_pre_chain --'

structlog events are handed to stdlib logging as an unrendered dict, and
records that came *from* stdlib get pushed through ``foreign_pre_chain`` so they
gain the same keys (timestamp, level, logger). Both then hit one renderer.
"""

from __future__ import annotations

import logging
import sys

import structlog

# Loggers whose own handlers we tear down so their records propagate to the root
# handler we install. Otherwise uvicorn prints its own differently-formatted
# lines and you get two log formats interleaved.
_HIJACKED_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "sqlalchemy.engine",
    "alembic",
)


def configure_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog + stdlib logging. Safe to call more than once."""

    # Processors shared by both paths, so a uvicorn line and one of our lines
    # carry the same fields.
    shared_processors: list[structlog.typing.Processor] = [
        # Pulls in anything bound via bind_contextvars() -- this is how the
        # request_id set by the middleware lands on every log line emitted
        # during that request, without threading it through function args.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Must be last: instead of rendering, it packages the event dict
            # into a stdlib LogRecord for ProcessorFormatter to finish.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Safe because configure_logging() runs before any logger is bound.
        cache_logger_on_first_use=True,
    )

    if json_logs:
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
        # JSONRenderer can't serialise an exc_info tuple, so flatten it to a
        # "exception" string first. ConsoleRenderer does its own (prettier)
        # traceback formatting, which is why this is conditional.
        final_processors = [structlog.processors.format_exc_info, renderer]
    else:
        final_processors = [structlog.dev.ConsoleRenderer(colors=True)]

    formatter = structlog.stdlib.ProcessorFormatter(
        # Applied only to records that did NOT come from structlog.
        foreign_pre_chain=shared_processors,
        processors=[
            # Strips structlog's internal bookkeeping keys (_record, _from_structlog)
            # so they don't show up in the output.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *final_processors,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]  # replace, don't append -- avoids duplicates on re-call
    root.setLevel(level.upper())

    for name in _HIJACKED_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
