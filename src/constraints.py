"""Grammar checks used to constrain generation token by token.

Every scanner receives the bytes generated so far for one JSON value
(possibly followed by the characters that close it) and tells whether
they are an invalid sequence, a valid but unfinished prefix, or a
finished value.
"""
import re
from enum import Enum
from typing import Callable


class Status(Enum):
    """Result of scanning a partially generated value."""

    INVALID = 0
    PARTIAL = 1
    DONE = 2


Scanner = Callable[[bytes], tuple[Status, bytes]]

# Bytes allowed after a value once it is finished, e.g. the `",` in `"x",`.
CLOSING_BYTES = frozenset(b",} \t\n")
NUMBER_STOPS = frozenset(b",}")
MAX_NUMBER_LENGTH = 32
MAX_STRING_LENGTH = 512

NUMBER_PREFIX = re.compile(
    rb"-?(?:(?:0|[1-9]\d*)(?:\.\d*)?(?:(?<=\d)[eE][+-]?\d*)?)?"
)
NUMBER_FULL = re.compile(rb"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
INTEGER_PREFIX = re.compile(rb"-?(?:0|[1-9]\d*)?")
INTEGER_FULL = re.compile(rb"-?(?:0|[1-9]\d*)")
HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
SIMPLE_ESCAPES = frozenset(b'"\\/bfnrt')


def _is_closing(tail: bytes) -> bool:
    """Tell whether ``tail`` only contains characters closing a value."""
    return all(byte in CLOSING_BYTES for byte in tail)


def _scan_literal(
    buffer: bytes, prefix: re.Pattern[bytes], full: re.Pattern[bytes]
) -> tuple[Status, bytes]:
    """Scan an unquoted literal ended by ``,`` or ``}``."""
    stop = next(
        (i for i, byte in enumerate(buffer) if byte in NUMBER_STOPS), None
    )
    if stop is None:
        if len(buffer) > MAX_NUMBER_LENGTH or not prefix.fullmatch(buffer):
            return Status.INVALID, buffer
        return Status.PARTIAL, buffer
    value = buffer[:stop]
    if full.fullmatch(value) and _is_closing(buffer[stop:]):
        return Status.DONE, value
    return Status.INVALID, value


def scan_number(buffer: bytes) -> tuple[Status, bytes]:
    """Scan a JSON number (integer or float)."""
    return _scan_literal(buffer, NUMBER_PREFIX, NUMBER_FULL)


def scan_integer(buffer: bytes) -> tuple[Status, bytes]:
    """Scan a JSON integer."""
    return _scan_literal(buffer, INTEGER_PREFIX, INTEGER_FULL)


def make_choice_scanner(choices: list[bytes], quoted: bool) -> Scanner:
    """Build a scanner accepting exactly one of ``choices``.

    Args:
        choices: The accepted values.
        quoted: True if the value is a JSON string whose opening quote was
            already emitted, so that a ``"`` ends it. Otherwise the value is
            a bare literal ended by ``,`` or ``}``.

    Returns:
        The scanner function.
    """
    stops = frozenset(b'"') if quoted else NUMBER_STOPS

    def scan(buffer: bytes) -> tuple[Status, bytes]:
        stop = next(
            (i for i, byte in enumerate(buffer) if byte in stops), None
        )
        if stop is None:
            if any(choice.startswith(buffer) for choice in choices):
                return Status.PARTIAL, buffer
            return Status.INVALID, buffer
        value = buffer[:stop]
        tail = buffer[stop + 1:] if quoted else buffer[stop:]
        if value in choices and _is_closing(tail):
            return Status.DONE, value
        return Status.INVALID, value

    return scan


def scan_boolean(buffer: bytes) -> tuple[Status, bytes]:
    """Scan a JSON boolean."""
    return _BOOLEAN_SCANNER(buffer)


_BOOLEAN_SCANNER = make_choice_scanner([b"true", b"false"], quoted=False)


def scan_string(buffer: bytes) -> tuple[Status, bytes]:
    """Scan the body of a JSON string whose opening quote was emitted.

    The returned value is the raw (still escaped) string body.
    """
    index = 0
    while index < len(buffer):
        byte = buffer[index]
        if byte == ord('"'):
            if _is_closing(buffer[index + 1:]):
                return Status.DONE, buffer[:index]
            return Status.INVALID, buffer[:index]
        if byte < 0x20:
            return Status.INVALID, buffer
        if byte == ord("\\"):
            index += 1
            if index == len(buffer):
                break
            escape = buffer[index]
            if escape == ord("u"):
                digits = buffer[index + 1:index + 5]
                if not all(d in HEX_DIGITS for d in digits):
                    return Status.INVALID, buffer
                index += len(digits)
            elif escape not in SIMPLE_ESCAPES:
                return Status.INVALID, buffer
        index += 1
    if len(buffer) > MAX_STRING_LENGTH:
        return Status.INVALID, buffer
    return Status.PARTIAL, buffer


SCANNERS: dict[str, Scanner] = {
    "number": scan_number,
    "integer": scan_integer,
    "boolean": scan_boolean,
    "string": scan_string,
}
