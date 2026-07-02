"""Shared utility helpers."""


def format_exception(error: BaseException) -> str:
    """Return a useful exception string even when str(error) is empty."""
    name = type(error).__name__
    message = str(error).strip()
    return f"{name}: {message}" if message else name


def truncate_string(value: str, max_chars: int) -> str:
    """Return value truncated to max_chars with a short marker."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n...[truncated {omitted} chars]"
