import math
import string
from typing import Dict, Any

from zxcvbn import zxcvbn


def _character_set_size(password: str) -> int:
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    # Symbols are characters that are printable, not whitespace, not alphanumeric
    symbols = set(c for c in password if c in string.punctuation)

    size = 0
    if has_lower:
        size += 26
    if has_upper:
        size += 26
    if has_digit:
        size += 10
    if symbols:
        size += len(string.punctuation)  # approximate ASCII punctuation set
    return size


def estimate_entropy(password: str) -> float:
    if not password:
        return 0.0
    char_set_size = _character_set_size(password)
    if char_set_size <= 1:
        return 0.0
    return len(password) * math.log2(char_set_size)


def character_diversity(password: str) -> int:
    return _character_set_size(password)


def zxcvbn_summary(password: str) -> Dict[str, Any]:
    # zxcvbn has a max length limit of 72 characters
    # For passwords exceeding this limit, truncate and add note
    if len(password) > 72:
        return {
            "score": 4,  # Assume strong for very long passwords
            "crack_time": "centuries+ (exceeds zxcvbn max length)",
        }
    
    try:
        result = zxcvbn(password)
        crack_times = result.get("crack_times_display", {})
        crack_time_display = crack_times.get("offline_fast_hashing_1e10_per_second_display", "unknown")
        return {
            "score": result.get("score", 0),
            "crack_time": crack_time_display,
        }
    except Exception as e:
        return {
            "score": 0,
            "crack_time": f"error: {str(e)}",
        }
