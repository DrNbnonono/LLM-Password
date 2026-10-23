import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple


DATASET_FIELDS = [
    "Password",
    "Provider",
    "Model_ID",
    "Prompt_Type",
    "Prompt_Text",
    "Lang",
    "Temperature",
    "Max_Tokens",
    "Timestamp",
    "Length",
    "Character_Diversity",
    "Entropy",
    "Pattern_Score",
    "Crack_Time",
]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _read_existing_header(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        return [h.strip() for h in first_line.split(",")] if first_line else []


def ensure_dataset_path(csv_path: str, fieldnames: List[str]) -> str:
    """Ensure the CSV exists with given header. If existing header mismatches, create a new v2 file."""
    _ensure_parent_dir(csv_path)
    existing = _read_existing_header(csv_path)
    if not existing:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return csv_path
    if existing == fieldnames:
        return csv_path
    # migrate to a new file path to avoid header conflict
    root, ext = os.path.splitext(csv_path)
    migrated = f"{root}_v2{ext or '.csv'}"
    _ensure_parent_dir(migrated)
    if not os.path.exists(migrated) or os.path.getsize(migrated) == 0:
        with open(migrated, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    return migrated


def append_rows(csv_path: str, rows: List[Dict[str, Any]], fieldnames: List[str] = None) -> None:
    fieldnames = fieldnames or DATASET_FIELDS
    _ensure_parent_dir(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def make_row(password: str, provider: str, model_id: str, prompt_type: str, prompt_text: str,
             lang: str, temperature: float, max_tokens: int, length: int,
             char_diversity: int, entropy: float, pattern_score: float, crack_time: str) -> Dict[str, Any]:
    return {
        "Password": password,
        "Provider": provider,
        "Model_ID": model_id,
        "Prompt_Type": prompt_type,
        "Prompt_Text": prompt_text,
        "Lang": lang,
        "Temperature": float(temperature),
        "Max_Tokens": int(max_tokens),
        "Timestamp": datetime.utcnow().isoformat(),
        "Length": length,
        "Character_Diversity": char_diversity,
        "Entropy": round(entropy, 3),
        "Pattern_Score": round(pattern_score, 3),
        "Crack_Time": crack_time,
    }
