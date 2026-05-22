from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from data_masker.masker import DatasetMasker

DatasetFormat = Literal["auto", "json", "jsonl"]


TEXT_FIELD_NAMES = {
    "text",
    "content",
    "prompt",
    "completion",
    "question",
    "answer",
    "query",
    "response",
    "instruction",
    "input",
    "output",
    "value",
}


def detect_format(path: Path, requested_format: DatasetFormat = "auto") -> Literal["json", "jsonl"]:
    if requested_format != "auto":
        return requested_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "json"


def load_dataset(path: Path, dataset_format: DatasetFormat = "auto") -> tuple[Any, Literal["json", "jsonl"]]:
    resolved_format = detect_format(path, dataset_format)
    if resolved_format == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        return rows, resolved_format

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj), resolved_format


def save_dataset(data: Any, path: Path, dataset_format: Literal["json", "jsonl"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_format == "jsonl":
        if not isinstance(data, list):
            raise ValueError("JSONL output requires the top-level dataset to be a list")
        with path.open("w", encoding="utf-8") as file_obj:
            for item in data:
                file_obj.write(json.dumps(item, ensure_ascii=False))
                file_obj.write("\n")
        return

    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def mask_dataset(data: Any, masker: DatasetMasker, *, mode: str = "auto") -> Any:
    if mode == "all-text":
        return masker.mask_value(data)
    if isinstance(data, list):
        return [mask_record(item, masker) for item in data]
    if isinstance(data, dict):
        return mask_record(data, masker)
    return masker.mask_value(data)


def mask_record(record: Any, masker: DatasetMasker) -> Any:
    if isinstance(record, list):
        return [mask_record(item, masker) for item in record]
    if not isinstance(record, dict):
        return masker.mask_value(record)

    masked = dict(record)
    if _looks_like_llava_record(masked):
        masked["conversations"] = _mask_llava_conversations(masked["conversations"], masker)
        return masked

    for key, value in record.items():
        if isinstance(value, str) and key in TEXT_FIELD_NAMES:
            masked[key] = masker.mask_text(value)
        elif isinstance(value, list) and key in {"messages", "conversation", "conversations", "dialog", "dialogs"}:
            masked[key] = [mask_record(item, masker) for item in value]
        elif isinstance(value, dict):
            masked[key] = mask_record(value, masker)
    return masked


def _looks_like_llava_record(record: dict[str, Any]) -> bool:
    conversations = record.get("conversations")
    return isinstance(conversations, list) and all(isinstance(item, dict) for item in conversations)


def _mask_llava_conversations(conversations: Iterable[dict[str, Any]], masker: DatasetMasker) -> list[dict[str, Any]]:
    masked_conversations = []
    for turn in conversations:
        masked_turn = dict(turn)
        value = masked_turn.get("value")
        if isinstance(value, str):
            masked_turn["value"] = masker.mask_text(value)
        masked_conversations.append(masked_turn)
    return masked_conversations
