from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NER_LABEL_MAPPING = {
    "PER": "[姓名]",
    "PERSON": "[姓名]",
    "LOC": "[地点]",
    "LOCATION": "[地点]",
    "GPE": "[地点]",
    "ORG": "[机构]",
}


@dataclass(slots=True)
class NerEntity:
    start: int
    end: int
    label: str
    score: float


class OptionalNerMasker:
    def __init__(self, model_name: str = "dslim/bert-base-NER", device: int = -1) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline: Any | None = None
        self._load_error: Exception | None = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._pipeline is not None

    @property
    def load_error(self) -> Exception | None:
        self._ensure_loaded()
        return self._load_error

    def mask(self, text: str) -> str:
        self._ensure_loaded()
        if self._pipeline is None:
            return text

        raw_entities = self._pipeline(text)
        entities = self._normalize_entities(raw_entities)
        return self._replace_entities(text, entities)

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            self._pipeline = None

    def _normalize_entities(self, raw_entities: list[dict[str, Any]]) -> list[NerEntity]:
        entities: list[NerEntity] = []
        for item in raw_entities:
            label = str(item.get("entity_group") or item.get("entity") or "").upper()
            label = label.removeprefix("B-").removeprefix("I-")
            replacement = NER_LABEL_MAPPING.get(label)
            start = item.get("start")
            end = item.get("end")
            score = float(item.get("score") or 0)
            if replacement and isinstance(start, int) and isinstance(end, int) and start < end:
                entities.append(NerEntity(start=start, end=end, label=label, score=score))
        return self._merge_overlaps(entities)

    def _merge_overlaps(self, entities: list[NerEntity]) -> list[NerEntity]:
        if not entities:
            return []
        ordered = sorted(entities, key=lambda item: (item.start, -item.end))
        merged: list[NerEntity] = [ordered[0]]
        for entity in ordered[1:]:
            current = merged[-1]
            if entity.start <= current.end:
                if entity.end > current.end:
                    merged[-1] = NerEntity(
                        start=current.start,
                        end=entity.end,
                        label=current.label,
                        score=max(current.score, entity.score),
                    )
            else:
                merged.append(entity)
        return merged

    def _replace_entities(self, text: str, entities: list[NerEntity]) -> str:
        if not entities:
            return text
        parts: list[str] = []
        cursor = 0
        for entity in entities:
            parts.append(text[cursor:entity.start])
            parts.append(NER_LABEL_MAPPING.get(entity.label, "[实体]"))
            cursor = entity.end
        parts.append(text[cursor:])
        return "".join(parts)
