from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_ENTITY_TYPES = (
    "phone",
    "id_card",
    "bank_card",
    "email",
    "address",
    "date",
    "money",
    "name",
    "passport",
    "license_plate",
)


@dataclass(slots=True)
class MaskingConfig:
    enabled_entities: set[str] = field(default_factory=lambda: set(DEFAULT_ENTITY_TYPES))
    use_ner: bool = False
    ner_model: str = "dslim/bert-base-NER"
    ner_device: int = -1
    keep_image_tokens: bool = True
    strict_name_rules: bool = False

    @classmethod
    def from_entities(
        cls,
        entities: Iterable[str] | None,
        *,
        use_ner: bool = False,
        ner_model: str = "dslim/bert-base-NER",
        ner_device: int = -1,
        keep_image_tokens: bool = True,
        strict_name_rules: bool = False,
    ) -> "MaskingConfig":
        enabled_entities = set(DEFAULT_ENTITY_TYPES if entities is None else entities)
        return cls(
            enabled_entities=enabled_entities,
            use_ner=use_ner,
            ner_model=ner_model,
            ner_device=ner_device,
            keep_image_tokens=keep_image_tokens,
            strict_name_rules=strict_name_rules,
        )
