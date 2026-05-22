from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NER_REMOTE_MODEL = "dslim/bert-base-NER"
DEFAULT_NER_LOCAL_MODEL = PROJECT_ROOT / "weight" / "bert-base-NER"
LOCAL_NER_MODEL_FILES = ("pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack")


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


def resolve_default_ner_model() -> str:
    if DEFAULT_NER_LOCAL_MODEL.is_dir() and (DEFAULT_NER_LOCAL_MODEL / "config.json").is_file():
        if any((DEFAULT_NER_LOCAL_MODEL / file_name).is_file() for file_name in LOCAL_NER_MODEL_FILES):
            return str(DEFAULT_NER_LOCAL_MODEL)
    return DEFAULT_NER_REMOTE_MODEL


@dataclass(slots=True)
class MaskingConfig:
    enabled_entities: set[str] = field(default_factory=lambda: set(DEFAULT_ENTITY_TYPES))
    use_ner: bool = False
    ner_model: str = field(default_factory=resolve_default_ner_model)
    ner_device: int = -1
    keep_image_tokens: bool = True
    strict_name_rules: bool = False

    @classmethod
    def from_entities(
        cls,
        entities: Iterable[str] | None,
        *,
        use_ner: bool = False,
        ner_model: str | None = None,
        ner_device: int = -1,
        keep_image_tokens: bool = True,
        strict_name_rules: bool = False,
    ) -> "MaskingConfig":
        enabled_entities = set(DEFAULT_ENTITY_TYPES if entities is None else entities)
        return cls(
            enabled_entities=enabled_entities,
            use_ner=use_ner,
            ner_model=ner_model or resolve_default_ner_model(),
            ner_device=ner_device,
            keep_image_tokens=keep_image_tokens,
            strict_name_rules=strict_name_rules,
        )
