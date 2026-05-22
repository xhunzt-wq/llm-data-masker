from data_masker.config import MaskingConfig
from data_masker.ner import OptionalNerMasker
from data_masker.rules import apply_regex_rules


class DatasetMasker:
    def __init__(self, config: MaskingConfig | None = None) -> None:
        self.config = config or MaskingConfig()
        self.ner_masker = OptionalNerMasker(self.config.ner_model, self.config.ner_device) if self.config.use_ner else None

    def mask_text(self, text: str) -> str:
        masked = text
        if self.config.keep_image_tokens:
            masked = self._protect_llava_image_token(masked)
        masked = apply_regex_rules(
            masked,
            self.config.enabled_entities,
            strict_name_rules=self.config.strict_name_rules,
        )
        if self.ner_masker is not None:
            masked = self.ner_masker.mask(masked)
        if self.config.keep_image_tokens:
            masked = self._restore_llava_image_token(masked)
        return masked

    def mask_value(self, value: object) -> object:
        if isinstance(value, str):
            return self.mask_text(value)
        if isinstance(value, list):
            return [self.mask_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.mask_value(item) for key, item in value.items()}
        return value

    def _protect_llava_image_token(self, text: str) -> str:
        return text.replace("<image>", "__LLAVA_IMAGE_TOKEN__")

    def _restore_llava_image_token(self, text: str) -> str:
        return text.replace("__LLAVA_IMAGE_TOKEN__", "<image>")
