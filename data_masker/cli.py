from __future__ import annotations

import argparse
from pathlib import Path

from data_masker.config import DEFAULT_ENTITY_TYPES, MaskingConfig, resolve_default_ner_model
from data_masker.dataset import load_dataset, mask_dataset, save_dataset
from data_masker.masker import DatasetMasker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask private information in LLM Q&A or LLaVA-style multimodal datasets.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input dataset path, supports .json and .jsonl")
    parser.add_argument("-o", "--output", required=True, help="Output dataset path")
    parser.add_argument(
        "--format",
        choices=("auto", "json", "jsonl"),
        default="auto",
        help="Input and output dataset format. Defaults to auto detection by suffix.",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        default=None,
        choices=DEFAULT_ENTITY_TYPES,
        help="Entity types to mask. Defaults to all supported rule entities.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "all-text"),
        default="auto",
        help="auto masks common dataset text fields; all-text recursively masks every string value.",
    )
    parser.add_argument("--use-ner", action="store_true", help="Enable optional HuggingFace NER masking")
    parser.add_argument(
        "--ner-model",
        default=None,
        help="HuggingFace NER model name or local model path. Defaults to weight/bert-base-NER if available, otherwise dslim/bert-base-NER.",
    )
    parser.add_argument("--ner-device", type=int, default=-1, help="NER device, -1 for CPU, 0 for first GPU")
    parser.add_argument(
        "--no-keep-image-tokens",
        action="store_true",
        help="Do not protect LLaVA <image> tokens during masking",
    )
    parser.add_argument(
        "--strict-name-rules",
        action="store_true",
        help="Only mask names with explicit prefixes such as 姓名/联系人/患者",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    config = MaskingConfig.from_entities(
        args.entities,
        use_ner=args.use_ner,
        ner_model=args.ner_model or resolve_default_ner_model(),
        ner_device=args.ner_device,
        keep_image_tokens=not args.no_keep_image_tokens,
        strict_name_rules=args.strict_name_rules,
    )
    masker = DatasetMasker(config)
    data, resolved_format = load_dataset(input_path, args.format)
    masked = mask_dataset(data, masker, mode=args.mode)
    save_dataset(masked, output_path, resolved_format)

    print(f"Masked dataset written to: {output_path}")
    if masker.ner_masker is not None and not masker.ner_masker.available:
        print(f"NER masking was requested but unavailable: {masker.ner_masker.load_error}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
