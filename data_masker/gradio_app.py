from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr

from data_masker.config import DEFAULT_ENTITY_TYPES, MaskingConfig, resolve_default_ner_model
from data_masker.dataset import load_dataset, mask_record, save_dataset
from data_masker.masker import DatasetMasker


ALL_ENTITY_LABEL = "全选"
ENTITY_LABEL_TO_TYPE = {
    "姓名": "name",
    "地点/地址": "address",
    "手机号": "phone",
    "身份证号": "id_card",
    "银行卡号": "bank_card",
    "邮箱": "email",
    "日期": "date",
    "金额": "money",
    "护照号": "passport",
    "车牌号": "license_plate",
}
ENTITY_CHOICES = [ALL_ENTITY_LABEL, *ENTITY_LABEL_TO_TYPE.keys()]
MODE_LABEL_TO_VALUE = {
    "自动识别常见字段": "auto",
    "递归脱敏所有字符串": "all-text",
}
DATASET_SUFFIXES = {".json", ".jsonl"}
CANCEL_EVENT = threading.Event()


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths: list[Path] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            paths.append(Path(item))
        elif isinstance(item, dict):
            raw_path = item.get("path") or item.get("name")
            if raw_path:
                paths.append(Path(raw_path))
        elif hasattr(item, "name"):
            paths.append(Path(item.name))
    return paths


def _is_dataset_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DATASET_SUFFIXES


def _is_archive_file(path: Path) -> bool:
    return path.is_file() and (zipfile.is_zipfile(path) or tarfile.is_tarfile(path))


def _assert_safe_path(base_dir: Path, target_path: Path) -> None:
    base_dir = base_dir.resolve()
    target_path = target_path.resolve()
    try:
        target_path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"压缩包包含不安全路径: {target_path}") from exc


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target_path = target_dir / member.filename
                _assert_safe_path(target_dir, target_path)
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target_path = target_dir / member.name
            _assert_safe_path(target_dir, target_path)
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _collect_dataset_files(paths: list[Path], temp_dir: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for path in paths:
        if _is_dataset_file(path):
            files.append((path, Path(path.name)))
            continue
        if path.is_dir():
            for dataset_file in sorted(path.rglob("*")):
                if _is_dataset_file(dataset_file):
                    files.append((dataset_file, dataset_file.relative_to(path)))
            continue
        if _is_archive_file(path):
            extract_dir = temp_dir / f"extract_{path.stem}"
            _extract_archive(path, extract_dir)
            for dataset_file in sorted(extract_dir.rglob("*")):
                if _is_dataset_file(dataset_file):
                    files.append((dataset_file, dataset_file.relative_to(extract_dir)))
    return files


def _selected_entities(selected_labels: list[str] | None) -> list[str]:
    labels = selected_labels or []
    if ALL_ENTITY_LABEL in labels:
        return list(DEFAULT_ENTITY_TYPES)
    entities = [ENTITY_LABEL_TO_TYPE[label] for label in labels if label in ENTITY_LABEL_TO_TYPE]
    return entities


def _resolve_output_path(output_path: str | None) -> Path:
    if output_path and output_path.strip():
        return Path(output_path.strip()).expanduser()
    return Path.cwd() / "masked_output"


def _masked_relative_path(relative_path: Path) -> Path:
    return relative_path.with_name(f"{relative_path.stem}_masked{relative_path.suffix}")


def _preview_items(original: Any, masked: Any) -> list[tuple[Any, Any]]:
    if isinstance(original, list) and isinstance(masked, list):
        return list(zip(original[:3], masked[:3]))
    return [(original, masked)]


def _compact_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > 1800:
        return f"{text[:1800]}..."
    return text


def _zip_outputs(output_files: list[Path], output_root: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_file in output_files:
            try:
                archive_name = output_file.relative_to(output_root)
            except ValueError:
                archive_name = output_file.name
            archive.write(output_file, archive_name.as_posix())
    return zip_path


def _prepare_download_file(output_files: list[Path], output_root: Path) -> Path:
    download_dir = Path(tempfile.mkdtemp(prefix="llm_data_masker_download_"))
    if len(output_files) == 1:
        target_path = download_dir / output_files[0].name
        shutil.copy2(output_files[0], target_path)
        return target_path
    return _zip_outputs(output_files, output_root, download_dir / "masked_results.zip")


def _mask_dataset_with_cancel(data: Any, masker: DatasetMasker, *, mode: str) -> Any:
    if mode == "all-text":
        if isinstance(data, list):
            masked_items = []
            for item in data:
                if CANCEL_EVENT.is_set():
                    return masked_items
                masked_items.append(masker.mask_value(item))
            return masked_items
        return masker.mask_value(data)
    if isinstance(data, list):
        masked_records = []
        for item in data:
            if CANCEL_EVENT.is_set():
                return masked_records
            masked_records.append(mask_record(item, masker))
        return masked_records
    if isinstance(data, dict):
        return mask_record(data, masker)
    return masker.mask_value(data)


def process_datasets(
    dataset_input_value: Any,
    output_path: str | None,
    entity_labels: list[str] | None,
    mode_label: str,
    use_ner: bool,
    ner_model: str | None,
    ner_device: int | float,
    strict_name_rules: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[list[str]], str | None]:
    CANCEL_EVENT.clear()
    selected_entities = _selected_entities(entity_labels)
    if not selected_entities:
        raise gr.Error("请至少选择一种需要脱敏的数据类型。")

    input_paths = _as_path_list(dataset_input_value)
    if not input_paths:
        raise gr.Error("请先拖拽或选择输入数据集文件、文件夹或压缩包。")

    mode = MODE_LABEL_TO_VALUE.get(mode_label, "auto")
    output_base = _resolve_output_path(output_path)
    preview_rows: list[list[str]] = []
    output_files: list[Path] = []

    config = MaskingConfig.from_entities(
        selected_entities,
        use_ner=use_ner,
        ner_model=ner_model or resolve_default_ner_model(),
        ner_device=int(ner_device),
        strict_name_rules=strict_name_rules,
    )
    masker = DatasetMasker(config)

    with tempfile.TemporaryDirectory(prefix="llm_data_masker_") as temp_name:
        temp_dir = Path(temp_name)
        datasets = _collect_dataset_files(input_paths, temp_dir)
        if not datasets:
            raise gr.Error("没有找到可处理的 .json 或 .jsonl 数据集文件。")

        exact_output_file = len(datasets) == 1 and output_base.suffix.lower() in DATASET_SUFFIXES
        output_root = output_base.parent if exact_output_file else output_base
        output_root.mkdir(parents=True, exist_ok=True)

        total = len(datasets)
        for index, (dataset_path, relative_path) in enumerate(datasets, start=1):
            if CANCEL_EVENT.is_set():
                return "已终止脱敏任务。", preview_rows, None
            progress((index - 1) / total, desc=f"正在处理 {relative_path.as_posix()}")
            original, resolved_format = load_dataset(dataset_path, "auto")
            if CANCEL_EVENT.is_set():
                return "已终止脱敏任务。", preview_rows, None
            masked = _mask_dataset_with_cancel(original, masker, mode=mode)
            if CANCEL_EVENT.is_set():
                return "已终止脱敏任务。", preview_rows, None
            if exact_output_file:
                target_path = output_base
            else:
                target_path = output_root / _masked_relative_path(relative_path)
            save_dataset(masked, target_path, resolved_format)
            output_files.append(target_path)

            if len(preview_rows) < 3:
                for before, after in _preview_items(original, masked):
                    preview_rows.append([
                        relative_path.as_posix(),
                        str(len(preview_rows) + 1),
                        _compact_json(before),
                        _compact_json(after),
                    ])
                    if len(preview_rows) >= 3:
                        break
            progress(index / total, desc=f"已完成 {index}/{total}")

    download_path = str(_prepare_download_file(output_files, output_root))
    status_lines = [
        f"完成脱敏：共处理 {len(output_files)} 个数据集文件。",
        f"输出位置：`{output_root}`",
    ]
    if masker.ner_masker is not None and not masker.ner_masker.available:
        status_lines.append(f"NER 脱敏未启用成功：`{masker.ner_masker.load_error}`")
    return "\n\n".join(status_lines), preview_rows, download_path


def show_running_button() -> tuple[Any, Any, str]:
    CANCEL_EVENT.clear()
    return gr.update(visible=False), gr.update(visible=True), "正在准备脱敏任务..."


def show_start_button() -> tuple[Any, Any]:
    CANCEL_EVENT.clear()
    return gr.update(visible=True), gr.update(visible=False)


def cancel_processing() -> tuple[Any, Any, str]:
    CANCEL_EVENT.set()
    return gr.update(visible=True), gr.update(visible=False), "正在终止脱敏任务..."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="LLM 数据集脱敏工具") as demo:
        gr.Markdown("# LLM 数据集脱敏工具\n拖拽文件夹、JSON/JSONL 文件或压缩包，选择脱敏类型后开始处理。")
        dataset_input = gr.File(
            label="输入数据集文件、文件夹或压缩包",
            file_count="multiple",
            type="filepath",
            file_types=[".json", ".jsonl", ".zip", ".tar", ".gz", ".tgz"],
        )
        output_path = gr.Textbox(
            label="输出路径",
            placeholder="例如：/root/llm-data-masker/masked_output 或 /root/llm-data-masker/masked.json",
        )
        entity_labels = gr.CheckboxGroup(
            label="选择需要脱敏的数据类型",
            choices=ENTITY_CHOICES,
            value=ENTITY_CHOICES,
        )
        with gr.Row():
            mode_label = gr.Dropdown(
                label="处理模式",
                choices=list(MODE_LABEL_TO_VALUE.keys()),
                value="自动识别常见字段",
            )
            strict_name_rules = gr.Checkbox(label="姓名仅使用严格规则", value=False)
        with gr.Accordion("可选 NER 设置", open=False):
            use_ner = gr.Checkbox(label="启用 NER 辅助脱敏", value=False)
            ner_model = gr.Textbox(label="NER 模型", value=resolve_default_ner_model())
            ner_device = gr.Number(label="NER 设备（-1 为 CPU，0 为第一张 GPU）", value=-1, precision=0)
        start_button = gr.Button("开始脱敏", variant="primary")
        stop_button = gr.Button("终止", variant="stop", visible=False)
        status = gr.Markdown(label="处理状态")
        preview = gr.Dataframe(
            label="脱敏前后对比（最多 3 条）",
            headers=["文件", "序号", "脱敏前", "脱敏后"],
            datatype=["str", "str", "str", "str"],
            wrap=True,
        )
        download = gr.File(label="下载处理结果")

        start_event = start_button.click(
            fn=show_running_button,
            outputs=[start_button, stop_button, status],
            queue=False,
        )
        run_event = start_event.then(
            fn=process_datasets,
            inputs=[
                dataset_input,
                output_path,
                entity_labels,
                mode_label,
                use_ner,
                ner_model,
                ner_device,
                strict_name_rules,
            ],
            outputs=[status, preview, download],
        )
        run_event.then(
            fn=show_start_button,
            outputs=[start_button, stop_button],
            queue=False,
        )
        stop_button.click(
            fn=cancel_processing,
            outputs=[start_button, stop_button, status],
            cancels=[run_event],
            queue=False,
        )
    return demo


def launch_app(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False, inbrowser: bool = False) -> None:
    build_interface().queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio UI for LLM dataset masking.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()
    launch_app(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
