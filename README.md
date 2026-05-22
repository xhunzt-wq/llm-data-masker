# LLM Data Masker

面向大模型问答数据集和 LLaVA 多模态文档数据集的隐私脱敏工具。

## 功能

- 规则化字符匹配脱敏：手机号、身份证、银行卡、邮箱、地址、日期、金额、姓名候选等。
- 可选 NER 模型脱敏：支持 `dslim/bert-base-NER`。
- 支持 JSON 与 JSONL。
- 支持普通问答字段和 LLaVA `conversations[].value`。
- 默认保护 LLaVA `<image>` 标记，避免被误处理。

## 快速开始

```bash
python main.py --input data/raw.json --output data/masked.json
```

启用可选 NER：

```bash
pip install transformers torch
python main.py --input data/raw.json --output data/masked.json --use-ner --ner-device -1
```

## CLI 参数

```text
-i, --input              输入数据集路径
-o, --output             输出数据集路径
--format                auto/json/jsonl，默认 auto
--entities              指定要脱敏的实体类型
--mode                  auto/all-text
--use-ner               启用可选 NER 脱敏
--ner-model             HuggingFace NER 模型名，默认 dslim/bert-base-NER
--ner-device            -1 使用 CPU，0 使用第一张 GPU
--strict-name-rules     只脱敏带姓名上下文前缀的中文姓名
```

## 示例

输入：

```json
{
  "question": "张三的手机号是13812345678，银行卡号6222021234567890123，住在北京市朝阳区幸福路8号。",
  "answer": "请不要泄露张三的个人信息。"
}
```

输出：

```json
{
  "question": "张*的手机号是138****5678，银行卡号6222***********0123，住在[地址:12字]。",
  "answer": "请不要泄露张*的个人信息。"
}
```

## 注意事项

- 规则脱敏适合结构化隐私信息，速度快、无外部依赖。
- 中文姓名规则存在误报风险，可使用 `--strict-name-rules` 降低误报。
- `dslim/bert-base-NER` 主要适合英文命名实体；中文场景建议优先使用规则或替换为中文 NER 模型。
- 生产环境建议先抽样评估脱敏召回率和误报率，再批量处理训练数据。
