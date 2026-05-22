# LLM Data Masker

面向大模型问答数据集和 LLaVA 多模态文档数据集的隐私脱敏工具。

## 环境要求

- Python >= 3.10

默认规则脱敏、JSON/JSONL 读写、LLaVA 数据集处理只使用 Python 标准库。`requirement.md` 中的依赖主要用于可选 NER 模型脱敏。

## 安装环境


安装可选模型依赖：

```bash
pip install -r requirement.md
```

如果只使用规则脱敏，可以不安装第三方依赖，直接运行 `main.py`。

## NER 模型权重

启用 `--use-ner` 时，程序会优先查找本地模型目录 `weight/bert-base-NER`。如果该目录不存在或模型文件不完整，则使用 HuggingFace 模型名 `dslim/bert-base-NER`，由 `transformers` 在首次运行时自动下载并缓存。

如果自动下载失败，可以手动下载 `dslim/bert-base-NER` 的完整模型文件，并放到：

```text
weight/bert-base-NER/
```

目录中至少需要包含 `config.json`，以及 `pytorch_model.bin` 或 `model.safetensors` 等模型权重文件。重新运行后，默认会直接使用本地目录。

## 功能

- 规则化字符匹配脱敏：手机号、身份证、银行卡、邮箱、地址、日期、金额、姓名候选等。
- 可选 NER 模型脱敏：支持本地 `weight/bert-base-NER` 或远程 `dslim/bert-base-NER`。
- 支持 JSON 与 JSONL。
- 支持普通问答字段和 LLaVA `conversations[].value`。
- 默认保护 LLaVA `<image>` 标记，避免被误处理。

## 支持的脱敏实体

- `phone`：手机号
- `id_card`：身份证号
- `bank_card`：银行卡号
- `email`：邮箱
- `address`：中文地址片段
- `date`：日期
- `money`：金额
- `name`：中文姓名候选
- `passport`：护照号
- `license_plate`：车牌号

## 输入数据格式

支持：

- `.json`
- `.jsonl`

常见问答字段会自动脱敏：

- `text`
- `content`
- `prompt`
- `completion`
- `question`
- `answer`
- `query`
- `response`
- `instruction`
- `input`
- `output`
- `value`

LLaVA 格式会处理：

```json
{
  "image": "xxx.jpg",
  "conversations": [
    {"from": "human", "value": "<image> 姓名张三，电话13812345678"},
    {"from": "gpt", "value": "用户电话是13812345678"}
  ]
}
```

## 快速开始

```bash
python main.py --input data/raw.json --output data/masked.json
```

启动可视化界面：

```bash
python main.py --web
```

界面支持拖拽 JSON/JSONL 文件、文件夹或压缩包，支持勾选需要脱敏的实体类型，并会在处理完成后展示 1 到 3 条脱敏前后对比。

启用可选 NER：

```bash
python main.py --input data/raw.json --output data/masked.json --use-ner --ner-device -1
```

如果 `weight/bert-base-NER` 中已有完整模型文件，上面的命令会优先使用本地模型；否则会尝试自动下载 `dslim/bert-base-NER`。

处理 JSONL：

```bash
python main.py --input data/raw.jsonl --output data/masked.jsonl
```

只脱敏手机号、银行卡号和邮箱：

```bash
python main.py -i data/raw.json -o data/masked.json --entities phone bank_card email
```

递归脱敏所有字符串字段：

```bash
python main.py -i data/raw.json -o data/masked.json --mode all-text
```

## CLI 参数

```text
-i, --input              输入数据集路径
-o, --output             输出数据集路径
--format                auto/json/jsonl，默认 auto
--entities              指定要脱敏的实体类型
--mode                  auto/all-text
--use-ner               启用可选 NER 脱敏
--ner-model             HuggingFace NER 模型名或本地模型路径，默认优先使用 weight/bert-base-NER，否则使用 dslim/bert-base-NER
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
