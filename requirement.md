# Requirement

## Python 版本

- Python >= 3.10

## 必需依赖

当前规则脱敏、JSON/JSONL 读写、LLaVA 数据集处理均只使用 Python 标准库，无必需第三方依赖。

## 可选依赖：NER 模型脱敏

如果需要启用 `--use-ner`，安装以下依赖：

```bash
pip install transformers torch
```

默认模型：

```text
dslim/bert-base-NER
```

说明：

- 没有 GPU 时可以使用 CPU：`--ner-device -1`。
- 有 GPU 时可以指定第一张卡：`--ner-device 0`。
- 未安装 `transformers` 或模型加载失败时，程序会保留规则脱敏结果，并输出 NER 不可用原因。
- `dslim/bert-base-NER` 主要面向英文 NER；中文姓名、手机号、身份证、银行卡、地址等默认优先使用规则匹配。

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

## 基本运行

```bash
python main.py --input data/raw.json --output data/masked.json
```

JSONL：

```bash
python main.py --input data/raw.jsonl --output data/masked.jsonl
```

只脱敏手机号、银行卡号和邮箱：

```bash
python main.py -i data/raw.json -o data/masked.json --entities phone bank_card email
```

启用可选 NER：

```bash
python main.py -i data/raw.json -o data/masked.json --use-ner --ner-device -1
```

递归脱敏所有字符串字段：

```bash
python main.py -i data/raw.json -o data/masked.json --mode all-text
```
