import re
from collections.abc import Callable


MaskFunction = Callable[[re.Match[str]], str]


def _mask_keep_edges(value: str, prefix: int, suffix: int, mask_char: str = "*") -> str:
    if len(value) <= prefix + suffix:
        return mask_char * len(value)
    tail = value[-suffix:] if suffix > 0 else ""
    return f"{value[:prefix]}{mask_char * (len(value) - prefix - suffix)}{tail}"


def _mask_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    return _mask_keep_edges(value, 3, 4)


def _mask_id_card(match: re.Match[str]) -> str:
    value = match.group("id") if "id" in match.groupdict() and match.group("id") else match.group(0)
    masked = _mask_keep_edges(value, 3 if re.search(r"[A-Za-z]", value) else 6, 4)
    if "id" in match.groupdict() and match.group("id"):
        return match.group(0).replace(value, masked, 1)
    return masked


def _mask_bank_card(match: re.Match[str]) -> str:
    value = match.group(0)
    compact = re.sub(r"[ -]", "", value)
    masked = _mask_keep_edges(compact, 4, 4)
    return masked


def _mask_email(match: re.Match[str]) -> str:
    value = match.group(0)
    local, domain = value.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}"
    return f"{masked_local}@{domain}"


def _mask_address(match: re.Match[str]) -> str:
    value = match.group(0)
    return f"[地址:{len(value)}字]"


def _mask_name(match: re.Match[str]) -> str:
    value = match.group("name") if "name" in match.groupdict() else match.group(0)
    if re.fullmatch(r"[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*)*", value):
        masked = " ".join(_mask_keep_edges(part, 1, 0) for part in value.split())
    elif len(value) <= 1:
        masked = "*"
    else:
        masked = f"{value[0]}{'*' * (len(value) - 1)}"
    if "name" in match.groupdict():
        return match.group(0).replace(value, masked, 1)
    return masked


def _constant(token: str) -> MaskFunction:
    def replacer(match: re.Match[str]) -> str:
        return token

    return replacer


RULES: dict[str, tuple[re.Pattern[str], str | MaskFunction]] = {
    "phone": (
        re.compile(r"(?<![\d+])(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)|(?<![\d+])(?:\+?1[-. ]?)?(?:\(?\d{3}\)?[-. ]?)\d{3}[-. ]?\d{4}(?!\d)|(?<![\d+])\+\d{1,3}[-. ]?(?:\(?\d{1,4}\)?[-. ]?){2,5}\d{3,4}(?!\d)"),
        _mask_phone,
    ),
    "id_card": (
        re.compile(r"(?<!\d)\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)|\b(?:SSN|Social\s+Security\s+Number|National\s+ID|ID\s*(?:No\.?|Number|Card)?|Driver'?s?\s+License|DL)[:#：\s]*(?P<id>[A-Z0-9][A-Z0-9-]{4,24}[A-Z0-9])\b", re.IGNORECASE),
        _mask_id_card,
    ),
    "bank_card": (
        re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        _mask_bank_card,
    ),
    "email": (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        _mask_email,
    ),
    "passport": (
        re.compile(r"\b(?:Passport|Passport\s*(?:No\.?|Number))[:#：\s]*(?:[A-Z]{1,2}\d{6,9}|[A-Z0-9]{6,12})(?![A-Z0-9])|(?<![A-Z0-9])(?:[EGPS]\d{8}|[A-Z]{1,2}\d{6,9})(?![A-Z0-9])", re.IGNORECASE),
        _constant("[护照号]"),
    ),
    "license_plate": (
        re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{4,5}[A-Z0-9挂学警港澳]|\b(?:License\s*Plate|Plate\s*(?:No\.?|Number)?|Vehicle\s*Plate)[:#：\s]*(?:[A-Z0-9]{1,4}[- ]?){2,4}\b", re.IGNORECASE),
        _constant("[车牌号]"),
    ),
    "date": (
        re.compile(r"(?<!\d)(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}[-/.月]\d{1,2}日?)(?!\d)|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\b", re.IGNORECASE),
        _constant("[日期]"),
    ),
    "money": (
        re.compile(r"(?:￥|¥|人民币)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:元|万元|块|美元|美金|USD|RMB|CNY|dollars?|bucks?|yuan)|(?:\$|US\$|USD\s*)\s*\d+(?:,\d{3})*(?:\.\d+)?", re.IGNORECASE),
        _constant("[金额]"),
    ),
    "address": (
        re.compile(r"(?:[\u4e00-\u9fa5]{2,}(?:省|市|自治区|自治州|县|区|镇|乡|村|街道)(?:[\u4e00-\u9fa5A-Za-z0-9-]{0,24})|[\u4e00-\u9fa5]{2,}(?:路|大道|街|巷)\d+(?:号楼?|单元|室)?|\b\d{1,6}\s+[A-Za-z0-9.'’-]+(?:\s+[A-Za-z0-9.'’-]+){0,6}\s+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Square|Sq\.?)\b(?:,?\s*(?:Apt|Apartment|Unit|Suite|Ste\.?|#)\s*[A-Za-z0-9-]+)?|\b(?:Address|Location|City|State|Province|County|District)[:：\s]+[A-Z][A-Za-z.'’-]*(?:\s+[A-Z][A-Za-z.'’-]*){0,5}(?:,\s*[A-Z]{2})?(?:\s+\d{5}(?:-\d{4})?)?)", re.IGNORECASE),
        _mask_address,
    ),
    "name": (
        re.compile(r"(?P<prefix>姓名|名字|联系人|收件人|患者|客户|用户|开户人|持卡人|name|full\s+name|contact|recipient|patient|customer|user|cardholder)[:：\s]*(?P<name>[\u4e00-\u9fa5]{2,4}|[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3})", re.IGNORECASE),
        _mask_name,
    ),
}


LOOSE_CHINESE_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fa5])(?:[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣邓郁单欧闫])[\u4e00-\u9fa5]{1,2}(?=$|[\s,，.。;；:：!?！？、）)]|[的是在和])")
CONTEXT_CHINESE_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fa5])(?P<name>[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣邓郁单欧闫][\u4e00-\u9fa5]{1,2}?)(?=的?(?:邮箱|电子邮箱|电话|手机号|手机|身份证|银行卡|卡号|住址|地址|护照|车牌))")
CONTEXT_ENGLISH_NAME_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){1,3})(?=\s+(?:email|e-mail|phone|mobile|address|passport|card|bank\s+card|license\s+plate|account)\b)", re.IGNORECASE)


def apply_regex_rules(text: str, enabled_entities: set[str], *, strict_name_rules: bool = False) -> str:
    masked = text
    for entity_type, (pattern, replacement) in RULES.items():
        if entity_type not in enabled_entities:
            continue
        masked = pattern.sub(replacement, masked)

    if "name" in enabled_entities and not strict_name_rules:
        masked = CONTEXT_CHINESE_NAME_PATTERN.sub(_mask_name, masked)
        masked = CONTEXT_ENGLISH_NAME_PATTERN.sub(_mask_name, masked)
        masked = LOOSE_CHINESE_NAME_PATTERN.sub(lambda item: _mask_keep_edges(item.group(0), 1, 0), masked)

    return masked
