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
    value = match.group(0)
    return _mask_keep_edges(value, 6, 4)


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
    if len(value) <= 1:
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
        re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
        _mask_phone,
    ),
    "id_card": (
        re.compile(r"(?<!\d)\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"),
        _mask_id_card,
    ),
    "bank_card": (
        re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"),
        _mask_bank_card,
    ),
    "email": (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        _mask_email,
    ),
    "passport": (
        re.compile(r"\b(?:[EGPS]\d{8}|[A-Z]{1,2}\d{6,9})\b"),
        _constant("[护照号]"),
    ),
    "license_plate": (
        re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{4,5}[A-Z0-9挂学警港澳]"),
        _constant("[车牌号]"),
    ),
    "date": (
        re.compile(r"(?<!\d)(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}[-/.月]\d{1,2}日?)(?!\d)"),
        _constant("[日期]"),
    ),
    "money": (
        re.compile(r"(?:￥|¥|人民币)?\s*\d+(?:\.\d+)?\s*(?:元|万元|块|美元|美金|USD|RMB)"),
        _constant("[金额]"),
    ),
    "address": (
        re.compile(r"(?:[\u4e00-\u9fa5]{2,}(?:省|市|自治区|自治州|县|区|镇|乡|村|街道)(?:[\u4e00-\u9fa5A-Za-z0-9-]{0,24})|[\u4e00-\u9fa5]{2,}(?:路|大道|街|巷)\d+(?:号楼?|单元|室)?)"),
        _mask_address,
    ),
    "name": (
        re.compile(r"(?P<prefix>姓名|名字|联系人|收件人|患者|客户|用户|开户人|持卡人)[:：\s]*(?P<name>[\u4e00-\u9fa5]{2,4})"),
        _mask_name,
    ),
}


LOOSE_CHINESE_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fa5])(?:[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣邓郁单欧闫])[\u4e00-\u9fa5]{1,2}(?=$|[\s,，.。;；:：!?！？、）)]|[的是在和])")
CONTEXT_CHINESE_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fa5])(?P<name>[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣邓郁单欧闫][\u4e00-\u9fa5]{1,2}?)(?=的?(?:邮箱|电子邮箱|电话|手机号|手机|身份证|银行卡|卡号|住址|地址|护照|车牌))")


def apply_regex_rules(text: str, enabled_entities: set[str], *, strict_name_rules: bool = False) -> str:
    masked = text
    for entity_type, (pattern, replacement) in RULES.items():
        if entity_type not in enabled_entities:
            continue
        masked = pattern.sub(replacement, masked)

    if "name" in enabled_entities and not strict_name_rules:
        masked = CONTEXT_CHINESE_NAME_PATTERN.sub(_mask_name, masked)
        masked = LOOSE_CHINESE_NAME_PATTERN.sub(lambda item: _mask_keep_edges(item.group(0), 1, 0), masked)

    return masked
