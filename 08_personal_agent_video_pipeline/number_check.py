# -*- coding: utf-8 -*-
"""숫자 대조 — 대본의 숫자가 자료에 글자 그대로 있는지 코드가 맞대 본다. (발췌·익명화)

왜: 자료의 「0.12%」가 대본에서 「영 점 일이칠 퍼센트」, 화면에서 「0.17%」가 됐고, 자료 한 줄의 「13.02%」는
원문에 없었다. 「지어내지 마라」로는 안 잡힌다.

센 숫자 = 소수점이 있거나, 쉼표로 끊었거나, 세 자리 이상. 한두 자리 정수는 %·조·억이 붙을 때만 단위째.
날짜·시각(붙은 년·월·일·시·분·초)과 작은 수(3개월·5거래일)는 보지 않는다.
같은 숫자: 쉼표(7,003.74 = 7003.74) · 한국식 적기(1천680 = 1680) · 끝의 0(29.8 = 29.80) · 끊어 적은 정수(893 ← 893.22).
다른 숫자: 0.12 ≠ 0.127 · 13% ≠ 13.5% · 2.00 ≠ 2일.
"""
import re

_NUM = re.compile(r"(?<![\d.])(?<!\d,)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d|[.,]\d)")
_PCT_WORDS = r"(?:%|％|퍼센트|percent)"
_PCT = re.compile(r"\s*" + _PCT_WORDS)
_DATE_UNIT = re.compile(r"(?:년|월|일(?!자리|본|반)|시(?!대|장|가|세|총|즌|리즈)|분(?!의|기|양|석)|초(?!과|반|기))")
_BIG_UNIT = re.compile(r" ?(조|억)")
_ISO_DATE = re.compile(r"-\d{1,2}-\d{1,2}(?!\d)")
_UNITS = ("%", "조", "억")
STRONG_DIGITS = 3


def _scan(text):
    """(쉼표 뗀 숫자, 갈래 strong/date/small, 붙은 단위) 를 나온 차례로."""
    s = str(text or "")
    for m in _NUM.finditer(s):
        whole, frac = m.group(1), m.group(2)
        digits = whole.replace(",", "")
        token = digits + ("." + frac if frac is not None else "")
        rest = s[m.end():m.end() + 8]
        unit = "%" if _PCT.match(rest) else ""
        if frac is not None:
            kind = "strong"
        elif _DATE_UNIT.match(rest) or (len(digits) == 4 and _ISO_DATE.match(rest)):
            kind, unit = "date", ""
        elif "," in whole or len(digits) >= STRONG_DIGITS:
            kind = "strong"
        else:
            kind = "small"
            if not unit:
                big = _BIG_UNIT.match(rest)
                unit = big.group(1) if big else ""
        yield token, kind, unit


def _split(token):
    t = str(token or "")
    return (t[:-1], t[-1]) if t[-1:] in _UNITS else (t, "")


# 한국식 적기 전개. 큰 덩이부터 바꿔 「3만5천」 속의 「5천」이 따로 5000이 되지 않게 한다.
_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_D = r"(?<![\d.])(\d+)"
_DEC_UNIT = re.compile(_D + r"\.(\d{1,4})(만|천)(?!\d)")
_MAN_CHEON = re.compile(_D + r"만 ?(\d+)천(?:(\d{1,3})(?!백)| (\d{3}))?(?!\d)")
_MAN_NUM = re.compile(_D + r"만(?:(\d{1,4})| (\d{4}))(?!\d|천)")
_CHEON_NUM = re.compile(_D + r"천(?:(\d{1,3})(?!백)| (\d{3}))(?!\d)")
_CHEON = re.compile(_D + r"천(?!\d)")
_MAN = re.compile(_D + r"만(?!\d|큼)")      # 「2만큼」의 만은 조사


def _dec_unit(m):
    whole, frac, unit = m.group(1), m.group(2), m.group(3)
    width = 4 if unit == "만" else 3
    if len(frac) > width:
        return m.group(0)
    return str(int(whole) * 10 ** width + int(frac.ljust(width, "0")))


def expand(text):
    """1천680 → 1680 · 3만5천 → 35000 · 27만 6000 → 276000 · 7천 → 7000 · 27.6만 → 276000. 쉼표도 뗀다."""
    t = _COMMA.sub("", str(text or ""))
    t = _DEC_UNIT.sub(_dec_unit, t)
    t = _MAN_CHEON.sub(lambda m: str(int(m.group(1)) * 10000 + int(m.group(2)) * 1000 + int(m.group(3) or m.group(4) or 0)), t)
    t = _MAN_NUM.sub(lambda m: str(int(m.group(1)) * 10000 + int(m.group(2) or m.group(3))), t)
    t = _CHEON_NUM.sub(lambda m: str(int(m.group(1)) * 1000 + int(m.group(2) or m.group(3))), t)
    t = _CHEON.sub(lambda m: str(int(m.group(1)) * 1000), t)
    return _MAN.sub(lambda m: str(int(m.group(1)) * 10000), t)


def strong(text):
    """글의 센 숫자(단위 붙은 것은 단위째: 13% · 18조 · 13.0%)."""
    out = []
    for token, kind, unit in _scan(expand(text)):
        if kind == "strong":
            zero = unit == "%" and "." in token and not token.split(".", 1)[1].strip("0")
            key = token + "%" if zero else token
        elif kind == "small" and unit:
            key = token + unit
        else:
            continue
        if key not in out:
            out.append(key)
    return out


def flat(page_text):
    """원문을 대조하기 좋게: 쉼표 뗀 원문 + 한국식 적기를 편 원문."""
    t = _COMMA.sub("", str(page_text or ""))
    wide = expand(t)
    return t if wide == t else t + "\n" + wide


def has(flat_text, token):
    """원문에 그 숫자가 있나. 소수는 뒤에 숫자가 더 오면 다른 숫자(0.12 ≠ 0.127), 끝의 0은 같은 숫자,
    정수는 소수의 앞머리에도 맞는다(893 ← 893.22). 단위가 붙은 것은 단위까지 맞아야 한다."""
    t = str(flat_text or "")
    num, unit = _split(token)
    if not num or not t:
        return False
    whole, dot, frac = num.partition(".")
    frac = frac.rstrip("0")
    if frac:
        body = re.escape(whole) + r"\." + re.escape(frac) + r"0*(?!\d)"
    elif unit:
        body = re.escape(whole) + r"(?:\.0+)?(?!\d|\.\d)"
    elif dot:
        body = re.escape(whole) + r"\.0+(?!\d)"
    else:
        body = re.escape(whole) + r"(?!\d)"
    if unit == "%":
        body += r"\s*" + _PCT_WORDS
    elif unit:
        body += " ?" + unit
    return bool(re.search(r"(?<![\d.])" + body, t, re.I))


def missing(fact_text, page_text):
    """사실 문장의 센 숫자 가운데 원문에 없는 것."""
    ft = flat(page_text)
    return [tok for tok in strong(fact_text) if not has(ft, tok)]


def _round(num):
    """어림수(7000 · 300)인가 — 「7천선」은 7207의 오타가 아니다."""
    return "." not in num and len(num) >= 3 and not num[1:].strip("0")


def near(token, candidates):
    """모양이 같고 한 자리만 다르거나 이웃 두 자리가 바뀐 후보, 또는 소수 끝에 한두 자리가 더 붙거나 빠진 후보."""
    a, ua = _split(token)
    if len(a.replace(".", "")) < 3 or _round(a):
        return None
    for c in candidates or ():
        b, ub = _split(c)
        if ub != ua or b == a or len(b.replace(".", "")) < 3 or _round(b):
            continue
        if len(a) == len(b) and a.find(".") == b.find("."):
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if len(diff) == 1 or (len(diff) == 2 and diff[1] - diff[0] == 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]):
                return str(c)
        elif "." in a and "." in b and 1 <= abs(len(a) - len(b)) <= 2 and (a.startswith(b) or b.startswith(a)):
            return str(c)
    return None


def script_check(script, facts_text):
    """대본의 센 숫자를 자료와 맞대 본다 → {"numbers": 본 수, "missing": [[대본 숫자, 비슷한 자료 숫자 or None]], "small": 안 본 수}."""
    toks = strong(script)
    ft = flat(facts_text)
    cands = strong(facts_text)
    miss = [[t, near(t, cands)] for t in toks if not has(ft, t)]
    small = {token for token, kind, unit in _scan(expand(script)) if kind == "small" and not unit}
    return {"numbers": len(toks), "missing": miss, "small": len(small)}


if __name__ == "__main__":
    facts = "코스피는 7,003.74로 마감했고 거래대금은 27만 6000억원, 외국인은 0.12% 순매수. 2026-10-02 기준."
    print(strong(facts))                                   # ['7003.74', '276000', '0.12'] (연도는 날짜라 제외, 소수 %는 숫자만)
    print(script_check("코스피 7003선 마감, 거래대금 276000억, 외국인 0.127% 순매수, 3거래일 연속", facts))
    # {'numbers': 3, 'missing': [['0.127', '0.12']], 'small': 1}  — 7003 ← 7003.74 는 통과, 0.127 은 이웃 숫자 제시
