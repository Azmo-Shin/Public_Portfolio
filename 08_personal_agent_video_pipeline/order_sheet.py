# -*- coding: utf-8 -*-
"""주문서 관문 — 봇이 쓴 정해진 줄만 읽고, 자유 문장은 해석하지 않는다. (발췌·익명화)

    주제 : 신경망 순전파·역전파
    성격 : 개념            (개념 · 최신 · 비워 둠)
    길이 : 60초
    화면 : 가로            (가로 · 세로 · 정사각)
    느낌 : 깔끔            (스타일 사전의 이름)

원칙: 주제·화면·느낌은 있어야 주문서다. 성격·길이는 비어도 통과(기본값 + "짐작" 기록).
      허사만 남은 주제는 거절한다 — 문장이 잘못 잘렸다는 뜻이고, 그대로 만들면 20분이 헛것이 된다.
"""
import re
from collections import namedtuple

STYLES = {"깔끔": "clean", "강렬": "bold", "따뜻": "warm", "밤": "night", "리포트": "report"}   # 운영에서는 json 한 곳
DEFAULT_STYLE, DEFAULT_ASPECT, DEFAULT_SECONDS = "clean", "16:9", 60
SECONDS_MIN, SECONDS_MAX = 5, 300

# 화면비 어휘의 원본은 이 목록 하나. 표(ASPECT_TABLE)와 정규식(ASPECTS)이 같은 목록에서 나와야
# 「릴스」를 지침서는 알고 주문서 표만 모르는 일이 생기지 않는다.
ASPECT_WORDS = (("9:16", ("세로", "쇼츠", "숏폼", "쇼트", "릴스", "틱톡", "9:16")),
                ("1:1", ("정사각", "1:1")),
                ("16:9", ("가로", "유튜브 영상", "16:9")))

# 낱말이 낱말로 서 있는가: 앞에는 글자가 없어야 하고(어젯밤·쇼트트랙), 뒤에는 조사·꼬리만 허용한다.
WORD_TAIL = ("형태", "형식", "스타일", "분위기", "느낌", "버전", "모드", "짜리", "하게", "하고", "처럼", "같이",
             "으로", "형", "폼", "용", "판", "톤", "한", "은", "는", "이", "가", "을", "를", "의", "도", "만", "에", "로", "요")
_TAIL = "(?:%s)*" % "|".join(WORD_TAIL)
_HEAD_EDGE, _TAIL_EDGE = r"(?<![가-힣A-Za-z0-9])", r"(?![가-힣A-Za-z0-9])"


def word_pattern(words):
    body = "|".join(r"\s*".join(re.escape(p) for p in w.split()) for w in words)
    return re.compile(_HEAD_EDGE + "(?:" + body + ")" + _TAIL + _TAIL_EDGE)


ASPECTS = [(word_pattern(ws), val) for val, ws in ASPECT_WORDS]
ASPECT_TABLE = {w: val for val, ws in ASPECT_WORDS for w in ws}

KEYS = ("주제", "성격", "길이", "화면", "느낌")
OPTIONAL = ("성격", "길이")
NEEDED = tuple(k for k in KEYS if k not in OPTIONAL)

# 아무것도 이름 짓지 못하는 낱말. 이것만으로 된 주제는 주제가 아니다.
EMPTY_WORDS = {"것", "것들", "거", "그거", "이거", "그것", "이것", "대한", "대해", "대해서", "관한", "관해", "등", "들",
               "무엇", "뭐", "얘기", "이야기", "내용", "부분", "쪽", "면", "그", "저",
               "에", "의", "를", "을", "이", "가", "은", "는"}


def seconds(text):
    """길이 줄 → 초. 분은 펴고 5~300 안으로 잡는다(「1분」이 1초가 되지 않게). 숫자가 없으면 None."""
    s = str(text or "")
    m = re.search(r"([0-9]{1,3})\s*분", s)
    if m:
        secs = int(m.group(1)) * 60
    else:
        m = re.search(r"([0-9]{1,4})", s)
        if not m:
            return None
        secs = int(m.group(1))
    return max(SECONDS_MIN, min(SECONDS_MAX, secs))


def kind(text):
    """성격 줄 → "개념" · "최신" · ""(짐작해 달라). 임의로 채우면 사용자가 정한 것과 구별이 안 된다."""
    s = str(text or "")
    return "최신" if "최신" in s else ("개념" if "개념" in s else "")


def aspect(text):
    """화면 줄 → 화면비. 표에 없으면 낱말 경계 정규식으로, 그래도 없으면 기본값."""
    s = str(text or "").strip()
    if s in ASPECT_TABLE:
        return ASPECT_TABLE[s]
    for pat, val in ASPECTS:
        if pat.search(s):
            return val
    return DEFAULT_ASPECT


def style(text):
    """느낌 줄 → 스타일. 모르는 말이면 기본값 — 느낌은 색과 움직임일 뿐이라 기본값으로 두는 값이 싸다."""
    s = str(text or "").strip()
    table = {}
    for ko, en in STYLES.items():
        table[ko] = table[en] = en
    if s in table:
        return table[s]
    for key in sorted(table, key=len, reverse=True):      # 「밤」 같은 짧은 이름이 긴 이름 안에 들 수 있으니 긴 것부터
        if key and key in s:
            return table[key]
    return DEFAULT_STYLE


def junk_topic(topic):
    """낱말이 전부 조사·지시어면 True — 문장이 잘못 잘렸다는 뜻."""
    words = [w.strip(" .,!?~-") for w in str(topic or "").split()]
    words = [w for w in words if w]
    if not words:
        return True
    for w in words:
        bare = w
        for tail in ("에", "의", "를", "을", "이", "가", "은", "는", "들"):
            if bare.endswith(tail) and len(bare) > len(tail):
                bare = bare[:-len(tail)]
                break
        if w not in EMPTY_WORDS and bare not in EMPTY_WORDS:
            return False
    return True


Order = namedtuple("Order", "topic kind seconds aspect style")


def read(text):
    """주문서 → Order. 주문서가 아니면 None."""
    got = {}
    for line in str(text or "").splitlines():
        s = line.strip().lstrip("*-· ").strip()
        for key in KEYS:
            if s.startswith(key + " :") or s.startswith(key + ":"):
                val = s.split(":", 1)[1].strip().strip("*` ")
                if val and key not in got:
                    got[key] = val
                break
    if any(k not in got for k in NEEDED):
        return None
    topic = got["주제"].strip()
    if junk_topic(topic):          # 봇이 썼다고 덜 위험하지 않다
        return None
    secs = seconds(got.get("길이", ""))
    return Order(topic=topic, kind=kind(got.get("성격", "")), seconds=DEFAULT_SECONDS if secs is None else secs,
                 aspect=aspect(got["화면"]), style=style(got["느낌"]))


if __name__ == "__main__":
    print(read("주제 : 신경망 순전파·역전파\n성격 : 개념\n길이 : 1분\n화면 : 쇼츠형으로\n느낌 : 깔끔하게"))
    # Order(topic='신경망 순전파·역전파', kind='개념', seconds=60, aspect='9:16', style='clean')
    print(read("주제 : 에 대한 것들\n화면 : 가로\n느낌 : 깔끔"))    # None — 허사뿐인 주제
    print(aspect("쇼트트랙 경기"), aspect("세로축 그래프"))          # 16:9 16:9 — 낱말 경계
