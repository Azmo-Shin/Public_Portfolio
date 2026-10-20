# -*- coding: utf-8 -*-
"""zero-shot TTS 끝말 잘림 방어 — 희생 단어 · 길이 판정 · 받아쓰기 · 재생성. (발췌·익명화, TTS 서버 호출부는 생략)

문제: zero-shot TTS 라이브러리가 문장 끝을 확률적으로 일찍 끊는다(실측 약 60%, 평균 0.8초 짧음).
      대본 전체를 한 번에 넘기면 마지막 문장들이 통째로 사라져도 파이프라인은 모른다(339자 대본에서 단어 74%만 존재).

장치:
  1. 문장 단위로 생성하고 끝에 희생 단어(" 네.")를 붙인다 → 일찍 끊겨도 진짜 마지막 음절은 중간에 있다.
  2. 생성 후 마지막 침묵에서 잘라낸다. 자를 자리는 초 단위로 판정(비율로 보면 짧은 문장에서 희생 단어가 남는다).
  3. 잘렸는지는 「말한 길이 vs 글자 수로 기대되는 길이」로 판정한다 — 끝 음량으로 보던 판정은 세 번 다 틀렸다.
  4. 길이가 맞아도 다른 말일 수 있으니 로컬 Whisper 받아쓰기와 비교한다(≥0.60).
  5. 실패하면 시드를 바꿔 최대 MAX_RETRY. 전부 실패해도 소리가 있고 길이가 맞는 것이 있으면 경고와 함께 쓴다.
"""
import difflib
import random
import re

MAX_RETRY = 8
TAIL_WORD = " 네."
GAP_MIN_MS, GAP_THRESH = 110, 180          # 이만큼 이어지는 조용한 구간을 「문장 끝 침묵」으로 본다
STRIP_KEEP_MIN_S = 0.30                    # 자르고도 본문이 이만큼은 남아야
STRIP_CUT_MIN_S, STRIP_CUT_MAX_S = 0.08, 1.30
TAIL_VOICE_MAX_S = 0.60                    # 잘라낸 말소리가 이보다 길면 본문을 자른 것
CHARS_PER_SEC = 6.5                        # 이 기계·이 목소리에서 실측한 초당 글자 수(속도 1.0). 프로필마다 다르다
SPEED = 1.06
SHORT_RATIO = 0.78                         # 기대 길이의 이 비율에 못 미치면 잘린 것
LEN_MIN_CHARS = 6                          # 더 짧은 문장은 비율이 들쭉날쭉해 건너뛴다
ABS_MIN_SECS = 0.35
TOO_LONG_MIN_S, TOO_LONG_SECS_PER_CHAR = 7.0, 0.42   # 글자 수 대비 너무 길면 모델이 되풀이한 것
SILENCE_RMS_LIMIT = 100
HEARD_MIN, HEARD_SOFT, HEARD_MIN_CHARS = 0.60, 0.35, 6
LATIN_NAMES = {"a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지", "h": "에이치", "i": "아이",
               "j": "제이", "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알", "s": "에스",
               "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이", "z": "제트"}


# ---------- 소리 재기 ----------
def _frame_rms(samples, sr, ms=20):
    win = max(1, int(sr * ms / 1000.0))
    return win, [(sum(v * v for v in samples[i:i + win]) / win) ** 0.5 for i in range(0, len(samples) - win, win)]


def overall_rms(samples):
    return (sum(v * v for v in samples) / float(len(samples))) ** 0.5 if samples else 0.0


def cut_at_last_gap(samples, sr):
    """뒤에서부터 문장 끝 침묵을 찾는다 → (자를 위치 or None, 잘라낼 말소리 길이 초).
    끝의 침묵을 먼저 건너뛴다 — 바로 뒤에서부터 찾으면 희생 단어 뒤의 조용함만 잘라내고 단어는 남는다.
    희생 단어 분량보다 많이 잘라내는 후보는 문장 안의 쉼이므로 거부하고 더 느슨한 기준으로 다시 찾는다."""
    win, frames = _frame_rms(samples, sr)
    if not frames or len(samples) < win * 4:
        return None, 0.0

    def removed_secs(pos, last_voice):
        return (last_voice * win - pos) / float(sr)

    def ok(pos, last_voice):
        removed = removed_secs(pos, last_voice)
        return pos / float(sr) >= STRIP_KEEP_MIN_S and STRIP_CUT_MIN_S <= removed <= min(STRIP_CUT_MAX_S, TAIL_VOICE_MAX_S)

    for gap_ms, thresh in ((GAP_MIN_MS, GAP_THRESH), (70, GAP_THRESH), (50, 300)):
        need = max(1, int(gap_ms / 20))
        last_voice = len(frames) - 1
        while last_voice >= 0 and frames[last_voice] < thresh:
            last_voice -= 1
        run = 0
        for k in range(last_voice, -1, -1):
            if frames[k] < thresh:
                run += 1
                if run >= need:
                    pos = k * win
                    if ok(pos, last_voice):
                        return pos, removed_secs(pos, last_voice)
                    run = 0
            else:
                run = 0
    return None, 0.0


def strip_tail_word(samples, sr):
    cut, removed = cut_at_last_gap(samples, sr)
    return (samples, False, 0.0) if cut is None else (samples[:cut], True, removed)


# ---------- 판정 ----------
def expected_secs(text):
    return len(text) / (CHARS_PER_SEC * SPEED)


def is_short_for_text(text, secs):
    """말한 길이가 글자 수에 못 미치면 끊긴 것. 자르는 방법과 무관한 유일한 잣대."""
    return len(text) >= LEN_MIN_CHARS and secs < expected_secs(text) * SHORT_RATIO


def judge_take(text, samples, sr, stripped):
    secs = len(samples) / float(sr)
    if overall_rms(samples) < SILENCE_RMS_LIMIT:
        return "소리가 없어요"
    if not stripped:
        return "끝이 잘렸어요" if is_short_for_text(text, secs) else "뒷말이 그대로 남았어요"
    if is_short_for_text(text, secs):
        return "끝이 잘렸어요"
    if secs < ABS_MIN_SECS:
        return "너무 짧음"
    if secs > max(TOO_LONG_MIN_S, len(text) * TOO_LONG_SECS_PER_CHAR):
        return "너무 김"
    return "정상"


def _heard_key(text):
    """비교용 꼴: 알파벳은 글자 이름으로(「GPT」 ↔ 「지피티」), 숫자 사이 쉼표 제거, 한글·숫자만 남김."""
    s = re.sub(r"(?<=\d),(?=\d)", "", str(text or ""))
    s = re.sub(r"[A-Za-z]", lambda m: LATIN_NAMES[m.group(0).lower()], s)
    return re.sub(r"[^0-9가-힣]", "", s)


def heard_ok(want, heard):
    """→ (써도 되나, 비율). 받아쓴 말이 대본에 없는 희생 단어로 끝나면 비율과 상관없이 안 된다(잘라내기가 빗나간 것)."""
    tail = _heard_key(TAIL_WORD)
    if tail and _heard_key(heard).endswith(tail) and not _heard_key(want).endswith(tail):
        return False, 0.0
    if len(_heard_key(want)) < HEARD_MIN_CHARS:
        return True, None
    r = difflib.SequenceMatcher(None, _heard_key(want), _heard_key(heard)).ratio()
    return r >= HEARD_MIN, r


# ---------- 재생성 루프 뼈대 ----------
def generate_clean(text, synth, transcribe, warnings):
    """synth(text, seed) → (samples, sr) · transcribe(samples, sr) → str 를 받아 쓸 만한 녹음을 돌려준다.
    한 문장의 실패로 이미 만든 문장들(GPU 10~20분)을 버리지 않는다."""
    best, closest = None, None
    for attempt in range(1, MAX_RETRY + 1):
        seed = random.randint(1, 2_000_000_000)
        try:
            s, sr = synth(text + TAIL_WORD, seed)
        except Exception:                       # 한 번의 서버 오류는 한 번의 시도일 뿐
            continue
        s, stripped, _cut = strip_tail_word(s, sr)
        mark = judge_take(text, s, sr, stripped)
        if mark == "정상":
            try:
                heard = transcribe(s, sr)
            except Exception:
                warnings.append("받아쓰기를 못 해 말 내용은 확인 못 했어요")
                return s, sr
            ok, ratio = heard_ok(text, heard)
            if ok:
                return s, sr
            if closest is None or (ratio or 0) > closest[0]:
                closest = (ratio or 0, s, sr, heard)
            continue
        if mark == "뒷말이 그대로 남았어요" and (best is None or len(s) > len(best[0])):
            best = (s, sr)
    if best is not None:
        warnings.append("뒷말이 남은 녹음을 그대로 씁니다")
        return best
    if closest is not None and closest[0] >= HEARD_SOFT:
        warnings.append("받아쓰기와 %.0f%%만 같아요 — 들어 보세요" % (100 * closest[0]))
        return closest[1], closest[2]
    raise RuntimeError("%d번 다시 만들어도 쓸 만한 소리가 없어요" % MAX_RETRY)
