"""Deterministic transcript analysis and export. No AI here: these numbers are the ground truth the AI layer is
checked against."""
import json
import math
import re
from collections import Counter

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also get got going gonna yeah okay oh well really thing things one
""".split())
FILLERS = ("um", "uh", "uhm", "er", "erm", "ah", "hmm", "like", "basically", "literally", "actually")
FILLER_PHRASES = ("you know", "i mean", "sort of", "kind of")
LOW_CONFIDENCE_LOGPROB = -1.0  # ponytail: Whisper avg_logprob heuristic; tune per model if it flags too much
WORD_RE = re.compile(r"[\w']+", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?。！？]+")


def syllables(word: str) -> int:
    """Vowel-group heuristic (English). Good enough for readability scores, not for poetry."""
    w = word.lower()
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith("le") and groups > 1:
        groups -= 1
    return max(1, groups)


def text_stats(segments: list[dict], duration: float) -> dict:
    text = " ".join(s["text"].strip() for s in segments)
    words = WORD_RE.findall(text.lower())
    n = len(words)
    sentences = len([p for p in SENTENCE_RE.split(text) if p.strip()]) if n else 0
    speaking = sum(max(0.0, s["end"] - s["start"]) for s in segments)

    fillers = Counter(w for w in words if w in FILLERS)
    joined = " " + " ".join(words) + " "
    for phrase in FILLER_PHRASES:
        if c := joined.count(f" {phrase} "):
            fillers[phrase] = c
    keywords = Counter(w for w in words if len(w) > 2 and w not in STOPWORDS and w not in FILLERS and not w.isdigit())

    pauses = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap >= 1.0:
            pauses.append({"after_segment": i - 1, "at": round(segments[i - 1]["end"], 2), "seconds": round(gap, 2)})
    pauses.sort(key=lambda p: -p["seconds"])

    logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    syl = sum(syllables(w) for w in words)
    wps = n / sentences if sentences else 0
    return {
        "segment_count": len(segments),
        "word_count": n,
        "unique_words": len(set(words)),
        "lexical_diversity": round(len(set(words)) / n, 3) if n else 0,
        "sentence_count": sentences,
        "avg_sentence_words": round(wps, 1),
        "speaking_seconds": round(speaking, 2),
        "speech_ratio": round(speaking / duration, 3) if duration else 0,
        "wpm": round(n / speaking * 60, 1) if speaking else 0,
        # Flesch scores are English-calibrated; the UI labels them so.
        "flesch_reading_ease": round(206.835 - 1.015 * wps - 84.6 * syl / n, 1) if n and sentences else None,
        "fk_grade": round(0.39 * wps + 11.8 * syl / n - 15.59, 1) if n and sentences else None,
        "top_keywords": keywords.most_common(20),
        "filler_words": dict(fillers),
        "longest_pauses": pauses[:5],
        "avg_confidence": round(sum(math.exp(lp) for lp in logprobs) / len(logprobs), 3) if logprobs else None,
        "low_confidence_segments": [i for i, s in enumerate(segments)
                                    if s.get("avg_logprob") is not None and s["avg_logprob"] < LOW_CONFIDENCE_LOGPROB],
    }


def merge_translation(segments: list[dict], items: list[dict]) -> list[dict]:
    """Map AI translations back onto segment timing. Anything the AI skipped keeps its original text, flagged."""
    by_index = {it["index"]: it["text"] for it in items if isinstance(it.get("index"), int)}
    out = []
    for i, s in enumerate(segments):
        seg = {"start": s["start"], "end": s["end"], "text": by_index.get(i, s["text"])}
        if i not in by_index:
            seg["untranslated"] = True
        out.append(seg)
    return out


def fmt_ts(seconds: float, sep: str = ",") -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def short_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}:{rem // 60:02}:{rem % 60:02}" if h else f"{rem // 60:02}:{rem % 60:02}"


def _cue(text: str) -> str:
    """A blank line ends an SRT/VTT cue, so an edited segment may contain line breaks but never an empty line."""
    return re.sub(r"\n\s*\n+", "\n", text.strip())


def export(segments: list[dict], fmt: str, title: str = "Transcript") -> str:
    segments = [dict(s, text=_cue(s["text"])) for s in segments]
    if fmt == "txt":
        return "\n".join(s["text"].strip() for s in segments) + "\n"
    if fmt == "srt":
        return "\n".join(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text'].strip()}\n"
                         for i, s in enumerate(segments, 1))
    if fmt == "vtt":
        return "WEBVTT\n\n" + "\n".join(f"{fmt_ts(s['start'], '.')} --> {fmt_ts(s['end'], '.')}\n{s['text'].strip()}\n"
                                        for s in segments)
    if fmt == "md":
        return f"# {title}\n\n" + "".join(f"**[{short_ts(s['start'])}]** {s['text'].strip()}\n\n" for s in segments)
    if fmt == "json":
        return json.dumps(segments, ensure_ascii=False, indent=2)
    raise ValueError(f"unknown export format: {fmt}")
