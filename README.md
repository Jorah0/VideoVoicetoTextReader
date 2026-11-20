# Video Voice-to-Text Reader

Drop in a video file or paste a URL (YouTube and the 1,800 other sites yt-dlp supports, or a direct media link).
You get an editable, timestamped transcript plus text, video and audio analysis. An optional Claude layer then
double-checks the language and coherence, categorizes the content, translates it and reads on-screen text.

```
pip install -r requirements.txt        # needs ffmpeg/ffprobe on PATH
set ANTHROPIC_API_KEY=sk-ant-...       # optional: enables the AI layer
python -m app.server                   # opens http://127.0.0.1:8765
```

## What runs

| Stage | Engine | Runs |
|---|---|---|
| Acquire | yt-dlp (URL) or upload | first |
| Probe | ffprobe: container, codecs, resolution, fps, bitrate | first |
| Speech → text | faster-whisper (`large-v3` on GPU by default), VAD, auto language detection | **in parallel** |
| Video scan | ffmpeg scene-cut detection, shot length, keyframes | **in parallel** |
| Audio scan | EBU R128 loudness, silence detection | **in parallel** |
| On-screen text | Claude vision reads every keyframe (titles, slides, captions) | after the video scan |
| Text analysis | words, words per minute, readability, keywords, filler words, pauses, low-confidence lines (no AI) | after transcription |
| AI review | Claude: language check against Whisper, coherence score, per-line corrections, category, topics, sentiment, entities, chapters, quotes, action items | after analysis |
| Translate | Claude, any language (Whisper fallback for English without a key) | last |

Outputs land in `outputs/<job>/`: `transcript.txt`, `transcript.srt`, `translation.txt` and `job.json`. Edits you
save in the UI are written next to them as `transcript_edited.<fmt>`. The raw machine transcript is never
overwritten, and AI corrections are suggestions you apply line by line.

The model is `claude-opus-5-5` with structured outputs, adaptive thinking and server-side refusal fallback
(`app/ai.py`). Without an API key everything except the AI panels still works.

## Verify

```
python scripts/verify.py          # compile + 36 unit/API tests
python scripts/verify.py --full   # + end-to-end: synthesized speech video -> Whisper on GPU -> API -> export
```

Not included: speaker diarization (who said what). It needs pyannote and a Hugging Face token; add it when you need it.
