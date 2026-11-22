import unittest

from app.pipeline import keyframe_times, parse_loudness, parse_scenes, parse_silence

SCENE_LOG = """
[Parsed_showinfo_2 @ 000001] n:   0 pts:  50000 pts_time:4.115   duration:1 fmt:yuv420p
[Parsed_showinfo_2 @ 000001] n:   1 pts: 160000 pts_time:12.8    duration:1 fmt:yuv420p
"""
AUDIO_LOG = """
[silencedetect @ 0001] silence_start: 1.5
[silencedetect @ 0001] silence_end: 3.5 | silence_duration: 2
[Parsed_ebur128_1 @ 0002] t: 0.1  TARGET:-23 LUFS    M: -70.0 S:-120.7     I: -70.0 LUFS       LRA:   0.0 LU
[silencedetect @ 0001] silence_start: 9
[Parsed_ebur128_1 @ 0002] Summary:

  Integrated loudness:
    I:         -18.4 LUFS
    Threshold: -28.6 LUFS

  Loudness range:
    LRA:         6.1 LU
"""


class Parsers(unittest.TestCase):
    def test_scenes(self):
        self.assertEqual(parse_scenes(SCENE_LOG), [4.115, 12.8])

    def test_silence_open_ended_runs_to_duration(self):
        self.assertEqual(parse_silence(AUDIO_LOG, duration=10.0),
                         [{"start": 1.5, "end": 3.5}, {"start": 9.0, "end": 10.0}])

    def test_loudness_uses_summary(self):
        self.assertEqual(parse_loudness(AUDIO_LOG), {"integrated_lufs": -18.4, "loudness_range_lu": 6.1})

    def test_loudness_missing(self):
        self.assertEqual(parse_loudness(""), {"integrated_lufs": None, "loudness_range_lu": None})


class Keyframes(unittest.TestCase):
    def test_pads_with_even_spacing_when_few_scenes(self):
        t = keyframe_times([], duration=60.0, minimum=6, maximum=12)
        self.assertEqual(len(t), 6)
        self.assertTrue(all(0 < x < 60 for x in t))

    def test_caps_many_scenes(self):
        t = keyframe_times([float(i) for i in range(1, 100)], duration=100.0, minimum=6, maximum=12)
        self.assertEqual(len(t), 12)
        self.assertEqual(t, sorted(t))

    def test_zero_duration(self):
        self.assertEqual(keyframe_times([], duration=0, minimum=6, maximum=12), [])


class CudaFallback(unittest.TestCase):
    def setUp(self):
        from app import pipeline
        self.p = pipeline
        self.saved = (pipeline._load, pipeline._cuda_broken, pipeline.cuda_available)
        self.loads = []

        def fake_load(name, device):
            self.loads.append(device)
            return device

        pipeline._load, pipeline._cuda_broken = fake_load, False
        pipeline._model.clear()
        pipeline.cuda_available = lambda: not pipeline._cuda_broken

    def tearDown(self):
        self.p._load, self.p._cuda_broken, self.p.cuda_available = self.saved
        self.p._model.clear()

    def test_missing_cuda_libs_fall_back_to_cpu(self):
        def work():
            if self.p.get_model("tiny") == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return "ok"
        self.assertEqual(self.p.with_cuda_fallback(work), "ok")
        self.assertEqual(self.loads, ["cuda", "cpu"])
        self.assertTrue(self.p._cuda_broken)

    def test_other_errors_propagate(self):
        def work():
            raise RuntimeError("corrupt file")
        with self.assertRaises(RuntimeError):
            self.p.with_cuda_fallback(work)
        self.assertFalse(self.p._cuda_broken)


class SpeakerStageFailure(unittest.TestCase):
    def test_speaker_failure_keeps_the_transcript(self):
        import tempfile
        from pathlib import Path
        from app import pipeline
        names = ("probe", "extract_audio", "transcribe", "scan_audio")
        saved, saved_diarize = {n: getattr(pipeline, n) for n in names}, pipeline.speakers.diarize
        pipeline.probe = lambda src: {"duration": 2.0, "audio": {"codec": "aac"}, "video": None}
        pipeline.extract_audio = lambda src, d, dur: Path(d) / "audio.wav"
        pipeline.transcribe = lambda wav, m, lang, p: {"segments": [{"start": 0.0, "end": 1.0, "text": "hi there"}],
                                                        "language": "en", "language_probability": 1.0}
        pipeline.scan_audio = lambda src, dur: {"silences": []}

        def offline(*a):
            raise OSError("model download failed: no network")
        pipeline.speakers.diarize = offline
        stages = {}

        def update(stage, status=None, progress=None, **fields):
            if status:
                stages[stage] = status
        opts = {"model": "tiny", "language": None, "ai": False, "ocr": False, "speakers": True, "num_speakers": 0,
                "translate_to": ""}
        try:
            with tempfile.TemporaryDirectory() as d:
                pipeline.process({"dir": d}, update, Path(d) / "x.mp4", None, opts)
                self.assertEqual((Path(d) / "transcript.txt").read_text(encoding="utf-8"), "hi there\n")
        finally:
            for n, f in saved.items():
                setattr(pipeline, n, f)
            pipeline.speakers.diarize = saved_diarize
        self.assertEqual((stages["speakers"], stages["transcribe"]), ("failed", "done"))


class ModelDownload(unittest.TestCase):
    def test_checksum_mismatch_keeps_nothing_and_match_installs(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from app import speakers
        with tempfile.TemporaryDirectory() as d:
            src, dest = Path(d) / "src.bin", Path(d) / "models" / "m.onnx"
            src.write_bytes(b"model bytes")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                speakers._fetch(src.as_uri(), dest, "0" * 64)
            self.assertEqual(list(dest.parent.iterdir()), [])  # no model, no .part left behind
            good = hashlib.sha256(b"model bytes").hexdigest()
            self.assertEqual(speakers._fetch(src.as_uri(), dest, good).read_bytes(), b"model bytes")
            dest.write_bytes(b"truncated")  # a damaged cached copy is replaced, not trusted
            self.assertEqual(speakers._fetch(src.as_uri(), dest, good).read_bytes(), b"model bytes")


class Timeouts(unittest.TestCase):
    def test_budget_scales_with_duration(self):
        from app.pipeline import budget
        self.assertGreater(budget(3600), budget(60))
        self.assertGreaterEqual(budget(0), 3600)  # unknown length: generous fixed cap

    def test_hung_subprocess_becomes_readable_error(self):
        import sys
        from app.pipeline import run
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.5)


if __name__ == "__main__":
    unittest.main()
