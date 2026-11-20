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


if __name__ == "__main__":
    unittest.main()
