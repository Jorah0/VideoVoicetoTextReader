import json
import unittest

from app.analysis import export, fmt_ts, merge_translation, syllables, text_stats

SEGS = [
    {"start": 0.0, "end": 4.0, "text": "Um, the quick brown fox jumps over the lazy dog.", "avg_logprob": -0.2},
    {"start": 6.5, "end": 10.0, "text": "The fox is quick. Like, really quick!", "avg_logprob": -1.4},
]


class TextStats(unittest.TestCase):
    def setUp(self):
        self.s = text_stats(SEGS, duration=12.0)

    def test_counts(self):
        self.assertEqual(self.s["word_count"], 17)
        self.assertEqual(self.s["sentence_count"], 3)
        self.assertEqual(self.s["segment_count"], 2)

    def test_rates_use_speaking_time(self):
        self.assertAlmostEqual(self.s["speaking_seconds"], 7.5)
        self.assertAlmostEqual(self.s["wpm"], 17 / 7.5 * 60, places=1)
        self.assertAlmostEqual(self.s["speech_ratio"], 7.5 / 12.0, places=3)

    def test_keywords_skip_stopwords(self):
        words = [w for w, _ in self.s["top_keywords"]]
        self.assertEqual(words[:2], ["quick", "fox"])
        self.assertNotIn("the", words)

    def test_fillers_and_pauses(self):
        self.assertEqual(self.s["filler_words"], {"um": 1, "like": 1})
        self.assertEqual(self.s["longest_pauses"][0], {"after_segment": 0, "at": 4.0, "seconds": 2.5})

    def test_low_confidence_flagged(self):
        self.assertEqual(self.s["low_confidence_segments"], [1])

    def test_empty_transcript(self):
        s = text_stats([], duration=0)
        self.assertEqual(s["word_count"], 0)
        self.assertEqual(s["wpm"], 0)
        self.assertEqual(s["flesch_reading_ease"], None)


class Helpers(unittest.TestCase):
    def test_syllables(self):
        self.assertEqual(syllables("the"), 1)
        self.assertEqual(syllables("quickly"), 2)
        self.assertEqual(syllables("beautiful"), 3)

    def test_fmt_ts(self):
        self.assertEqual(fmt_ts(3725.5), "01:02:05,500")
        self.assertEqual(fmt_ts(1.0, "."), "00:00:01.000")

    def test_merge_translation_keeps_untranslated_segments(self):
        merged = merge_translation(SEGS, [{"index": 1, "text": "El zorro es rápido."}, {"index": 9, "text": "x"}])
        self.assertEqual(merged[1]["text"], "El zorro es rápido.")
        self.assertEqual(merged[0]["text"], SEGS[0]["text"])
        self.assertTrue(merged[0]["untranslated"])
        self.assertEqual(len(merged), 2)


class Export(unittest.TestCase):
    def test_srt(self):
        self.assertEqual(
            export(SEGS, "srt"),
            "1\n00:00:00,000 --> 00:00:04,000\nUm, the quick brown fox jumps over the lazy dog.\n\n"
            "2\n00:00:06,500 --> 00:00:10,000\nThe fox is quick. Like, really quick!\n",
        )

    def test_vtt(self):
        out = export(SEGS, "vtt")
        self.assertTrue(out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:04.000\n"))

    def test_txt_and_md_and_json(self):
        self.assertEqual(export(SEGS, "txt").splitlines()[0], SEGS[0]["text"])
        self.assertIn("**[00:06]**", export(SEGS, "md", title="Demo"))
        self.assertEqual(json.loads(export(SEGS, "json"))[1]["start"], 6.5)

    def test_blank_lines_inside_a_segment_cannot_split_a_cue(self):
        out = export([{"start": 0, "end": 1, "text": "one\n\n\ntwo"}], "srt")
        self.assertEqual(out, "1\n00:00:00,000 --> 00:00:01,000\none\ntwo\n")

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            export(SEGS, "exe")


if __name__ == "__main__":
    unittest.main()
