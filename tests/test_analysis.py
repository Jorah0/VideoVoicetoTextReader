import json
import unittest

from app.analysis import assign_speakers, export, fmt_ts, merge_translation, syllables, text_stats

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


TURNS = [{"start": 0.0, "end": 5.0, "speaker": 3}, {"start": 5.5, "end": 10.0, "speaker": 1}]
DIALOG = [dict(SEGS[0], speaker="Speaker 1"), {"start": 4.2, "end": 6.0, "text": "And more.", "speaker": "Speaker 1"},
          dict(SEGS[1], speaker="Speaker 2")]


class Speakers(unittest.TestCase):
    def test_label_by_largest_overlap_numbered_in_order_of_first_speech(self):
        out = assign_speakers(SEGS, TURNS)
        self.assertEqual([s["speaker"] for s in out], ["Speaker 1", "Speaker 2"])
        self.assertNotIn("speaker", SEGS[0])  # input untouched

    def test_segment_far_from_any_turn_stays_unlabelled(self):
        out = assign_speakers([{"start": 30.0, "end": 31.0, "text": "x"}, {"start": 10.5, "end": 11.0, "text": "y"}], TURNS)
        self.assertNotIn("speaker", out[0])
        self.assertEqual(out[1]["speaker"], "Speaker 1")  # no overlap, but within 2s of a turn; first voice labelled

    def test_voice_heard_only_outside_speech_takes_no_number(self):
        turns = [{"start": 0.0, "end": 1.0, "speaker": 7}] + TURNS  # e.g. music before anyone talks
        out = assign_speakers([{"start": 2.0, "end": 4.0, "text": "a"}, {"start": 6.0, "end": 9.0, "text": "b"}], turns)
        self.assertEqual([s["speaker"] for s in out], ["Speaker 1", "Speaker 2"])

    def test_unlabelled_line_does_not_split_a_turn(self):
        segs = [dict(DIALOG[0]), {"start": 4.0, "end": 4.1, "text": "hm"}, dict(DIALOG[1])]
        self.assertEqual(text_stats(segs, 12.0)["speakers"][0]["turns"], 1)

    def test_no_turns_no_labels(self):
        self.assertEqual(assign_speakers(SEGS, []), SEGS)

    def test_per_speaker_stats(self):
        sp = {x["speaker"]: x for x in text_stats(DIALOG, duration=12.0)["speakers"]}
        self.assertEqual(sp["Speaker 1"]["turns"], 1)
        self.assertEqual(sp["Speaker 1"]["words"], 12)
        self.assertAlmostEqual(sp["Speaker 1"]["seconds"], 5.8)
        self.assertAlmostEqual(sp["Speaker 2"]["share"], 3.5 / 9.3, places=3)
        self.assertEqual(text_stats(SEGS, 12.0)["speakers"], [])

    def test_exports_carry_speakers(self):
        txt = export(DIALOG, "txt").splitlines()
        self.assertEqual(txt, ["Speaker 1: " + SEGS[0]["text"], "And more.", "Speaker 2: " + SEGS[1]["text"]])
        self.assertIn("\nSpeaker 1: And more.\n", export(DIALOG, "srt"))
        self.assertIn("**[00:06] Speaker 2:**", export(DIALOG, "md"))
        self.assertEqual(json.loads(export(DIALOG, "json"))[2]["speaker"], "Speaker 2")

    def test_vtt_uses_voice_tags_and_escapes_markup(self):
        out = export([{"start": 0, "end": 1, "text": "a < b & c", "speaker": "Ann <x>"}], "vtt")
        self.assertIn("<v Ann x>a &lt; b &amp; c", out)

    def test_line_breaks_in_names_title_or_text_cannot_forge_cues_or_headings(self):
        seg = {"start": 0, "end": 1, "text": "a\r\rb", "speaker": "A\n\n99\n00:00:00,000 --> 00:00:01,000\nB"}
        srt = export([seg], "srt")
        self.assertEqual(sum(line.startswith("00:") for line in srt.splitlines()), 1)  # one timing line, one cue
        self.assertNotIn("\r", srt)
        self.assertNotIn("\n\n", srt.rstrip("\n"))
        md = export([seg], "md", title="T\n\n# injected")
        self.assertNotIn("\n# injected", md)
        self.assertNotIn("\r", export([seg], "vtt"))

    def test_translation_keeps_speaker(self):
        self.assertEqual(merge_translation(DIALOG, [{"index": 2, "text": "Hola"}])[2]["speaker"], "Speaker 2")


if __name__ == "__main__":
    unittest.main()
