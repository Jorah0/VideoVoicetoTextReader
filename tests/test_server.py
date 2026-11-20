import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="vvtr-test-")
os.environ["VVTR_OUTPUT"] = TMP  # must precede the import: server reads it at import time

from fastapi.testclient import TestClient  # noqa: E402

from app import server  # noqa: E402

JOB_ID = "abcdef012345"
SEGS = [{"start": 0.0, "end": 1.5, "text": "Hello there."}, {"start": 2.0, "end": 3.0, "text": "Bye."}]


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(server.app)
        d = Path(TMP) / JOB_ID
        d.mkdir(exist_ok=True)
        (d / "job.json").write_text(json.dumps({
            "id": JOB_ID, "dir": str(d), "name": "fake.mp4", "created": 1.0, "status": "done", "error": None,
            "stages": {}, "result": {"transcript": {"segments": SEGS}}}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_rejects_non_http_url(self):
        r = self.c.post("/api/jobs", data={"url": "file:///etc/passwd"})
        self.assertEqual(r.status_code, 400)

    def test_rejects_missing_source(self):
        self.assertEqual(self.c.post("/api/jobs", data={"url": ""}).status_code, 400)

    def test_rejects_unknown_model_and_bad_language(self):
        self.assertEqual(self.c.post("/api/jobs", data={"url": "https://x.test/v", "model": "gpt"}).status_code, 400)
        self.assertEqual(self.c.post("/api/jobs", data={"url": "https://x.test/v", "language": "../"}).status_code, 400)

    def test_job_id_and_frame_name_are_validated(self):
        self.assertEqual(self.c.get("/api/jobs/..%2F..%2Fetc").status_code, 404)
        self.assertEqual(self.c.get(f"/api/jobs/{JOB_ID}/frames/..%2Fjob.json").status_code, 404)

    def test_get_and_list(self):
        self.assertEqual(self.c.get(f"/api/jobs/{JOB_ID}").json()["status"], "done")
        self.assertNotIn("dir", self.c.get(f"/api/jobs/{JOB_ID}").json())
        self.assertIn(JOB_ID, [j["id"] for j in self.c.get("/api/jobs").json()])

    def test_export_download_and_save_keeps_raw(self):
        edited = [dict(SEGS[0], text="Hello, there!"), SEGS[1]]
        r = self.c.post(f"/api/jobs/{JOB_ID}/export", json={"fmt": "srt", "segments": edited, "save": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertIn("Hello, there!", r.text)
        job = self.c.get(f"/api/jobs/{JOB_ID}").json()
        self.assertEqual(job["result"]["transcript_edited"][0]["text"], "Hello, there!")
        self.assertEqual(job["result"]["transcript"]["segments"][0]["text"], "Hello there.")  # raw untouched
        self.assertTrue((Path(TMP) / JOB_ID / "transcript_edited.srt").exists())

    def test_export_rejects_bad_format(self):
        r = self.c.post(f"/api/jobs/{JOB_ID}/export", json={"fmt": "exe", "segments": SEGS})
        self.assertEqual(r.status_code, 400)

    def test_cross_site_post_rejected(self):
        r = self.c.post("/api/jobs", data={"url": "https://x.test/v"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.c.delete(f"/api/jobs/{JOB_ID}", headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_dns_rebinding_host_rejected(self):
        self.assertEqual(self.c.get(f"/api/jobs/{JOB_ID}", headers={"Host": "evil.example:8765"}).status_code, 400)

    def test_failed_job_marks_running_stages_failed_and_persists(self):
        d = Path(TMP) / "0123456789ab"
        d.mkdir(exist_ok=True)
        job = {"id": d.name, "dir": str(d), "name": "x", "created": 2.0, "status": "queued", "error": None,
               "stages": {}, "result": {}}
        saved = server.pipeline.process

        def boom(job, update, *a):
            update("acquire", "done", 1)
            update("probe", "running", 0)
            raise RuntimeError("ffprobe could not read the file")
        server.pipeline.process = boom
        try:
            server.run_job(job, None, None, {})
        finally:
            server.pipeline.process = saved
        on_disk = json.loads((d / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "failed")
        self.assertEqual(on_disk["stages"]["probe"]["status"], "failed")
        self.assertEqual(on_disk["stages"]["acquire"]["status"], "done")

    def test_interrupted_jobs_marked_on_startup(self):
        d = Path(TMP) / "fedcba987654"
        d.mkdir(exist_ok=True)
        (d / "job.json").write_text(json.dumps({"id": d.name, "dir": str(d), "name": "y", "created": 3.0,
                                                "status": "running", "stages": {}, "result": {}}), encoding="utf-8")
        server.mark_interrupted()
        self.assertEqual(self.c.get(f"/api/jobs/{d.name}").json()["status"], "interrupted")

    def test_index_served(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Voice-to-Text Reader", r.text)


if __name__ == "__main__":
    unittest.main()
