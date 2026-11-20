"""The project's one verify command.
    python scripts/verify.py          compile + unit/API tests (seconds)
    python scripts/verify.py --full   + end-to-end: synthesized speech video -> real Whisper (tiny) -> API -> export
Writes .verify/last.json. A skipped step is listed as skipped, never as passed."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "The quick brown fox jumps over the lazy dog. Speech recognition turns spoken words into written text."


class SkipStep(Exception):
    pass


def step(name, fn, results):
    t = time.time()
    try:
        fn()
        results.append((name, "PASS", time.time() - t, ""))
    except SkipStep as e:
        results.append((name, "SKIP", time.time() - t, str(e)))
    except Exception as e:  # report every failure, keep going
        results.append((name, "FAIL", time.time() - t, f"{type(e).__name__}: {e}"))


def sh(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr)[-2000:])
    return p


def compile_all():
    sh([sys.executable, "-m", "compileall", "-q", "app", "tests", "scripts"])


def unit_tests():
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ROOT,
                       capture_output=True, text=True)
    print("\n".join(p.stderr.strip().splitlines()[-3:]))
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])


def e2e():
    if sys.platform != "win32":
        raise SkipStep("speech synthesis for the fixture uses Windows SAPI")
    tmp = Path(tempfile.mkdtemp(prefix="vvtr-e2e-"))
    wav, mp4 = tmp / "speech.wav", tmp / "clip.mp4"
    ps = (f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.Rate=-1; $s.SetOutputToWaveFile('{wav}'); $s.Speak('{SENTENCE}'); $s.Dispose()")
    sh(["powershell", "-NoProfile", "-Command", ps])
    # two solid-colour shots (a scene cut at 4s) under the synthesized speech
    sh(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=640x360:d=4:r=25",
        "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=6:r=25", "-i", str(wav),
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "10", str(mp4)])

    os.environ["VVTR_OUTPUT"] = str(tmp / "out")
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from app import server
    c = TestClient(server.app)
    with mp4.open("rb") as f:
        r = c.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")},
                   data={"model": "tiny", "use_ai": "false", "ocr": "false", "translate_to": ""})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", f"job ended {job['status']}: {job.get('error')}"
    res = job["result"]
    text = " ".join(s["text"] for s in res["transcript"]["segments"]).lower()
    print("    transcript:", text)
    for word in ("quick", "brown", "fox", "speech"):
        assert word in text, f"'{word}' missing from transcript"
    assert res["transcript"]["language"] == "en", res["transcript"]["language"]
    import ctranslate2
    if ctranslate2.get_cuda_device_count():  # a GPU is present: the CPU fallback must not be hiding broken CUDA libs
        assert res["transcript"]["device"] == "cuda", "GPU present but transcription ran on CPU (CUDA libs missing?)"
    assert res["visual"]["scene_count"] >= 2, res["visual"]
    assert res["visual"]["keyframes"], "no keyframes extracted"
    assert res["stats"]["word_count"] >= 10, res["stats"]
    assert res["audio"]["integrated_lufs"] is not None, res["audio"]
    assert all(job["stages"][s]["status"] == "done" for s in ("transcribe", "visual_scan", "audio_scan")), job["stages"]
    assert c.get(f"/api/jobs/{job_id}/media").status_code == 200
    assert c.get(f"/api/jobs/{job_id}/frames/f_000.jpg").status_code == 200
    out = Path(os.environ["VVTR_OUTPUT"]) / job_id
    assert "fox" in (out / "transcript.txt").read_text(encoding="utf-8").lower()
    srt = c.post(f"/api/jobs/{job_id}/export", json={"fmt": "srt", "segments": res["transcript"]["segments"]})
    assert srt.status_code == 200 and "-->" in srt.text
    assert c.delete(f"/api/jobs/{job_id}").status_code == 200 and not out.exists()


def main():
    full = "--full" in sys.argv
    results = []
    step("compile", compile_all, results)
    step("unit + API tests", unit_tests, results)
    if full:
        step("end-to-end (Whisper tiny, synthesized speech)", e2e, results)
    else:
        results.append(("end-to-end", "SKIP", 0, "run with --full"))
    print()
    for name, status, secs, msg in results:
        print(f"  {status:4}  {name}  ({secs:.1f}s)" + (f"\n        {msg}" if msg else ""))
    passed = all(s != "FAIL" for _, s, _, _ in results)
    skipped = [n for n, s, _, _ in results if s == "SKIP"]
    (ROOT / ".verify").mkdir(exist_ok=True)
    (ROOT / ".verify" / "last.json").write_text(json.dumps(
        {"passed": passed, "full": full, "skipped": skipped, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    print("\nVERIFY", "PASSED" if passed else "FAILED", f"(skipped: {', '.join(skipped)})" if skipped else "")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
