"""Who spoke when: sherpa-onnx speaker diarization (pyannote segmentation + NeMo TitaNet embeddings) on CPU via ONNX.
No PyTorch and no Hugging Face token. The two models (~46 MB) download once from the sherpa-onnx GitHub releases."""
import hashlib
import os
import shutil
import tarfile
import threading
import urllib.request
import wave
from pathlib import Path

REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL = f"{REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = f"{REL}/speaker-recongition-models/nemo_en_titanet_small.onnx"  # sic: "recongition" is the real tag name
MODELS = Path(os.environ.get("VVTR_MODELS", Path.home() / ".cache" / "vvtr-models"))
THRESHOLD = 0.5  # ponytail: clustering distance; lower splits voices more eagerly. Set a speaker count to override.
_lock = threading.Lock()


# SHA-256 of the exact release assets this was tested with; anything else is refused.
SEG_TAR_SHA = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
SEG_SHA = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"  # model.onnx inside the tarball
EMB_SHA = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str) -> Path:
    """Download once; a damaged, partial or tampered file (cached or fresh) is never used."""
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if _sha256(part) != sha256:
            raise RuntimeError(f"checksum mismatch for {url}: refusing to use it")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def models() -> tuple[Path, Path]:
    with _lock:
        seg_dir = MODELS / "sherpa-onnx-pyannote-segmentation-3-0"
        seg = seg_dir / "model.onnx"
        if not (seg.exists() and _sha256(seg) == SEG_SHA):
            tar = _fetch(SEG_URL, MODELS / "segmentation.tar.bz2", SEG_TAR_SHA)
            tmp = MODELS / "segmentation.extracting"
            shutil.rmtree(tmp, ignore_errors=True)
            with tarfile.open(tar) as t:
                t.extractall(tmp, filter="data")
            shutil.rmtree(seg_dir, ignore_errors=True)
            (tmp / seg_dir.name).replace(seg_dir)  # one rename: an interrupted extract never looks installed
            shutil.rmtree(tmp, ignore_errors=True)
            tar.unlink()
        return seg, _fetch(EMB_URL, MODELS / "nemo_en_titanet_small.onnx", EMB_SHA)


def diarize(wav: Path, num_speakers: int = 0, on_progress=lambda p: None) -> list[dict]:
    """Speaker turns for a 16 kHz mono 16-bit wav: [{start, end, speaker: int}], sorted by start."""
    import numpy as np
    import sherpa_onnx

    seg, emb = models()
    threads = max(1, (os.cpu_count() or 2) // 2)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=THRESHOLD))
    if not cfg.validate():
        raise RuntimeError(f"speaker models are unreadable: delete {MODELS} and run again")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    with wave.open(str(wav)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (diarizer.sample_rate, 1, 2):
            raise RuntimeError("speaker labelling needs 16 kHz mono 16-bit audio")
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

    def progress(done, total):
        if total:
            on_progress(min(0.99, done / total))
        return 0  # non-zero would abort

    result = diarizer.process(samples, callback=progress).sort_by_start_time()
    return [{"start": round(r.start, 2), "end": round(r.end, 2), "speaker": r.speaker} for r in result]
