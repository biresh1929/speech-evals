"""Score every engine's outputs with identical code: Piper, Pocket TTS, Chatterbox, Rumik.

    python scripts/compare_tts_eval.py

* Piper is generated here (CPU subprocess, the real PiperProvider) on the same
  three sentences, with timing.
* Pocket TTS outputs and timings come from scripts/pocket_tts_eval.py.
* Chatterbox from scripts/chatterbox_eval.py (GPU).
* Rumik's speed is the already-measured benchmark (~25x slower than realtime
  on this card), not re-run: 3 sentences would take ~15 min of GPU for a
  number already in BUILD_LOG.

Scores: faster-whisper small WER against the text, and VoiceEncoder similarity
to the author's reference clip (meaningful for clones only; built-in voices
are listed for scale).
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from primer_length_trials import cos, embed, load_voice_encoder  # noqa: E402
from wer_hindi_corrections import edit_counts, normalise  # noqa: E402

from voicenarrator.audio import save_wav  # noqa: E402
from voicenarrator.providers.tts.piper import PiperProvider  # noqa: E402

ROOT = Path(DATA_ROOT + r"/rumik_samples")
OUT = ROOT / "tts_compare"
REFERENCE = ROOT / "chatterbox_eval" / "reference_en.wav"
PIPER_MODEL = DATA_ROOT + r"/code/en_US-lessac-low.onnx"
TEXTS = [
    "You refactored the payment service and moved the retry logic into its own module.",
    "Three tests were added for the rate limiter, and one of them is currently failing.",
    "The authentication middleware now checks the token expiry before touching the database.",
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUT / "piper").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    piper = PiperProvider(model=PIPER_MODEL)
    piper.synthesize("Warm up.")
    for i, text in enumerate(TEXTS, start=1):
        t0 = time.perf_counter()
        buf = piper.synthesize(text)
        secs = time.perf_counter() - t0
        path = save_wav(buf, OUT / "piper" / f"piper_{i}.wav")
        rows.append({"engine": "piper (lessac-low, CPU)", "voice": "built-in", "file": str(path), "text": text,
                     "audio_seconds": round(buf.duration_seconds, 2), "generate_seconds": round(secs, 2),
                     "realtime_factor": round(secs / buf.duration_seconds, 3), "first_chunk_seconds": None})

    pocket = OUT / "pocket" / "pocket_eval.json"
    if pocket.is_file():
        pj = json.loads(pocket.read_text(encoding="utf-8"))
        for label, block in pj["voices"].items():
            for u in block.get("utterances", []):
                rows.append({"engine": "pocket-tts (CPU)", "voice": label, **u})

    cb = ROOT / "chatterbox_eval" / "chatterbox_eval.json"
    if cb.is_file():
        for key, block in json.loads(cb.read_text(encoding="utf-8")).items():
            if not key.endswith("_en"):
                continue
            for u in block["utterances"]:
                rows.append({"engine": f"chatterbox {key.split('_')[0]} (GPU)", "voice": "clone_author",
                             "file": str(ROOT / "chatterbox_eval" / u["file"]), "text": u["text"],
                             "audio_seconds": u["audio_seconds"], "generate_seconds": u["generate_seconds"],
                             "realtime_factor": u["realtime_factor"], "first_chunk_seconds": None})

    ve = load_voice_encoder()
    ref, _ = sf.read(str(REFERENCE))
    e_ref = embed(ve, ref)
    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8")
    for r in rows:
        y, sr = sf.read(r["file"])
        r["spk_sim_to_author"] = round(cos(e_ref, embed(ve, y if sr == 24000 else _resample(y, sr))), 4)
        segs, _ = asr.transcribe(r["file"], language="en", beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        s, d, i = edit_counts(normalise(r["text"]), normalise(hyp))
        r["wer"] = round((s + d + i) / len(normalise(r["text"])), 3)
        r["asr_text"] = hyp

    summary: dict = {}
    for r in rows:
        k = f"{r['engine']} / {r['voice']}"
        summary.setdefault(k, []).append(r)
    table = {}
    for k, rs in summary.items():
        table[k] = {
            "n": len(rs),
            "mean_rtf": round(float(np.mean([x["realtime_factor"] for x in rs])), 3),
            "mean_first_chunk_s": (round(float(np.mean([x["first_chunk_seconds"] for x in rs])), 3)
                                   if rs[0].get("first_chunk_seconds") is not None else None),
            "mean_wer": round(float(np.mean([x["wer"] for x in rs])), 3),
            "mean_spk_sim_to_author": round(float(np.mean([x["spk_sim_to_author"] for x in rs])), 4),
        }
        print(f"{k:<45} {table[k]}", flush=True)
    (OUT / "compare.json").write_text(json.dumps({"rows": rows, "summary": table}, indent=2), encoding="utf-8")
    return 0


def _resample(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=24000)


if __name__ == "__main__":
    sys.exit(main())
