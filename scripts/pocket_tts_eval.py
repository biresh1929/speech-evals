"""Standalone Pocket TTS evaluation (Kyutai). NOT wired into the provider registry.

Runs in its own venv, because pocket-tts needs numpy>=2 and the main venv is
pinned to numpy 1.26 by other dependencies:

    $SPEECH_EVALS_ROOT/venv-pocket/Scripts/python.exe scripts/pocket_tts_eval.py

Set HF_HUB_CACHE (not HF_HOME: that would also move the saved login token the
gated voice-cloning weights need) so the weights land on D:, and
HF_HUB_DISABLE_XET=1 because Xet downloads stall on this network.

Measures on CPU: model load, time to first audio chunk (streaming), total
generation time and realtime factor, for (a) a built-in voice and (b) a clone of
the author's own English reference (the same clip Chatterbox is evaluated on).
Writes WAVs plus timings; intelligibility (WER) and voice similarity are scored
afterwards in the main venv by scripts/compare_tts_eval.py, so all engines are
scored by identical code.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

OUT = Path(DATA_ROOT + r"/rumik_samples/tts_compare/pocket")
REFERENCE = Path(DATA_ROOT + r"/rumik_samples/chatterbox_eval/reference_en.wav")
TEXTS = [
    "You refactored the payment service and moved the retry logic into its own module.",
    "Three tests were added for the rate limiter, and one of them is currently failing.",
    "The authentication middleware now checks the token expiry before touching the database.",
]


def timed_generation(model, state, text: str) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    first = None
    chunks = []
    for chunk in model.generate_audio_stream(state, text):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(chunk.detach().cpu().numpy().reshape(-1))
    total = time.perf_counter() - t0
    return np.concatenate(chunks), first or total, total


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from pocket_tts import TTSModel

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results: dict = {"torch_threads": torch.get_num_threads(), "device": "cpu", "voices": {}}

    t0 = time.perf_counter()
    model = TTSModel.load_model()   # default: english (= english_2026-04)
    results["load_seconds"] = round(time.perf_counter() - t0, 2)
    results["sample_rate"] = model.sample_rate
    print(f"loaded in {results['load_seconds']}s, sr={model.sample_rate}", flush=True)

    # Warm-up so the first measured utterance does not pay one-off init costs.
    warm = model.get_state_for_audio_prompt("alba")
    model.generate_audio(warm, "Warm up.")

    voices = {"builtin_alba": "alba"}
    if REFERENCE.is_file():
        voices["clone_author"] = str(REFERENCE)
    for label, prompt in voices.items():
        t1 = time.perf_counter()
        try:
            state = model.get_state_for_audio_prompt(prompt)
        except Exception as exc:  # noqa: BLE001 - e.g. gated cloning weights unavailable
            results["voices"][label] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{label}: {exc}", flush=True)
            continue
        block = {"prompt_seconds": round(time.perf_counter() - t1, 2), "utterances": []}
        for i, text in enumerate(TEXTS, start=1):
            audio, ttfa, total = timed_generation(model, state, text)
            path = OUT / f"{label}_{i}.wav"
            sf.write(str(path), audio, model.sample_rate)
            dur = len(audio) / model.sample_rate
            block["utterances"].append({"file": str(path), "text": text, "audio_seconds": round(dur, 2),
                                        "first_chunk_seconds": round(ttfa, 3),
                                        "generate_seconds": round(total, 2),
                                        "realtime_factor": round(total / dur, 3)})
            u = block["utterances"][-1]
            print(f"{label}_{i}: {u['audio_seconds']}s, first chunk {u['first_chunk_seconds']}s, "
                  f"total {u['generate_seconds']}s (RTF {u['realtime_factor']})", flush=True)
        results["voices"][label] = block

    (OUT / "pocket_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
