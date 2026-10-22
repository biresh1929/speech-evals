"""Verify candidate words are ACTUALLY mispronounced by Rumik before tracing anything.

    python scripts/check_mispronunciations.py

For each candidate word, generate a short sentence containing it (English,
Ira, naive mode -- continuity is irrelevant here, this is single-chunk), then
transcribe with faster-whisper and check whether the word survives. Also
saves the audio so a human can listen, since ASR can itself mis-hear a
correctly-spoken rare word.

Only words that fail here get used in Phase 1's causal tracing -- building
that on an assumed failure case the model does not actually exhibit would be
exactly the mistake the prompt warned against.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longform_trials import LOCAL_PATH  # noqa: E402
from wer_hindi_corrections import edit_counts, normalise  # noqa: E402

from voicenarrator.providers.tts.rumik import RumikProvider  # noqa: E402

OUT = Path(DATA_ROOT + r"/rumik_samples/mispronunciation_check")

# One candidate per sentence, word placed mid-sentence (not first/last, where
# generation is noisiest) and repeated once so a single bad frame doesn't
# decide the whole check.
CANDIDATES = {
    "PostgreSQL": "The team migrated the database from MySQL to PostgreSQL last week, "
                  "and PostgreSQL has been stable since.",
    "kubectl": "You can check the pod status by running kubectl get pods, "
              "and kubectl logs shows the output.",
    "async": "The function was refactored to be async, so every async call "
            "now runs without blocking the main thread.",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    from voicenarrator.audio import save_wav

    provider = RumikProvider(model_path=LOCAL_PATH, vram_mode="codec_cpu", speaker="Ira",
                             description="clear, moderate pace, neutral tone")
    provider._load()

    results = {}
    for word, text in CANDIDATES.items():
        t0 = time.perf_counter()
        buf = provider.synthesize(text, continuity="naive")
        path = save_wav(buf, OUT / f"{word}.wav")
        print(f"{word}: generated {buf.duration_seconds:.1f}s in {time.perf_counter()-t0:.0f}s", flush=True)
        results[word] = {"text": text, "file": str(path), "audio_seconds": round(buf.duration_seconds, 2)}

    del provider
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8")
    for word, r in results.items():
        segs, _ = asr.transcribe(r["file"], language="en", beam_size=5, word_timestamps=True)
        segs = list(segs)
        hyp = " ".join(s.text for s in segs).strip()
        r["asr_text"] = hyp
        ref_w, hyp_w = normalise(r["text"]), normalise(hyp)
        s, d, i = edit_counts(ref_w, hyp_w)
        r["wer"] = round((s + d + i) / len(ref_w), 3)
        # Did the target word (case-insensitive, either occurrence) actually appear
        # in the transcription, verbatim or as a a clear near-miss?
        target = word.lower()
        words_out = [w.strip(".,").lower() for w in hyp.split()]
        r["target_word_recognized_verbatim"] = target in words_out
        r["hyp_words"] = words_out
        print(f"{word}: WER={r['wer']}  verbatim_recognized={r['target_word_recognized_verbatim']}", flush=True)
        print(f"   hyp: {hyp}", flush=True)

    (OUT / "check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== which words are confirmed mispronounced (word not recognized verbatim by ASR) ===")
    for word, r in results.items():
        verdict = "MISPRONOUNCED (candidate for tracing)" if not r["target_word_recognized_verbatim"] \
            else "recognized correctly -- NOT a tracing candidate without a human listen contradicting ASR"
        print(f"  {word}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
