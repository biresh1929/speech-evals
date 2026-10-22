"""Measure the Chatterbox provider on this machine: cloning, speed, VRAM, watermark.

    python scripts/chatterbox_eval.py [--multilingual]

* Reference voices come from the author's OWN recordings (the voice dataset):
  the most speech-dense unflagged English segment of 10-16s, and for
  --multilingual the most speech-dense Hindi segment (at least 8s) of the new
  Hindi recording. Consent is not a question here: it is the author's voice.
* Speed: model load, per-utterance generation time and realtime factor, and
  peak VRAM, through the real ChatterboxProvider (not the raw library).
* Intelligibility: faster-whisper WER of each generation against its text
  (small for English, medium for Hindi).
* Voice match: cosine similarity of VoiceEncoder embeddings, clone vs
  reference. BIASED UPWARD: Chatterbox conditions on this same encoder's
  embedding, so the number flatters it. A same-machine contrast (reference vs
  Rumik's Ira) is reported next to it so the scale is readable.
* Watermark: Perth's detector on each generation and on the untouched
  reference recording (expect ~1.0 vs ~0.0).

Writes $SPEECH_EVALS_ROOT/rumik_samples/chatterbox_eval/{*.wav, chatterbox_eval.json}.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from primer_length_trials import cos, embed, load_voice_encoder  # noqa: E402
from wer_hindi_corrections import edit_counts, normalise  # noqa: E402

from voicenarrator.providers.tts.chatterbox import ChatterboxProvider  # noqa: E402

MANIFEST = Path(DATA_ROOT + r"/voice_dataset/manifest.jsonl")
SNAP = Path(DATA_ROOT + r"/code/models/hf_cache/models--ResembleAI--chatterbox/snapshots/"
            r"5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
OUT = Path(DATA_ROOT + r"/rumik_samples/chatterbox_eval")
IRA = Path(DATA_ROOT + r"/rumik_samples/trials/full/naive_1.wav")

EN_TEXTS = [
    "You refactored the payment service and moved the retry logic into its own module.",
    "Three tests were added for the rate limiter, and one of them is currently failing.",
    "The authentication middleware now checks the token expiry before touching the database.",
]
HI_TEXTS = [
    "आपने पेमेंट सर्विस में बदलाव किया है और रीट्राई लॉजिक को अलग मॉड्यूल में डाल दिया है।",
    "रेट लिमिटर के लिए तीन नए टेस्ट जोड़े गए हैं, और उनमें से एक अभी फेल हो रहा है।",
]


def pick_reference(language: str, source: str | None, lo: float, hi: float) -> dict:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    cands = [r for r in rows if r["language"] == language and lo <= r["duration"] <= hi
             and not r.get("flags") and (source is None or r["source"] == source)]
    return max(cands, key=lambda r: len(r["text"]) / r["duration"])


def cut(row: dict, dest: Path) -> Path:
    y, sr = sf.read(row["audio_path"], start=int(row["start"] * 24000), stop=int(row["end"] * 24000))
    sf.write(str(dest), y, sr)
    return dest


def run_variant(variant: str, ref_row: dict, texts: list[str], language: str, ve, results: dict) -> None:
    import perth
    import torch

    ref = cut(ref_row, OUT / f"reference_{language}.wav")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opts = {"voice": str(ref), "model_dir": str(SNAP), "variant": variant, "device": "cuda"}
    if variant == "multilingual":
        opts["language"] = language
    tts = ChatterboxProvider(**opts)

    ref_audio, _ = sf.read(str(ref))
    e_ref = embed(ve, ref_audio)
    wm = perth.PerthImplicitWatermarker()
    block = {"reference": {"source": ref_row["source"], "start": ref_row["start"],
                           "seconds": round(ref_row["duration"], 2), "text": ref_row["text"]},
             "utterances": []}
    for i, text in enumerate(texts, start=1):
        buf = tts.synthesize(text)
        wav = np.frombuffer(buf.pcm, dtype="<i2").astype(np.float32) / 32767.0
        path = OUT / f"{variant}_{language}_{i}.wav"
        sf.write(str(path), wav, buf.sample_rate)
        block["utterances"].append({
            "file": path.name, "text": text,
            "audio_seconds": round(buf.duration_seconds, 2),
            "generate_seconds": round(tts.last_timings["generate_seconds"], 2),
            "realtime_factor": round(tts.last_timings["generate_seconds"] / buf.duration_seconds, 3),
            "spk_sim_to_reference": round(cos(e_ref, embed(ve, wav)), 4),
            "watermark_score": float(wm.get_watermark(wav, sample_rate=buf.sample_rate)),
        })
        u = block["utterances"][-1]
        print(f"  {path.name}: {u['audio_seconds']}s in {u['generate_seconds']}s "
              f"(RTF {u['realtime_factor']}), sim {u['spk_sim_to_reference']}, wm {u['watermark_score']}", flush=True)
    block["load_seconds"] = round(tts.last_timings.get("load_seconds", 0), 1)
    block["voice_prepare_seconds"] = round(tts.last_timings.get("voice_prepare_seconds", 0), 2)
    block["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    block["watermark_score_on_untouched_reference"] = float(wm.get_watermark(ref_audio, sample_rate=24000))
    if IRA.is_file():
        ira, _ = sf.read(str(IRA))
        block["contrast_sim_reference_vs_rumik_ira"] = round(cos(e_ref, embed(ve, ira)), 4)
    results[f"{variant}_{language}"] = block
    del tts
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def asr_pass(results: dict) -> None:
    from faster_whisper import WhisperModel

    for key, lang, model_name in (("english_en", "en", "small"), ("multilingual_hi", "hi", "medium"),
                                  ("multilingual_en", "en", "small")):
        if key not in results:
            continue
        kwargs = {"download_root": DATA_ROOT + r"/code/models/whisper"} if model_name == "medium" else {}
        asr = WhisperModel(model_name, device="cpu", compute_type="int8", **kwargs)
        for u in results[key]["utterances"]:
            segs, _ = asr.transcribe(str(OUT / u["file"]), language=lang, beam_size=5)
            hyp = " ".join(s.text for s in segs).strip()
            ref_w, hyp_w = normalise(u["text"]), normalise(hyp)
            s, d, i = edit_counts(ref_w, hyp_w)
            u["asr_text"], u["wer"] = hyp, round((s + d + i) / max(len(ref_w), 1), 3)
            print(f"  {u['file']}: WER {u['wer']}  | {hyp}", flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--multilingual", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ve = load_voice_encoder()
    results: dict = {}

    en_ref = pick_reference("en", None, 10, 16)
    print(f"English reference: {en_ref['source']} @ {en_ref['start']}s ({en_ref['duration']:.1f}s)", flush=True)
    run_variant("english", en_ref, EN_TEXTS, "en", ve, results)
    if args.multilingual:
        hi_ref = pick_reference("hi", "new-hindi-recording.mp4", 8, 20)
        print(f"Hindi reference: @ {hi_ref['start']}s ({hi_ref['duration']:.1f}s)", flush=True)
        run_variant("multilingual", hi_ref, HI_TEXTS, "hi", ve, results)
    (OUT / "chatterbox_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ASR check (CPU, GPU released)", flush=True)
    asr_pass(results)
    (OUT / "chatterbox_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
