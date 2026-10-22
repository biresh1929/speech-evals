"""Does a continuity primer past Rumik's speaker "handover" window change the seam?

    python scripts/primer_length_trials.py [trials]

Background: rumik.ai/research/decoding-voice reports that speaker identity
transfers from the prompt's speaker tag to the model's own generated audio
"somewhere around 8-32 frames (~3s)". The continuity primer is 8 frames
(0.64s), the start of that window. This tests 0 (no primer, i.e. naive
generation), 8, 16 and 24 frames.

CONTROLS
* Chunk 1 is generated ONCE per trial and shared by every primer length, so
  the audio being continued is identical across conditions.
* Every chunk-2 generation in a trial uses the same torch seed. The prompts
  differ by the primer, so sampling still diverges, but no condition gets a
  luckier RNG stream than another.
* Every chunk 2 is decoded the same way (8 frames of decoder context from
  chunk 1), so decoding cannot differ between conditions; only the primer does.
* The same short texts every time; chunk 2 is short (~5s) to keep 16
  generations affordable on a card that runs Rumik at ~25x realtime.

METRICS (objective, not perceptual)
* Speaker similarity, chunk-1 tail (last 3s) vs chunk-2 head (first 3s):
  cosine of Chatterbox's VoiceEncoder embeddings. Calibrated in the same run
  against a same-voice pair (two independent chunk 1s) and a different-speaker
  pair (Ira vs the user's own recording), so the scale is interpretable.
* Median F0 step across the seam, in semitones (librosa pyin).
* Loudness step across the seam, in dB (RMS of 1s either side).
* Content: does chunk 2 say its text? Word error rate of faster-whisper small
  on chunk 2 alone. A longer primer is more context the model might "continue"
  instead of starting the new sentence.

Output: $SPEECH_EVALS_ROOT/rumik_samples/primer_trials/{wav/, primer_trials.json}
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
from longform_trials import LOCAL_PATH  # noqa: E402
from wer_hindi_corrections import edit_counts, normalise  # noqa: E402

from voicenarrator.providers.tts.rumik import RumikProvider  # noqa: E402
from voicenarrator.providers.tts.rumik_longform import tail_primer_tokens  # noqa: E402

OUT = Path(DATA_ROOT + r"/rumik_samples/primer_trials")
VE_PATH = Path(DATA_ROOT + r"/code/models/hf_cache/models--ResembleAI--chatterbox/snapshots/"
               r"5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/ve.safetensors")
USER_VOICE = Path(DATA_ROOT + r"/voice_dataset/wav/new-hindi-recording.wav")
SR = 24000
PRIMERS = [0, 8, 16, 24]
DEFAULT_TRIALS = 4

TEXT_1 = ("The authentication module has been rewritten this afternoon. "
          "Passwords are now hashed with PBKDF2 instead of being stored in plain text.")
TEXT_2 = "There is a new rate limiter in front of the API. It allows one hundred requests per minute."


def load_voice_encoder():
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(str(VE_PATH)))
    return ve.eval()   # CPU: tiny LSTM, keeps the GPU for Rumik


def embed(ve, wav: np.ndarray) -> np.ndarray:
    e = ve.embeds_from_wavs([wav.astype(np.float32)], sample_rate=SR, as_spk=True)
    return e / np.linalg.norm(e)


def cos(a, b) -> float:
    return float(np.dot(a.ravel(), b.ravel()))


def median_f0(y: np.ndarray) -> float | None:
    import librosa

    f0, voiced, _ = librosa.pyin(y, fmin=120, fmax=450, sr=SR, frame_length=1024, hop_length=240)
    sel = voiced & ~np.isnan(f0)
    return float(np.median(f0[sel])) if sel.sum() >= 5 else None


def db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Parsed here, not at module level: this module is imported by
    # chatterbox_eval.py for its helper functions, and module-level sys.argv
    # parsing once broke that by reading the IMPORTER's argv (--multilingual)
    # as a trial count.
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRIALS
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    import torch

    provider = RumikProvider(model_path=LOCAL_PATH, vram_mode="codec_cpu", speaker="Ira",
                             description="clear, moderate pace, neutral tone")
    provider._load()
    cfg = provider._model.config
    first_unit, nq = int(cfg.first_unit_id), int(cfg.num_quantizers)
    ve = load_voice_encoder()

    results: dict = {"primers": PRIMERS, "trials": [], "text_1": TEXT_1, "text_2": TEXT_2}
    chunk1_wavs = []
    for t in range(1, trials + 1):
        torch.manual_seed(t)
        t0 = time.perf_counter()
        codes1, _ = provider._generate_codes(TEXT_1)
        wav1 = provider._decode(codes1)
        chunk1_wavs.append(wav1)
        sf.write(str(OUT / "wav" / f"t{t}_chunk1.wav"), wav1, SR)
        arr1 = codes1[0].cpu().numpy()
        ctx = codes1[:, :, -provider.decoder_context_frames:]
        print(f"trial {t}: chunk1 {len(wav1)/SR:.1f}s in {time.perf_counter()-t0:.0f}s", flush=True)

        tail = wav1[-3 * SR:]
        e_tail = embed(ve, tail)
        f0_tail = median_f0(wav1[-2 * SR:])
        trial = {"trial": t, "chunk1_seconds": round(len(wav1) / SR, 2), "conditions": {}}
        for p in PRIMERS:
            torch.manual_seed(1000 + t)
            primer = tail_primer_tokens(arr1, first_unit, nq, p) if p else None
            t1 = time.perf_counter()
            codes2, n_tok = provider._generate_codes(TEXT_2, primer=primer)
            wav2 = provider._decode_with_context(codes2, ctx)
            name = f"t{t}_primer{p:02d}_chunk2.wav"
            sf.write(str(OUT / "wav" / name), wav2, SR)
            f0_head = median_f0(wav2[: 2 * SR])
            trial["conditions"][str(p)] = {
                "file": name,
                "chunk2_seconds": round(len(wav2) / SR, 2),
                "generation_seconds": round(time.perf_counter() - t1, 1),
                "spk_sim_tail_head": round(cos(e_tail, embed(ve, wav2[: 3 * SR])), 4),
                "spk_sim_chunk1_chunk2": round(cos(embed(ve, wav1), embed(ve, wav2)), 4),
                "f0_step_semitones": (round(12 * np.log2(f0_head / f0_tail), 2)
                                      if f0_head and f0_tail else None),
                "loudness_step_db": round(db(wav2[:SR]) - db(wav1[-SR:]), 2),
            }
            c = trial["conditions"][str(p)]
            print(f"   primer {p:>2}: {c['chunk2_seconds']}s  sim_seam={c['spk_sim_tail_head']}  "
                  f"f0_step={c['f0_step_semitones']}  loud_step={c['loudness_step_db']}dB", flush=True)
        results["trials"].append(trial)
        (OUT / "primer_trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- calibration of the similarity scale ---
    same = [cos(embed(ve, chunk1_wavs[i]), embed(ve, chunk1_wavs[j]))
            for i in range(len(chunk1_wavs)) for j in range(i + 1, len(chunk1_wavs))]
    user, _ = sf.read(str(USER_VOICE), start=60 * 24000, frames=20 * 24000)
    e_user = embed(ve, user)
    within = [cos(embed(ve, w[: 3 * SR]), embed(ve, w[-3 * SR:])) for w in chunk1_wavs]
    diff = [cos(embed(ve, w), e_user) for w in chunk1_wavs]
    results["calibration"] = {
        "same_voice_independent_generations": [round(x, 4) for x in same],
        "ira_vs_user_recording": [round(x, 4) for x in diff],
        "same_file_first3s_vs_last3s_no_seam": [round(x, 4) for x in within],
    }

    # --- free the GPU, then check content with Whisper ---
    del provider
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8")
    for trial in results["trials"]:
        for c in trial["conditions"].values():
            segs, _ = asr.transcribe(str(OUT / "wav" / c["file"]), language="en", beam_size=5)
            hyp = " ".join(s.text for s in segs).strip()
            ref_w, hyp_w = normalise(TEXT_2), normalise(hyp)
            s, d, i = edit_counts(ref_w, hyp_w)
            c["asr_text"] = hyp
            c["wer"] = round((s + d + i) / len(ref_w), 3)

    summary = {}
    for p in PRIMERS:
        cs = [tr["conditions"][str(p)] for tr in results["trials"]]

        def stat(key, cs=cs):
            v = [c[key] for c in cs if c[key] is not None]
            return {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                    "values": v}

        summary[str(p)] = {k: stat(k) for k in
                           ("spk_sim_tail_head", "spk_sim_chunk1_chunk2", "f0_step_semitones",
                            "loudness_step_db", "wer", "chunk2_seconds")}
    results["summary"] = summary
    (OUT / "primer_trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nCALIBRATION", json.dumps(results["calibration"]))
    for p, s in summary.items():
        print(f"primer {p:>2}: " + "  ".join(f"{k}={v['mean']}±{v['sd']}" for k, v in s.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
