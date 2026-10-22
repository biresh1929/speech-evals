"""Controlled continuity-vs-naive comparison.

WHY MORE THAN ONE SAMPLE EACH
Generation samples with `temperature=0.8, top_k=30`. Two utterances of the same
text differ audibly from sampling alone, so one clip per mode cannot separate a
continuity effect from ordinary randomness. The first comparison did exactly
that and produced a result nobody could trust.

This runs N trials per mode, interleaved (naive, continuity, naive, ...) so any
drift in machine state affects both arms equally rather than loading onto
whichever ran second.

SEAM ISOLATION
Judging a 30-second utterance asks the listener to hold a lot in memory. The
chunk join is the only place the two modes can differ by construction, so each
trial also gets a short clip centred on it, using the exact seam offsets the
provider now reports (no guessing at a percentage through the file).

Output: $SPEECH_EVALS_ROOT/rumik_samples/trials/
  full/   naive_1.wav, continuity_1.wav, ...
  seams/  naive_1_seam.wav, ...   (~3s each, the join at the midpoint)
  trials.json
  LISTEN.md   blind-ordered listening sheet
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voicenarrator.audio import save_wav  # noqa: E402
from voicenarrator.providers.tts.rumik import RumikProvider  # noqa: E402

LOCAL_PATH = (
    DATA_ROOT + r"/code/models/hf_cache/models--rumik-ai--rumik-oss-1/"
    r"snapshots/0ed3c98684e14350c910129c0efa179841c41ad2"
)
OUT = Path(DATA_ROOT + r"/rumik_samples/trials")
DEFAULT_TRIALS_PER_MODE = 4
SEAM_WINDOW_S = 3.0

# Long enough to force a second chunk (so a seam exists), short enough that
# 2 x N trials finish in hours rather than overnight.
TEXT = (
    "The authentication module has been rewritten this afternoon. "
    "Passwords are now hashed with PBKDF2 instead of being stored in plain text. "
    "An account locks itself after five failed attempts, which closes the brute "
    "force hole we found last week. "
    "There is a new rate limiter in front of the API as well. "
    "It allows one hundred requests per minute."
)


def cut_seam(wav_path: Path, seam_time: float, dest: Path, window: float = SEAM_WINDOW_S) -> dict:
    """Cut `window` seconds centred on the seam."""
    data, sr = sf.read(str(wav_path))
    centre = int(seam_time * sr)
    half = int(window * sr / 2)
    start, end = max(0, centre - half), min(len(data), centre + half)
    clip = data[start:end]
    sf.write(str(dest), clip, sr)
    return {
        "seam_time_s": round(seam_time, 3),
        "clip_seconds": round(len(clip) / sr, 3),
        "seam_at_clip_offset_s": round((centre - start) / sr, 3),
    }


def main() -> int:
    # Parsed here, not at module level: this module is imported by
    # primer_length_trials.py (for LOCAL_PATH/TEXT), which chatterbox_eval.py
    # in turn imports -- module-level sys.argv parsing broke that chain by
    # reading an IMPORTER's own CLI flags (e.g. --multilingual) as a trial count.
    trials_per_mode = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRIALS_PER_MODE
    (OUT / "full").mkdir(parents=True, exist_ok=True)
    (OUT / "seams").mkdir(parents=True, exist_ok=True)

    provider = RumikProvider(
        model_path=LOCAL_PATH,
        vram_mode="codec_cpu",
        speaker="Ira",
        description="clear, moderate pace, neutral tone",
    )
    print("loading model once, reused for every trial...", flush=True)
    t0 = time.perf_counter()
    provider._load()
    print(f"loaded in {time.perf_counter() - t0:.1f}s\n", flush=True)

    # Interleave so machine-state drift hits both arms equally.
    schedule = []
    for i in range(1, trials_per_mode + 1):
        schedule.append(("naive", i))
        schedule.append(("continuity", i))

    results = []
    for mode, n in schedule:
        label = f"{mode}_{n}"
        print(f"--- {label} ---", flush=True)
        t1 = time.perf_counter()
        buf = provider.synthesize(TEXT, continuity=mode)
        elapsed = time.perf_counter() - t1

        full = OUT / "full" / f"{label}.wav"
        save_wav(buf, full)

        rec = {
            "label": label,
            "mode": mode,
            "trial": n,
            "full_file": str(full),
            "audio_seconds": round(buf.duration_seconds, 2),
            "generation_seconds": round(elapsed, 1),
            "realtime_factor": round(elapsed / buf.duration_seconds, 1)
            if buf.duration_seconds else None,
            "chunk_seconds": [round(c, 2) for c in provider.last_chunk_seconds],
            "seam_times": [round(s, 3) for s in provider.last_seam_times],
        }

        if provider.last_seam_times:
            seam_file = OUT / "seams" / f"{label}_seam.wav"
            rec["seam"] = cut_seam(full, provider.last_seam_times[0], seam_file)
            rec["seam_file"] = str(seam_file)
            print(f"    seam at {provider.last_seam_times[0]:.2f}s -> {seam_file.name}", flush=True)
        else:
            print("    WARNING: only one chunk, no seam to isolate", flush=True)

        # Objective-ish descriptors. These do not measure perceived quality --
        # recorded so a listening result can be checked against them, not
        # substituted for one.
        data, sr = sf.read(str(full))
        rec["rms"] = round(float(np.sqrt(np.mean(data**2))), 4)
        rec["peak"] = round(float(np.max(np.abs(data))), 4)
        if provider.last_seam_times:
            c = int(provider.last_seam_times[0] * sr)
            w = int(0.25 * sr)
            before, after = data[max(0, c - w):c], data[c:c + w]
            if len(before) and len(after):
                rb = float(np.sqrt(np.mean(before**2)))
                ra = float(np.sqrt(np.mean(after**2)))
                rec["seam_rms_before"] = round(rb, 4)
                rec["seam_rms_after"] = round(ra, 4)
                rec["seam_rms_ratio"] = round(ra / rb, 3) if rb else None

        results.append(rec)
        print(
            f"    {rec['audio_seconds']}s audio in {rec['generation_seconds']}s "
            f"({rec['realtime_factor']}x), chunks={rec['chunk_seconds']}",
            flush=True,
        )
        (OUT / "trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    write_listening_sheet(results)

    print("\n=== summary ===")
    for mode in ("naive", "continuity"):
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        durs = [r["audio_seconds"] for r in rows]
        rmss = [r["rms"] for r in rows]
        ratios = [r["seam_rms_ratio"] for r in rows if r.get("seam_rms_ratio")]
        print(
            f"{mode:11} n={len(rows)}  duration {np.mean(durs):.1f}s "
            f"(sd {np.std(durs):.1f})  rms {np.mean(rmss):.4f} "
            f"(sd {np.std(rmss):.4f})"
            + (f"  seam_rms_ratio {np.mean(ratios):.2f} (sd {np.std(ratios):.2f})"
               if ratios else "")
        )
    print(f"\nwrote {OUT}")
    print("Listen via LISTEN.md -- the order there is randomised and the key is at the bottom.")
    return 0


def write_listening_sheet(results: list[dict]) -> None:
    """Blind-ordered sheet: labels hidden until the key at the end.

    Filenames say which mode they are, which is fine for debugging and useless
    for judging. Knowing which one is 'the clever one' is enough to bias a
    listener -- including the person who built it.
    """
    seam_rows = [r for r in results if r.get("seam_file")]
    order = list(range(len(seam_rows)))
    random.seed(12345)  # fixed so the sheet is reproducible
    random.shuffle(order)

    lines = [
        "# Blind listening sheet",
        "",
        f"{len(seam_rows)} seam clips, {SEAM_WINDOW_S:.0f}s each, the chunk join at the midpoint.",
        "Presented in randomised order. **The key is at the very bottom -- do not read it first.**",
        "",
        "For each clip, note whether you hear anything odd at the halfway point:",
        "a jump in pitch, a change in pace or loudness, a click, or a voice that",
        "sounds like a different take. Then score 1-5 for how natural the join is",
        "(5 = cannot tell there is one).",
        "",
        "| # | clip | smooth 1-5 | what you noticed |",
        "|---|---|---|---|",
    ]
    for position, idx in enumerate(order, start=1):
        name = Path(seam_rows[idx]["seam_file"]).name
        lines.append(f"| {position} | `seams/{name}` |  |  |")

    lines += [
        "",
        "Also worth doing: pick any two **full** clips from `full/` at random and",
        "ask which you would rather listen to for a minute. That is closer to how",
        "the tool is actually used than judging a 3-second seam.",
        "",
        "---",
        "",
        "<details><summary>KEY (read after scoring)</summary>",
        "",
        "| # | clip | mode |",
        "|---|---|---|",
    ]
    for position, idx in enumerate(order, start=1):
        r = seam_rows[idx]
        lines.append(f"| {position} | `{Path(r['seam_file']).name}` | **{r['mode']}** |")
    lines += ["", "</details>", ""]

    (OUT / "LISTEN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
