"""Audio output and WAV helpers.

Playback deliberately has no hard dependency: we try a few backends and fall
back to writing a file. A narration tool that crashes because a sound library
is missing is worse than one that quietly saves a .wav.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from .providers.base import AudioBuffer


def to_float32(buf: AudioBuffer):
    """Raw 16-bit PCM -> mono float32 samples in [-1, 1]. Local numpy import,
    same as this module's other numpy-touching functions, so importing
    `voicenarrator.audio` never requires numpy for callers who only save/read
    WAV files."""
    import numpy as np

    if not buf.pcm:
        return np.zeros(0, dtype=np.float32)
    return (np.frombuffer(buf.pcm, dtype="<i2").astype(np.float32) / 32768.0).copy()


def from_float32(samples, sample_rate: int) -> AudioBuffer:
    """Inverse of `to_float32`, clamping to avoid int16 wraparound on a
    sample that ended up (even slightly) outside [-1, 1]."""
    import numpy as np

    clamped = np.clip(samples, -1.0, 1.0)
    pcm = (clamped * 32767.0).astype("<i2").tobytes()
    return AudioBuffer(pcm=pcm, sample_rate=sample_rate)


def to_wav_bytes(buf: AudioBuffer) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(buf.sample_rate)
        w.writeframes(buf.pcm)
    return bio.getvalue()


def save_wav(buf: AudioBuffer, path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_wav_bytes(buf))
    return path


def frame_rms(pcm: bytes, frame_samples: int) -> list[float]:
    """RMS energy (0-1, normalised to full-scale int16) per fixed-size frame.

    The last partial frame, if any, is scored against its own (shorter)
    length rather than dropped -- discarding it would silently ignore
    trailing audio shorter than one frame.
    """
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size == 0 or frame_samples <= 0:
        return []
    out = []
    for start in range(0, samples.size, frame_samples):
        frame = samples[start : start + frame_samples]
        out.append(float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0)
    return out


def trim_silence(
    buf: AudioBuffer, *, threshold: float = 0.02, frame_ms: float = 20.0, pad_ms: float = 0.0
) -> AudioBuffer:
    """Cut leading/trailing frames below an energy threshold.

    Deliberately simple energy gating, not a VAD model -- good enough to
    drop dead air at the start/end of a push-to-talk recording (button
    pressed a beat before speaking, released a beat after) without pulling
    in a new dependency. `threshold` is a fraction of full-scale amplitude;
    0.02 was chosen as comfortably above typical mic self-noise/room tone
    and comfortably below a spoken syllable.
    """
    frame_samples = max(1, int(buf.sample_rate * frame_ms / 1000))
    energies = frame_rms(buf.pcm, frame_samples)
    if not energies:
        return buf

    loud = [i for i, e in enumerate(energies) if e >= threshold]
    if not loud:
        return AudioBuffer(pcm=b"", sample_rate=buf.sample_rate)

    # pad_ms keeps a little audio either side of the loud region: soft onsets like
    # the "wh" of "where" sit below the threshold, and cutting them made Whisper
    # hear "where does" as "or does" in a real run.
    pad = int(pad_ms / frame_ms)
    first, last = max(0, loud[0] - pad), min(len(energies) - 1, loud[-1] + pad)
    bytes_per_sample = 2
    start_byte = first * frame_samples * bytes_per_sample
    end_byte = min(len(buf.pcm), (last + 1) * frame_samples * bytes_per_sample)
    return AudioBuffer(pcm=buf.pcm[start_byte:end_byte], sample_rate=buf.sample_rate)


def is_effectively_silent(buf: AudioBuffer, *, threshold: float = 0.02, frame_ms: float = 20.0) -> bool:
    """True if no frame in the buffer clears the energy threshold at all.

    Used to skip an STT call entirely on a push-to-talk recording that
    captured only button-press noise or room tone -- no speech was ever
    said, so there is nothing for STT to usefully transcribe.
    """
    energies = frame_rms(buf.pcm, max(1, int(buf.sample_rate * frame_ms / 1000)))
    return not energies or max(energies) < threshold


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {w.getsampwidth()*8}-bit")
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
        if w.getnchannels() == 2:
            # Downmix to mono by taking every other sample pair average.
            import numpy as np

            arr = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)
            pcm = arr.mean(axis=1).astype("<i2").tobytes()
    return AudioBuffer(pcm=pcm, sample_rate=rate)


class PlaybackError(Exception):
    pass


def _try_sounddevice(buf: AudioBuffer) -> bool:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        return False
    try:
        arr = np.frombuffer(buf.pcm, dtype="<i2")
        sd.play(arr, buf.sample_rate)
        sd.wait()
        return True
    except Exception:  # noqa: BLE001 - any audio-device failure means "try next"
        return False


def _try_winsound(buf: AudioBuffer) -> bool:
    try:
        import winsound
    except ImportError:
        return False
    try:
        # winsound needs a real file or a WAV-in-memory bytes object.
        winsound.PlaySound(to_wav_bytes(buf), winsound.SND_MEMORY)
        return True
    except Exception:  # noqa: BLE001
        return False


def _try_ffplay(buf: AudioBuffer) -> bool:
    if not shutil.which("ffplay"):
        return False
    tmp = Path(tempfile.gettempdir()) / "voicenarrator_play.wav"
    try:
        save_wav(buf, tmp)
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(tmp)],
            check=True,
            timeout=300,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def play(buf: AudioBuffer, fallback_path: str | Path | None = None) -> str:
    """Play audio. Returns a short description of what actually happened.

    Backends are tried in order of latency: sounddevice (in-process, lowest),
    winsound (Windows built-in), ffplay (spawns a process). If all fail we save
    a file and say so, rather than raising -- losing audio output should not
    kill a background daemon.
    """
    if not buf.pcm:
        return "no audio (empty buffer)"

    for backend, fn in (
        ("sounddevice", _try_sounddevice),
        ("winsound", _try_winsound),
        ("ffplay", _try_ffplay),
    ):
        if fn(buf):
            return f"played via {backend}"

    path = Path(fallback_path or Path(tempfile.gettempdir()) / "voicenarrator_out.wav")
    save_wav(buf, path)
    return f"no audio backend available; saved to {path}"
