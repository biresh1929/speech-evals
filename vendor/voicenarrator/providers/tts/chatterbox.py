"""Chatterbox TTS provider -- zero-shot voice cloning from a short reference clip.

Resemble AI's Chatterbox (github.com/resemble-ai/chatterbox). Verified 2026-09-14
against the installed package (chatterbox-tts 0.1.7) and the Hugging Face repo:

LICENSING -- checked for the full pipeline, not just the headline:
* chatterbox-tts code: MIT.
* Model weights, ResembleAI/chatterbox on Hugging Face: MIT (model card).
* resemble-perth (the watermarker, including its bundled network checkpoint): MIT.
* s3tokenizer and diffusers (dependencies): Apache-2.0. Permissive and compatible;
  they are installed as dependencies, not vendored into this repo.
Unlike Rumik (CC-BY-NC), nothing here restricts commercial use. It is still an
optional extra: a user who only wants Piper never installs torch.

WATERMARKING -- every generation is watermarked, and this provider keeps it that
way. `ChatterboxTTS.generate` calls Perth's `apply_watermark` unconditionally;
there is no flag to turn it off. It could only be removed by replacing the
model's `watermarker` attribute, which this code never does. The watermark is
imperceptible and detectable with `perth.PerthImplicitWatermarker().get_watermark`
(see scripts/chatterbox_eval.py), so speech made with a cloned voice can be
identified as AI-generated. That is a feature.

CONSENT -- Clone only your own voice or a voice whose owner has explicitly
agreed. The dashboard upload enforces this with a required confirmation.

MEASURED 2026-09-14 (scripts/chatterbox_eval.py, RTX 4050 6GB, see BUILD_LOG):
peak VRAM 3.7GB (fits this 6GB card without spilling, unlike Rumik's 6.4GB);
~1x realtime. English cloning from an 11s reference: speaker similarity
0.92-0.94 (indistinguishable from same-speaker noise), WER 0-7%. Hindi cloning
from only an 8.4s reference (below the ~10s recommendation): similarity
0.85-0.87 (still far above cross-speaker ~0.7, but below same-speaker), WER
35-42% with real mispronunciations on Hindi-English code-switched technical
vocabulary. Whether that gap is the short reference or the multilingual
checkpoint itself is unresolved -- re-test with a >=10s Hindi reference before
relying on Hindi cloning.

VARIANTS
* variant="english"      (default) ChatterboxTTS, English only.
* variant="multilingual" ChatterboxMultilingualTTS, 23 languages including
  Hindi; pass `language` (e.g. "hi").

Options: voice (reference .wav path; omitted -> the model's built-in voice),
variant, language, device ("cuda"/"cpu"/auto), model_dir (a local snapshot
directory; omitted -> download from Hugging Face), exaggeration, cfg_weight,
temperature, max_chunk_chars.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..base import AudioBuffer, ProviderError, TTSProvider

log = logging.getLogger(__name__)

# Chatterbox caps a generation at 1000 speech tokens (25 tokens/s, so 40s). A
# long narration is split into sentence groups well under that, so no chunk
# is silently truncated.
DEFAULT_MAX_CHUNK_CHARS = 300
# Shorter references clone poorly; Resemble's examples use about 10s.
MIN_REFERENCE_SECONDS = 5.0
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def split_for_generation(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Group sentences into chunks of at most `max_chars`. A single sentence
    longer than that is split at commas, then hard-wrapped at spaces."""
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        buf = ""
        for part in re.split(r"(?<=,)\s+", sentence):
            while len(part) > max_chars:
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            if buf and len(buf) + 1 + len(part) > max_chars:
                pieces.append(buf)
                buf = part
            else:
                buf = f"{buf} {part}".strip()
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    current = ""
    for p in pieces:
        if current and len(current) + 1 + len(p) > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks


def float_to_pcm16(wav: np.ndarray) -> bytes:
    wav = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


class ChatterboxProvider(TTSProvider):
    name = "chatterbox"
    supports_streaming = False
    watermarked = True

    def __init__(self, **options: Any):
        self.variant = str(options.get("variant", "english")).lower()
        if self.variant not in ("english", "multilingual"):
            raise ProviderError(f"chatterbox variant must be 'english' or 'multilingual', got {self.variant!r}")
        self.language = options.get("language")
        if self.variant == "multilingual" and not self.language:
            raise ProviderError("chatterbox variant 'multilingual' needs a 'language' option, e.g. 'hi'.")

        voice = options.get("voice")
        self.voice = Path(str(voice)).expanduser() if voice else None
        if self.voice is not None and not self.voice.is_file():
            raise ProviderError(f"Chatterbox reference voice not found: {self.voice}")

        self.device = options.get("device")
        self.model_dir = options.get("model_dir")
        self.exaggeration = float(options.get("exaggeration", 0.5))
        self.cfg_weight = float(options.get("cfg_weight", 0.5))
        self.temperature = float(options.get("temperature", 0.8))
        self.max_chunk_chars = int(options.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))
        self.gap_ms = float(options.get("gap_ms", 120.0))

        self._model = None
        self._conds_for: Path | None = None
        self.sample_rate = 24000
        self.last_timings: dict[str, float] = {}

    # ---- loading ---------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise ProviderError(
                "Chatterbox is not installed. `pip install chatterbox-tts` "
                "(MIT; pulls in torch)."
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = ChatterboxMultilingualTTS if self.variant == "multilingual" else ChatterboxTTS
        t0 = time.perf_counter()
        try:
            if self.model_dir:
                self._model = cls.from_local(Path(str(self.model_dir)).expanduser(), device)
            else:
                self._model = cls.from_pretrained(device=torch.device(device))
        except FileNotFoundError as exc:
            raise ProviderError(
                f"Chatterbox weights missing in {self.model_dir}: {exc}. The multilingual "
                f"variant needs t3_mtl23ls_v2.safetensors, s3gen.pt, ve.pt and the "
                f"grapheme/Cangjie JSON files."
            ) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise ProviderError(
                "Out of GPU memory loading Chatterbox. Another model (Rumik, a local LLM) "
                "is probably holding the card; free it, or set tts.options.device: cpu."
            ) from exc
        self.sample_rate = int(self._model.sr)
        self.device = device
        self.last_timings["load_seconds"] = time.perf_counter() - t0

    def _ensure_voice(self) -> None:
        """Compute conditioning from the reference clip once per voice file."""
        if self.voice is None or self._conds_for == self.voice:
            return
        try:
            import soundfile as sf

            seconds = sf.info(str(self.voice)).duration
        except Exception:  # noqa: BLE001 - unreadable header: let librosa try
            seconds = None
        if seconds is not None and seconds < MIN_REFERENCE_SECONDS:
            raise ProviderError(
                f"Reference voice is {seconds:.1f}s; Chatterbox needs at least "
                f"{MIN_REFERENCE_SECONDS:.0f}s of clean speech (about 10s is better)."
            )
        t0 = time.perf_counter()
        self._model.prepare_conditionals(str(self.voice), exaggeration=self.exaggeration)
        self._conds_for = self.voice
        self.last_timings["voice_prepare_seconds"] = time.perf_counter() - t0

    def set_voice(self, path: str | Path) -> None:
        """Switch the cloned voice (e.g. after a dashboard upload)."""
        p = Path(str(path)).expanduser()
        if not p.is_file():
            raise ProviderError(f"Chatterbox reference voice not found: {p}")
        self.voice = p

    # ---- synthesis -------------------------------------------------------

    def synthesize(self, text: str, **kwargs: Any) -> AudioBuffer:
        chunks = split_for_generation(text, int(kwargs.get("max_chunk_chars", self.max_chunk_chars)))
        if not chunks:
            return AudioBuffer(pcm=b"", sample_rate=self.sample_rate)
        self._load()
        self._ensure_voice()

        gen_kwargs = {
            "exaggeration": float(kwargs.get("exaggeration", self.exaggeration)),
            "cfg_weight": float(kwargs.get("cfg_weight", self.cfg_weight)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
        }
        if self.variant == "multilingual":
            gen_kwargs["language_id"] = str(kwargs.get("language", self.language))

        gap = np.zeros(int(self.sample_rate * self.gap_ms / 1000.0), dtype=np.float32)
        parts: list[np.ndarray] = []
        t0 = time.perf_counter()
        for i, chunk in enumerate(chunks):
            try:
                wav = self._model.generate(chunk, **gen_kwargs)
            except Exception as exc:  # noqa: BLE001 - surface as a provider error
                raise ProviderError(f"Chatterbox generation failed: {type(exc).__name__}: {exc}") from exc
            arr = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            if i:
                parts.append(gap)
            parts.append(arr.astype(np.float32).reshape(-1))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self.last_timings["generate_seconds"] = time.perf_counter() - t0
        self.last_timings["audio_seconds"] = len(audio) / self.sample_rate
        return AudioBuffer(pcm=float_to_pcm16(audio), sample_rate=self.sample_rate)

    def health_check(self) -> tuple[bool, str]:
        try:
            self._load()
        except ProviderError as exc:
            return False, str(exc)
        return super().health_check()
