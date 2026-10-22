"""Long-form synthesis support for Rumik-OSS-1.

THE PROBLEM
-----------
Rumik-OSS-1 is one autoregressive sequence: text tokens and audio tokens share
a single context window of 8192 tokens. At 12.5 frames/sec x 8 codebooks it
emits 100 audio tokens per second of speech, and its own model card says
training utterances were capped at 30s with a recommendation not to exceed 35s.

So anything longer than ~35s has to be generated as several calls. The naive
approach -- split the text, generate each piece independently, concatenate --
produces an audible "reset" at every seam: pitch, pace and energy jump back to
the speaker's default because each call starts with no acoustic history.

THE APPROACH HERE
-----------------
Two continuity mechanisms, either of which can be disabled to A/B them:

1. TEXT CONDITIONING: the previous chunk's final sentence is prepended to the
   next chunk's prompt text, so the model has the linguistic run-up that
   shaped the prosody it is continuing from.

2. AUDIO PRIMER: the last N frames of the previous chunk's *codec tokens* are
   appended to the next chunk's prompt, after the `<audio>` marker. The model
   then continues an in-progress utterance rather than starting one. Because
   `generate_audio` returns prompt+generated and we slice off everything up to
   `input_ids.shape[1]`, the primer frames are consumed as context but never
   re-emitted -- so there is no duplicated audio to cut out.

Seams then get a short equal-power crossfade to smooth any residual step.

HONEST LIMITATION: this is a practical workaround, not a real long-context
solution. The model still has no memory beyond the primer window, so very long
passages will drift in tone over minutes. A true fix needs either a
longer-context model or explicit global prosody conditioning. Do not describe
this as "solving" long-form generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Sentence-ish boundaries, including Devanagari danda for Hindi/Indic text.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")

# Rumik emits 100 audio tokens per second (12.5 fps x 8 codebooks).
AUDIO_TOKENS_PER_SECOND = 100.0
# Mimi at 24kHz and 12.5 frames/sec: each codec frame decodes to 1920 samples.
SAMPLES_PER_FRAME = 1920


def context_trim_samples(context_frames: int, samples_per_frame: int = SAMPLES_PER_FRAME) -> int:
    """Samples to drop from a decode that was given `context_frames` of lead-in.

    Mimi's decoder is causal and frame-aligned, so frame i decodes to samples
    [i*1920, (i+1)*1920). Decoding [context + chunk] and dropping the context's
    samples leaves the chunk's audio, now decoded with history instead of cold.
    """
    return max(0, int(context_frames)) * int(samples_per_frame)
# Conservative words-per-second used only to size text chunks before any audio
# exists. Under-estimating is safe (smaller chunks); over-estimating risks
# hitting the model's length ceiling mid-sentence.
WORDS_PER_SECOND = 2.6


@dataclass
class TextChunk:
    text: str
    index: int
    # Previous chunk's last sentence, used as conditioning context.
    lead_in: str = ""


@dataclass
class ChunkPlan:
    chunks: list[TextChunk] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split((text or "").strip()) if s.strip()]


def plan_chunks(
    text: str,
    target_seconds: float = 22.0,
    max_seconds: float = 30.0,
) -> ChunkPlan:
    """Group sentences into chunks of roughly `target_seconds` of speech.

    Splits only at sentence boundaries. A single sentence longer than
    `max_seconds` is left intact in its own chunk rather than cut mid-clause:
    the model has no documented way to resume a half-finished sentence, and
    cutting one produces a truncated audio token stream, not a pause.

    Defaults sit below the model card's 35s ceiling on purpose -- the estimate
    is words-based and approximate, so the margin absorbs a bad guess.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ChunkPlan([])

    target_words = target_seconds * WORDS_PER_SECOND
    max_words = max_seconds * WORDS_PER_SECOND

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_words = 0
    prev_last_sentence = ""

    def flush() -> None:
        nonlocal current, current_words, prev_last_sentence
        if not current:
            return
        chunks.append(
            TextChunk(text=" ".join(current), index=len(chunks), lead_in=prev_last_sentence)
        )
        prev_last_sentence = current[-1]
        current = []
        current_words = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > max_words:
            flush()
        current.append(sentence)
        current_words += n
        if current_words >= target_words:
            flush()

    flush()
    return ChunkPlan(chunks)


def crossfade_concat(
    segments: list[np.ndarray],
    sample_rate: int = 24000,
    crossfade_ms: float = 15.0,
) -> np.ndarray:
    """Concatenate float32 waveforms with an equal-power crossfade at seams.

    Equal-power (cosine) rather than linear: a linear fade across two
    uncorrelated signals dips in perceived loudness at the midpoint, which is
    audible as a small "hole" exactly where we are trying to hide a seam.
    """
    segments = [s for s in segments if s is not None and s.size > 0]
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return segments[0].astype(np.float32)

    n_fade = max(1, int(sample_rate * crossfade_ms / 1000.0))
    out = segments[0].astype(np.float32)

    for seg in segments[1:]:
        seg = seg.astype(np.float32)
        n = int(min(n_fade, out.size, seg.size))
        if n <= 0:
            out = np.concatenate([out, seg])
            continue
        t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        mixed = out[-n:] * fade_out + seg[:n] * fade_in
        out = np.concatenate([out[:-n], mixed, seg[n:]])

    return out


def codes_to_unit_tokens(codes: np.ndarray, first_unit_id: int, num_quantizers: int) -> list[int]:
    """Flatten `[num_quantizers, n_frames]` codec codes into unit token ids.

    Exact inverse of the model's own `audio_tokens_to_codes`, whose layout is
    code-major / quantizer-minor:

        token_id = first_unit_id + code * num_quantizers + quantizer

    Verified against configuration_rumik_oss.py's docstring and the tokenizer
    (`<0_0>` = 261008 ... `<2047_7>` = 277391).
    """
    if codes.ndim != 2:
        raise ValueError(f"expected [num_quantizers, n_frames], got shape {codes.shape}")
    n_q, n_frames = codes.shape
    if n_q != num_quantizers:
        raise ValueError(f"expected {num_quantizers} quantizers, got {n_q}")
    tokens: list[int] = []
    for frame in range(n_frames):
        for q in range(num_quantizers):
            tokens.append(int(first_unit_id + int(codes[q, frame]) * num_quantizers + q))
    return tokens


def tail_primer_tokens(
    codes: np.ndarray,
    first_unit_id: int,
    num_quantizers: int,
    n_frames: int,
) -> list[int]:
    """Unit tokens for the LAST `n_frames` frames -- the acoustic primer."""
    if codes.size == 0 or n_frames <= 0:
        return []
    tail = codes[:, -n_frames:]
    return codes_to_unit_tokens(tail, first_unit_id, num_quantizers)
