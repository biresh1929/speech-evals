"""Rumik-OSS-1 TTS provider -- Indic multilingual, 22 languages + English.

=============================================================================
LICENSING -- READ BEFORE ENABLING THIS PROVIDER
=============================================================================
The Rumik-OSS-1 *weights* are CC-BY-NC 4.0 plus the Cohere Labs Acceptable Use
Policy. The model card states, verbatim, that the licence "does not authorize
commercial products, paid synthesis services, or other uses directed toward
commercial advantage, INCLUDING SELF-HOSTED DEPLOYMENTS."

So: research, personal use, and free non-commercial tools are fine. Shipping a
product that uses this, or hosting it as a service (free or paid), is not.
A research label does not make a commercial activity non-commercial.

voicenarrator itself is MIT; enabling this provider does not change that, but
it does constrain what *you* may do with the output and the deployment. The
bundled Mimi codec is separately CC-BY-4.0 (Kyutai) and is commercially
reusable with attribution -- that permission covers the codec only, not the
tiny-aya-derived TTS weights.
=============================================================================

Usage pattern below follows a verified working run on this hardware, not the
model card's prose. Confirmed facts it relies on:
  * generation stops via a trained `stop_predictor` head, not just max_new_tokens
  * speakers (Ira/Aisha/Siya/Zoya) are plain text in the prompt, NOT special tokens
  * output is 24kHz; `mimi.decode(codes)` returns float waveform
  * there is NO `decode_step`/rustymimi API on this checkpoint's codec class
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ..base import AudioBuffer, ProviderError, TTSProvider
from .rumik_longform import (
    AUDIO_TOKENS_PER_SECOND,
    context_trim_samples,
    crossfade_concat,
    plan_chunks,
    tail_primer_tokens,
)

log = logging.getLogger(__name__)

RUMIK_SAMPLE_RATE = 24000
SPEAKERS = ("Ira", "Aisha", "Siya", "Zoya")

# VRAM modes. This model + codec allocate ~6.76GB in bf16, which does not fit a
# 6GB card without the driver spilling to shared system memory (measured ~22-25x
# slower than realtime). These modes exist to find what actually fits.
VRAM_MODES = ("bf16_gpu", "codec_cpu", "4bit")

# MEASURED allocations from scripts/bench_rumik_vram.py on an RTX 4050 6GB,
# not estimates:
#   bf16_gpu   6.763 GB   (LM + codec both on GPU)
#   codec_cpu  6.404 GB   (codec on CPU frees ~0.36 GB)
#   4bit       OOM at 5.24 GB allocated -- quantised load needs more transient
#              memory than the finished model does, and the 277k-token tied
#              embedding is not quantised
_VRAM_NEEDED_GB = {
    "bf16_gpu": 6.76,
    "codec_cpu": 6.40,
    "4bit": 5.3,
}

# How much overflow the Windows WDDM driver will absorb into shared system
# memory before the allocation dies instead of spilling. Empirical: codec_cpu
# allocated 6.40GB with only 5.27GB free (a 1.13GB spill) and completed, just
# ~25x slower than realtime. 1.6 leaves margin above the one data point without
# pretending the limit is known precisely.
_MAX_SPILL_GB = 1.6


def _preflight_vram(mode: str) -> None:
    """Fail with an explanation instead of segfaulting.

    When the GPU is short on memory, loading this model does not raise a clean
    torch OutOfMemoryError -- it dies inside the driver and takes the process
    with it (observed: exit code 139, no traceback, no Python-level exception
    to catch). By the time allocation fails there is nothing left to report, so
    the check has to happen before the load starts.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    needed = _VRAM_NEEDED_GB.get(mode, 6.9)

    if free_gb >= needed:
        return

    used_gb = total_gb - free_gb

    # A shortfall the driver can absorb is survivable: on Windows WDDM spills
    # the overflow into shared system memory. Verified working -- codec_cpu
    # allocated 6.40GB with 5.27GB free and produced audio, at ~25x realtime.
    # Beyond the spill limit the allocation dies inside the driver and takes the
    # process with it (exit 139, nothing catchable), so refuse up front.
    if free_gb >= needed - _MAX_SPILL_GB:
        log.warning(
            "Rumik mode %r needs ~%.2fGB but only %.2fGB is free. Attempting anyway: "
            "the driver will spill ~%.2fGB into shared system memory, and generation "
            "will be far slower than realtime (measured ~25x on a 6GB card).",
            mode, needed, free_gb, needed - free_gb,
        )
        return

    hint = ""
    if used_gb > 1.0:
        hint = (
            f" Something else is already using {used_gb:.1f}GB of this GPU "
            f"(a local LLM server such as LM Studio or Ollama with a model "
            f"loaded is the usual cause). Unload it, or point the LLM at a "
            f"cloud provider."
        )

    raise ProviderError(
        f"Not enough free VRAM for Rumik in mode {mode!r}: need about "
        f"{needed:.1f}GB, only {free_gb:.2f}GB free of {total_gb:.1f}GB total.{hint} "
        f"Refusing to load -- at this shortfall the CUDA allocation crashes the "
        f"process rather than raising a catchable error."
    )


class RumikProvider(TTSProvider):
    name = "rumik"
    supports_streaming = False  # see rumik_longform docstring: no decode_step API

    def __init__(self, **options: Any):
        self.model_path = str(
            options.get("model_path")
            or options.get("model")
            or "rumik-ai/rumik-oss-1"
        )
        self.speaker = str(options.get("speaker", "Ira"))
        if self.speaker not in SPEAKERS:
            raise ProviderError(
                f"Unknown Rumik speaker {self.speaker!r}. Available: {', '.join(SPEAKERS)}"
            )
        self.description = str(
            options.get("description", "clear, moderate pace, neutral tone")
        )
        self.vram_mode = str(options.get("vram_mode", "codec_cpu"))
        if self.vram_mode not in VRAM_MODES:
            raise ProviderError(
                f"Unknown vram_mode {self.vram_mode!r}. Available: {', '.join(VRAM_MODES)}"
            )
        self.allow_4bit_experimental = bool(options.get("allow_4bit_experimental", False))
        # Bypass the VRAM refusal. Only for benchmarking, where the point is to
        # find out empirically what this card can do and each attempt runs in
        # its own subprocess so a crash is contained.
        self.ignore_vram_preflight = bool(options.get("ignore_vram_preflight", False))
        # Opt-in alternative load strategy; see _load for why it is not default.
        self.stream_to_gpu = bool(options.get("stream_to_gpu", False))
        self.temperature = float(options.get("temperature", 0.8))
        self.top_k = int(options.get("top_k", 30))

        # Long-form settings.
        self.chunk_target_seconds = float(options.get("chunk_target_seconds", 18.0))
        # 24s, not 25: with the 1.25 headroom below, 25s would need 3125 tokens
        # and the model card's generation ceiling is 3072. 24s needs exactly
        # 3000, which fits.
        self.chunk_max_seconds = float(options.get("chunk_max_seconds", 24.0))

        # max_new_tokens is an AUDIO LENGTH LIMIT, not a text budget: the model
        # emits 100 audio tokens per second (12.5 frames x 8 codebooks). The
        # default of 2048 therefore caps a chunk at 20.48s -- shorter than the
        # 22s chunks this provider was asking for, so every long chunk was
        # silently truncated mid-sentence. Transcribing the output is what
        # exposed it: the naive A/B sample lost a whole sentence at the seam.
        # Derive the cap from the chunk ceiling instead, with 25% headroom
        # because speech-rate estimates are approximate, and keep it within the
        # 3072 the model card uses.
        derived = int(self.chunk_max_seconds * AUDIO_TOKENS_PER_SECOND * 1.25)
        self.max_new_tokens = int(
            options.get("max_new_tokens", min(max(derived, 1024), 3072))
        )
        if self.max_new_tokens < self.chunk_max_seconds * AUDIO_TOKENS_PER_SECOND:
            log.warning(
                "max_new_tokens=%d allows only %.1fs of audio but chunks may be "
                "up to %.1fs -- chunks will be truncated mid-sentence.",
                self.max_new_tokens,
                self.max_new_tokens / AUDIO_TOKENS_PER_SECOND,
                self.chunk_max_seconds,
            )
        self.crossfade_ms = float(options.get("crossfade_ms", 15.0))
        # "continuity" conditions each chunk on the previous chunk's trailing
        # audio codes; "naive" generates each chunk cold.
        #
        # DEFAULT IS "naive", deliberately. In a real listening test the naive
        # output was judged clearer and more articulate, and continuity showed a
        # noticeably different tone without a confident quality win. The
        # continuity path is the more complex one, so it does not get to be the
        # default on the strength of a hypothesis. See BUILD_LOG.
        self.continuity = str(options.get("continuity", "naive"))
        if self.continuity not in ("naive", "continuity"):
            raise ProviderError(
                f"tts.continuity must be 'naive' or 'continuity', got {self.continuity!r}"
            )
        # Continuity mode only: frames of the previous chunk given to the Mimi
        # decoder as history. 8 measured sufficient (see _decode_with_context).
        self.decoder_context_frames = int(options.get("decoder_context_frames", 8))
        # Populated by synthesize(): where chunk joins land, for seam analysis.
        self.last_seam_times: list[float] = []
        self.last_chunk_seconds: list[float] = []
        self.primer_frames = int(options.get("primer_frames", 8))  # 8 frames = 0.64s

        self._model = None
        self._tokenizer = None
        self._mimi = None
        self._load_seconds = 0.0

    # ---- loading ---------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return

        # NOTE: PyTorch's OOM message recommends
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for fragmentation.
        # Tried it here; torch reports "expandable_segments not supported on
        # this platform" on Windows, so it is a no-op and is deliberately not
        # set. Left documented so nobody re-adds it expecting it to help.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
        except ImportError as exc:
            raise ProviderError(
                "Rumik needs torch + transformers. Install with "
                "`pip install voicenarrator[rumik]`."
            ) from exc

        if not torch.cuda.is_available():
            raise ProviderError(
                "Rumik-OSS-1 requires a CUDA GPU (the model card states this "
                "explicitly). No CUDA device is visible. Use the piper provider "
                "for CPU-only machines."
            )

        if self.ignore_vram_preflight:
            log.warning(
                "VRAM preflight bypassed for mode %r -- if there is not enough "
                "memory this will crash the process, not raise.",
                self.vram_mode,
            )
        else:
            _preflight_vram(self.vram_mode)

        t0 = time.perf_counter()
        path = self.model_path
        log.info("loading Rumik-OSS-1 from %s (mode=%s)", path, self.vram_mode)

        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        if self.vram_mode == "4bit":
            # Known broken on this stack: bitsandbytes 0.50.2 + transformers
            # 5.2.0 + this trust_remote_code Cohere2 subclass segfaults during
            # quantised load (exit 139, no traceback). Verified it is specific
            # to this model -- a standalone bnb Linear4bit forward pass works
            # fine, and the crash reproduces with both device_map={"":0} and
            # device_map="auto". Gated behind an explicit opt-in so it cannot
            # take down a background daemon by surprise.
            if not self.allow_4bit_experimental:
                raise ProviderError(
                    "vram_mode '4bit' is not usable with this model on this "
                    "stack -- the quantised load segfaults (bitsandbytes "
                    "0.50.2 / transformers 5.2.0 / custom Cohere2 remote code). "
                    "Set tts.allow_4bit_experimental: true to try it anyway "
                    "(it may kill the process), or use 'codec_cpu'."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ProviderError(
                    "4bit mode needs bitsandbytes. `pip install voicenarrator[train]`."
                ) from exc
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                quantization_config=quant,
                device_map={"": 0},
            ).eval()
        elif self.stream_to_gpu:
            # Streams each shard straight onto the GPU instead of materialising
            # the whole model in CPU RAM first. Lower peak in principle, but on
            # this machine it traded a CUDA OOM for a Windows paging-file
            # exhaustion (os error 1455) during the memory-mapped safetensors
            # read, so it is opt-in rather than the default until it has been
            # shown to help somewhere.
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            ).eval()
        else:
            # The verified path: this is what produced the benchmark numbers and
            # the working audio samples. Do not replace it with something
            # cleverer without measuring the replacement first.
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, dtype=torch.bfloat16
                )
                .eval()
                .cuda()
            )

        # Codec placement is the whole point of codec_cpu mode: Mimi is small
        # relative to the 3B LM, and moving it off-GPU buys back headroom on a
        # card that cannot hold both.
        mimi = MimiModel.from_pretrained(path, subfolder="codec").eval()
        self._mimi = mimi if self.vram_mode == "codec_cpu" else mimi.cuda()

        self._load_seconds = time.perf_counter() - t0
        log.info("Rumik loaded in %.2fs (mode=%s)", self._load_seconds, self.vram_mode)

    def vram_report(self) -> dict[str, float]:
        try:
            import torch
        except ImportError:
            return {}
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "load_seconds": self._load_seconds,
        }

    # ---- generation ------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        """`<text>SPEAKER: <description="..."> TEXT<audio>` -- verified format.

        NOTE: an earlier version prepended the previous chunk's final sentence
        here as "linguistic context". That was wrong. Rumik's prompt format has
        no field for context that should inform prosody without being spoken --
        everything between the description and `<audio>` gets read aloud. So the
        lead-in was simply spoken a second time, and transcribing the output
        showed the seam sentence duplicated verbatim.

        It went unnoticed at first because a separate bug (max_new_tokens
        capping chunks at 20.48s) was truncating the repeat before it finished.
        Fixing the truncation made the duplication audible.

        Continuity across chunks now comes solely from the audio-code primer in
        `_generate_codes`, which conditions the model acoustically and is
        structurally incapable of duplicating text: the primer tokens are
        consumed as context and sliced off the output.
        """
        return f'<text>{self.speaker}: <description="{self.description}"> {text}<audio>'

    def _generate_codes(self, text: str, primer: list[int] | None = None):
        """One generation call. Returns codec codes as `[n_quantizers, frames]`."""
        import torch

        prompt = self._build_prompt(text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if primer:
            # Append the previous chunk's trailing codec frames so the model
            # continues an in-progress utterance instead of starting cold.
            # These are consumed as context only: generate_audio returns
            # prompt+generated and the caller slices from input_ids.shape[1],
            # so primer frames are never re-emitted as duplicate audio.
            primer_t = torch.tensor([primer], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, primer_t], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(primer_t)], dim=1
            )

        prompt_len = input_ids.shape[1]
        ids = self._model.generate_audio(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            do_sample=True,
        )
        audio_tokens = ids[0].tolist()[prompt_len:]
        if not audio_tokens:
            raise ProviderError("Rumik generated no audio tokens.")

        codes = self._model.audio_tokens_to_codes(audio_tokens)  # [1, n_q, frames]
        return codes, len(audio_tokens)

    def _decode_with_context(self, codes, context_codes) -> np.ndarray:
        """Decode a chunk with the previous chunk's trailing codes as history.

        Why this exists -- a real defect found by blind listening. In continuity
        mode the LM continues mid-utterance, so a chunk's first codec frames are
        mid-speech. But each chunk used to be decoded independently, so Mimi's
        decoder met those frames with no history and produced an onset
        transient right on top of speech. Measured on re-encoded real audio:
        RMS error in the first 80ms of a cold mid-speech decode was 0.378 of the
        signal, versus 0.113 once 8 frames of context are supplied -- the same
        level as steady state. Both clips a listener flagged for a laugh-like
        sound at the join were continuity clips that started with no pause.

        Naive mode does not need this: its chunks begin in silence, where a
        cold-start transient has nothing audible to land on.
        """
        import torch

        if context_codes is None or context_codes.shape[-1] == 0:
            return self._decode(codes)
        joined = torch.cat([context_codes.to(codes.device), codes], dim=-1)
        wav = self._decode(joined)
        return wav[context_trim_samples(context_codes.shape[-1]):]

    def _decode(self, codes) -> np.ndarray:
        """Decode one chunk's codes to a waveform, then release everything.

        Mimi's decoder is a stack of transposed convolutions, and on CPU its
        intermediate activations are large. Running it while the 3B LM is
        already spilling into shared system memory exhausted RAM in a real run
        (`RuntimeError: bad allocation` inside conv_transpose1d). Copying the
        result out to numpy and dropping the torch tensors immediately keeps
        peak usage to one chunk at a time rather than the whole utterance.
        """
        import gc

        import torch

        with torch.inference_mode():
            decoded = self._mimi.decode(codes.to(self._mimi.device))
            wav = decoded.audio_values[0, 0].float().cpu().numpy().copy()

        del decoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return wav

    def synthesize(self, text: str, **kwargs: Any) -> AudioBuffer:
        text = (text or "").strip()
        if not text:
            return AudioBuffer(pcm=b"", sample_rate=RUMIK_SAMPLE_RATE)

        self._load()

        plan = plan_chunks(
            text,
            target_seconds=float(kwargs.get("chunk_target_seconds", self.chunk_target_seconds)),
            max_seconds=float(kwargs.get("chunk_max_seconds", self.chunk_max_seconds)),
        )
        if not plan.chunks:
            return AudioBuffer(pcm=b"", sample_rate=RUMIK_SAMPLE_RATE)

        continuity = str(kwargs.get("continuity", self.continuity))
        use_continuity = continuity == "continuity"

        cfg = self._model.config
        first_unit_id = int(cfg.first_unit_id)
        num_quantizers = int(cfg.num_quantizers)

        segments: list[np.ndarray] = []
        primer: list[int] = []
        total_tokens = 0
        t0 = time.perf_counter()

        prev_tail_codes = None
        for chunk in plan.chunks:
            codes, n_tokens = self._generate_codes(
                chunk.text, primer=primer if use_continuity else None
            )
            total_tokens += n_tokens
            if use_continuity and self.decoder_context_frames > 0:
                segments.append(self._decode_with_context(codes, prev_tail_codes))
                prev_tail_codes = codes[:, :, -self.decoder_context_frames:]
            else:
                segments.append(self._decode(codes))

            if use_continuity:
                arr = codes[0].cpu().numpy() if hasattr(codes, "cpu") else np.asarray(codes[0])
                primer = tail_primer_tokens(
                    arr, first_unit_id, num_quantizers, self.primer_frames
                )

        crossfade_ms = float(kwargs.get("crossfade_ms", self.crossfade_ms))

        # Record where the joins land in the final waveform so a caller can cut
        # the seam out and listen to it in isolation. Without this, evaluating
        # the crossfade means guessing at a percentage through the file, which
        # is how the first comparison ended up inconclusive. Each seam sits at
        # the running total of preceding segments minus the overlap consumed by
        # earlier crossfades.
        overlap_samples = max(1, int(RUMIK_SAMPLE_RATE * crossfade_ms / 1000.0))
        seam_times: list[float] = []
        running = 0
        for i, seg in enumerate(segments[:-1]):
            running += len(seg)
            if i > 0:
                running -= min(overlap_samples, len(segments[i - 1]), len(seg))
            seam_times.append(running / RUMIK_SAMPLE_RATE)
        self.last_seam_times = seam_times
        self.last_chunk_seconds = [len(s) / RUMIK_SAMPLE_RATE for s in segments]

        wav = crossfade_concat(
            segments,
            sample_rate=RUMIK_SAMPLE_RATE,
            crossfade_ms=crossfade_ms,
        )
        elapsed = time.perf_counter() - t0
        duration = len(wav) / RUMIK_SAMPLE_RATE if len(wav) else 0.0
        log.info(
            "Rumik: %d chunk(s), %d tokens, %.2fs audio in %.1fs (%.1fx realtime), mode=%s",
            len(plan.chunks), total_tokens, duration, elapsed,
            (elapsed / duration) if duration else 0.0, self.vram_mode,
        )

        pcm = np.clip(wav * 32767.0, -32768, 32767).astype("<i2").tobytes()
        return AudioBuffer(pcm=pcm, sample_rate=RUMIK_SAMPLE_RATE)

    def health_check(self) -> tuple[bool, str]:
        try:
            self._load()
        except ProviderError as exc:
            return False, str(exc)
        vram = self.vram_report()
        return True, (
            f"loaded in {self._load_seconds:.1f}s, mode={self.vram_mode}, "
            f"VRAM {vram.get('allocated_gb', 0):.2f}GB"
        )
