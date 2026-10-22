"""Piper TTS via an isolated subprocess.

LICENSING -- this is the whole reason this file shells out instead of importing:

The maintained Piper package (OHF-Voice/piper1-gpl, which is what `pip install
piper-tts` gives you today) is GPL-3.0-or-later. The archived MIT original
(rhasspy/piper) went read-only in October 2025. Importing GPL code into this
process would arguably make voicenarrator a derivative work and force the whole
project to GPL.

So: Piper runs as a separate OS process, communicating only over stdin/stdout
pipes, and `piper-tts` is deliberately NOT declared as a dependency in
pyproject.toml -- the user installs it themselves if they want this provider.
That keeps voicenarrator cleanly MIT. Do not "optimise" this into an import.

Verified CLI contract (piper 1.8.0, checked via `python -m piper --help`):
    -m/--model PATH      .onnx voice file
    --output-raw         stream raw PCM to stdout  <- what we use
    -f/--output-file     write a WAV instead
    --length-scale N     phoneme length; >1 slower, <1 faster
    text arrives on stdin
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..base import AudioBuffer, ProviderError, TTSProvider

DEFAULT_TIMEOUT = 120


class PiperProvider(TTSProvider):
    name = "piper"
    supports_streaming = False

    def __init__(self, **options: Any):
        model = options.get("model") or options.get("voice")
        if not model:
            raise ProviderError(
                "Piper needs a 'model' option pointing at a .onnx voice file. "
                "Download voices from https://github.com/OHF-Voice/piper1-gpl "
                "(or the rhasspy VOICES.md list)."
            )
        self.model_path = Path(str(model)).expanduser()
        if not self.model_path.is_file():
            raise ProviderError(f"Piper voice not found: {self.model_path}")

        # Piper ships the sample rate in a sidecar JSON. Read it rather than
        # assuming 22050 -- voices genuinely differ (lessac-low is 16000), and
        # guessing here produces audio that plays at the wrong pitch.
        self.sample_rate = self._read_sample_rate(self.model_path)

        self.length_scale = float(options.get("length_scale", 1.0))
        speed = options.get("speed")
        if speed:
            # speed is the intuitive knob; piper wants its inverse.
            self.length_scale = 1.0 / float(speed)
        self.use_cuda = bool(options.get("cuda", False))
        self.timeout = float(options.get("timeout", DEFAULT_TIMEOUT))
        self.command = self._resolve_command(options.get("command"))

    @staticmethod
    def _read_sample_rate(model_path: Path) -> int:
        cfg_path = Path(str(model_path) + ".json")
        if not cfg_path.is_file():
            cfg_path = model_path.with_suffix(".onnx.json")
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                rate = cfg.get("audio", {}).get("sample_rate")
                if rate:
                    return int(rate)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        raise ProviderError(
            f"Could not read sample_rate from {cfg_path}. Piper voices need their "
            f".onnx.json sidecar file next to the .onnx."
        )

    @staticmethod
    def _resolve_command(explicit: Any) -> list[str]:
        """Prefer a real `piper` binary; fall back to `python -m piper`."""
        if explicit:
            return explicit if isinstance(explicit, list) else [str(explicit)]
        binary = shutil.which("piper")
        if binary:
            return [binary]
        return [sys.executable, "-m", "piper"]

    def synthesize(self, text: str, **kwargs: Any) -> AudioBuffer:
        text = (text or "").strip()
        if not text:
            return AudioBuffer(pcm=b"", sample_rate=self.sample_rate)

        length_scale = float(kwargs.get("length_scale", self.length_scale))
        argv = [
            *self.command,
            "-m", str(self.model_path),
            "--output-raw",
            "--length-scale", str(length_scale),
        ]
        if self.use_cuda:
            argv.append("--cuda")

        try:
            proc = subprocess.run(
                argv,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                f"Could not run Piper ({' '.join(self.command)}). "
                f"Install it with `pip install piper-tts` (note: GPL-3.0), or set "
                f"tts.command in your config."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Piper timed out after {self.timeout}s.") from exc

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ProviderError(f"Piper exited {proc.returncode}: {err[:400]}")

        if not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ProviderError(f"Piper produced no audio. stderr: {err[:300]}")

        return AudioBuffer(pcm=proc.stdout, sample_rate=self.sample_rate)
