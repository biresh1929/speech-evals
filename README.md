# speech-evals

Evaluation and interpretability studies I ran while building **WapasBol**, a real-time English / Hindi / Hinglish voice
assistant for programmers ([demo film](https://drive.google.com/file/d/15mDHMd_WOsVjU2hHr9mRmsJB2GdGiqnC/view?usp=sharing),
[birulabs.com](https://birulabs.com)). The product code is private; this repository holds the research parts: the code
as it was run, the data I can share, and the raw logs, including the experiments that failed.

| Study | Question | Result | Write-up |
|---|---|---|---|
| Hindi ASR on code-switched speech | How well does Whisper-small transcribe Hindi programming talk? | **77.2% pooled WER** on the 57 correctable segments (mean per-segment 79.6%, 95% CI 67.5–91.8%); 138 of 195 segments unrecoverable | [docs/01](docs/01_hindi_asr_wer.md) |
| TTS benchmark | Which TTS is fast, intelligible and close to a reference voice? | Pocket-TTS: RTF 0.32, first chunk 127 ms; cloned Chatterbox: 0.93 speaker similarity; multilingual Hindi round-trip WER 0.35–0.42 vs ≤0.07 English | [docs/02](docs/02_tts_benchmark.md) |
| TTS interpretability (Rumik-OSS-1, 3.38B) | Where does a word's identity enter the audio, and can activation steering fix a mispronunciation? | Causal tracing and head knockouts point to layers 11 and 27; **steering did not fix it (0/6 seeds)** | [docs/03](docs/03_tts_interpretability.md) |
| Voice-pipeline latency | Where does the time go, per stage? | ASR p50 965 ms → 155 ms by moving the *same* model to CUDA; time-to-first-audio p50 819 → 549 ms | [docs/04](docs/04_latency.md) |

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
python scripts/wer_hindi_corrections.py             # study 1: prints the WER, CIs and bias counts; no GPU needed
python latency/tools/latency_report.py              # study 4: recomputes every latency number from the logs
```

Studies 2 and 3 download open models (Piper, Pocket-TTS, Chatterbox, Rumik-OSS-1) and need a CUDA GPU for reasonable
speed; each write-up lists its commands. Scripts read and write under `workspace/` (set `SPEECH_EVALS_ROOT` to use
another folder). Generated audio is not committed; the logs and JSON results in `results/` are what the write-ups quote.

## What is and isn't here

- **Here:** the scripts exactly as run (only hard-coded local paths were replaced by `DATA_ROOT`), the thin TTS wrappers
  they import (`vendor/`), the Hindi review transcripts (text only), every log and JSON result, the latency logs (stage
  timings, text lengths and model names; no transcript text).
- **Not here:** audio (including my own voice and a cloned voice), the full speech dataset, the WapasBol product code.
- **Limits I know about:** each write-up ends with them. The Hindi study has one annotator and survivor bias; the TTS
  benchmark is small (n = 3 per system); the interpretability work tests one word in depth, not a held-out set.

## How this was made

I designed the questions, the experiments and their controls, ran them, and read every result; I wrote the code with
AI coding assistance (Claude Code) and reviewed it. [PROVENANCE.md](PROVENANCE.md) lists when each script was first
committed in my private working repositories.

Biresh Kumar Singh · bireshkumar1964@gmail.com · [LinkedIn](https://linkedin.com/in/biresh-kumar-singh)
