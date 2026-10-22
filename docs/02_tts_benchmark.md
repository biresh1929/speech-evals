# 2. TTS benchmark and pronunciation checking

**Question.** For a voice assistant that must answer quickly in Indian English and Hindi, how do open TTS systems
compare on speed, intelligibility and closeness to a reference voice? And can mispronounced technical terms be caught
automatically?

## Method

[`scripts/compare_tts_eval.py`](../scripts/compare_tts_eval.py), [`chatterbox_eval.py`](../scripts/chatterbox_eval.py),
[`pocket_tts_eval.py`](../scripts/pocket_tts_eval.py)

- **Speed:** real-time factor (synthesis time / audio duration) and time to the first audio chunk where streaming exists.
- **Intelligibility:** ASR round trip: synthesise a sentence, transcribe it with Whisper-small (CPU, int8), WER against
  the input text.
- **Voice similarity:** cosine similarity of speaker embeddings between the output and a reference recording.

## Results ([`results/tts_benchmark/compare_tts_eval.log`](../results/tts_benchmark/compare_tts_eval.log))

| system (device / voice) | n | RTF | first chunk | round-trip WER | speaker sim. |
|---|---|---|---|---|---|
| Piper lessac-low (CPU) / built-in | 3 | 0.33 | n/a | 0.078 | 0.56 |
| Pocket-TTS (CPU) / built-in | 3 | 0.32 | 0.127 s | 0.000 | 0.63 |
| Chatterbox English (GPU) / cloned reference | 3 | 1.07 | n/a | 0.046 | **0.93** |

Chatterbox multilingual ([`chatterbox_eval.log`](../results/tts_benchmark/chatterbox_eval.log)): English round-trip WER
0.071, 0.067, 0.000; **Hindi 0.353 and 0.421**, with speaker similarity 0.85–0.87. The Hindi gap is large even though
the voice stays close.

## Pronunciation checking ([`check_mispronunciations.py`](../scripts/check_mispronunciations.py))

Each technical term is synthesised in a sentence and transcribed back; a term the ASR does not recover verbatim is
flagged. "PostgreSQL" came back as "PaktiJoy SQL" (sentence WER 0.353): flagged. "kubectl" and "async" were recovered
verbatim: not flagged.

## Limits

n = 3 sentences per system: enough to rank, not to claim small differences. The round-trip WER mixes TTS errors with
ASR errors (Whisper-small is itself weak on Hindi, see study 1), so the Hindi numbers are an upper bound on TTS error.
"Not recovered verbatim" is a proxy for mispronunciation; a human listen is the real test.
