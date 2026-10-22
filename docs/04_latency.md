# 4. Latency of a streaming voice pipeline

**Question.** In a speak, transcribe, think, speak loop, where does the time go, and what actually reduced it?

## Instrumentation

Every stage is wrapped by [`latency/observability/latency.py`](../latency/observability/latency.py) and appended to a
JSONL log: stage name, duration, success, and metadata (text *length*, model name; no transcript text). Percentiles
come from the logs, never from estimates: `python latency/tools/latency_report.py`.

| log | period |
|---|---|
| `latency.before-singleshot.jsonl` | faster-whisper on CPU |
| `latency.before-nemotron.jsonl` | an LLM that was later replaced |
| `latency.after-qwen3.jsonl` | after the LLM change |
| `latency.jsonl` | current (GPU era): 3,962 rows |

## Results

| stage | CPU era p50 (n) | GPU era p50 (n) |
|---|---|---|
| `stt.transcribe` | 965 ms (135) | **155 ms** (486) |
| `voice.ttfa` (time to first audio) | 819 ms (271) | **549 ms** (436) |
| `tts.first_chunk` | 82 ms (294) | 82 ms (889) |

**What changed:** the *same* faster-whisper model was moved from CPU to CUDA (a deployment change, not a new model or
algorithm). Time to first audio fell less than ASR did, because the LLM and TTS stages did not get faster.

Also in the current log: the sentence-completeness gate decided "commit" 277 times (91.1%) and "wait" 27 times (8.9%);
27.1% of turns took the fast path (answer formed from a partial transcript); 0 of 3,962 rows raised an exception.

## Limits

The CPU and GPU logs come from different sessions and utterances, so this is not a paired comparison; p95/p99 are long
because some turns include long LLM answers. An exception count does not catch silent failures (an LLM returning empty
text was found by reading the logs, not by this counter).
