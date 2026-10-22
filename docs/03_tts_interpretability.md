# 3. Where does a word enter the audio? Interpretability of an LLM-based TTS model

**Model.** Rumik-OSS-1: 3.38B parameters, a Cohere2 decoder that predicts audio tokens for the Mimi codec
(8 quantizers per frame). Architecture read from source: [rumik_oss_1_architecture_notes.md](rumik_oss_1_architecture_notes.md).

**Question.** The model says "PostgreSQL" wrongly (study 2). Where, inside the network, is the word's identity read
into the audio, and can a runtime intervention (no weight edits) make it say the word correctly?

Every intervention is a forward hook enabled only during a run; no parameter is written.

## Step 1: causal tracing over layers ([`sonoedit_trace.py`](../scripts/sonoedit_trace.py))

Corrupt the word's text-token embeddings, then restore the clean hidden state at one layer and measure how much of the
correct output comes back (on the word's audio frames, first quantizer). Over 144 conditions per seed
([`sweep_summary_seed11.json`](../results/tts_interpretability/sonoedit/sweep_summary_seed11.json)):

- restoring at layers 0–10 recovers **≈100%**;
- at layers 11–26 only **≈70%**;
- from layer 27 on, **≈0%**.

So the word's identity is read out of its text tokens at **layer 11** and again at **layer 27**.

## Step 2: which heads? ([`sonoedit_heads.py`](../scripts/sonoedit_heads.py))

**Patterns.** Per head, the attention mass from the word's audio frames to the word's text tokens, and the norm of
what the head writes from them. Stable across seeds 11–13: layer 11 head 0 carries ≈39% of that layer's read from the
word; layer 27 is spread over heads 3, 4, 6 and 7.

**Knockout.** One head is forbidden to attend from positions after the word to the word's text tokens (scores set to
−∞ before softmax). This removes exactly that read and nothing else the head does.

**Effect** = fraction of the word's dependence removed = (p_clean − p_intervened) / (p_clean − p_corrupted), on the
word's frames. 1.0 means as bad as destroying the word's embedding.

| intervention | seed 11 | seed 12 | seed 13 |
|---|---|---|---|
| sanity: replacement forward, no intervention | −0.007 | +0.009 | −0.004 |
| all heads, layer 11 | 0.31 | 0.43 | 0.46 |
| all heads, layer 27 | 0.73 | 0.69 | 0.65 |
| all heads, both layers | 1.02 | 0.87 | 0.94 |
| top-1 head per layer | 0.49 | 0.60 | 0.47 |
| top-4 heads per layer | 0.63 | 0.79 | 0.66 |

**Controls.** (a) The no-intervention pass reproduces the stock model (|effect| ≤ 0.009). (b) Audio frames *before* the
word change by at most 0.004 under every knockout, so the knockouts do not degrade generation in general; the effect is
on the word. (c) Same result on three seeds. Log: [`sonoedit_heads_ablate.log`](../results/tts_interpretability/sonoedit_heads_ablate.log).

## Step 3: activation steering ([`sonoedit_steering.py`](../scripts/sonoedit_steering.py)): **did not work**

Direction per head (layer 11 head 0; layer 27 heads 3, 4, 6, 7) = mean head output when the word is said correctly
minus when it is said wrongly. A correct reference came from a respelling the model does pronounce correctly (heard as
"postgresql" on 3/3 seeds; [`sonoedit_steering_vectors.log`](../results/tts_interpretability/sonoedit_steering_vectors.log)).

- coefficients 1.5 and 2.0: **0/6 seeds** correct; at 5 and 8: 0/3, and the whole sentence degrades
  ([`steer`](../results/tts_interpretability/sonoedit_steering_steer.log), [`steer_high`](../results/tts_interpretability/sonoedit_steering_steer_high.log)).
- collateral check on sentences without the word: no harm (mean WER 0.065 off vs 0.015 on; small n)
  ([`collateral`](../results/tts_interpretability/sonoedit_steering_collateral.log)).

## What I conclude, and what I don't

- **Supported:** in this model, for this word, the word's identity is read into the audio at layers 11 and 27; a few
  heads carry much of it; the knockout effect is specific to the word, not general damage.
- **Not shown:** that this holds for other words (one word studied in depth; "async" was used for checks only); that
  the heads encode *identity* rather than some correlated feature; that a fix exists. Steering failed.
- **Next controls:** knock out random heads of the same count; a held-out set of 20+ words with the same head set;
  path patching from the layer-11 heads to the layer-27 heads.
