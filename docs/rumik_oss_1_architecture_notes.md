# Rumik-OSS-1 Architecture Notes

Grounded entirely in the actual repo contents at
`code/models/hf_cache/models--rumik-ai--rumik-oss-1/snapshots/0ed3c98684e14350c910129c0efa179841c41ad2/`
(`modeling_rumik_oss.py`, `configuration_rumik_oss.py`, `config.json`,
`tokenizer.json`, `codec/config.json`) plus the installed `transformers==5.2.0`
source for `MimiModel`. Every claim below is either read directly from these
files or computed from values in them — nothing here is inferred from general
LoRA/TTS knowledge without that grounding. Where something genuinely can't be
determined from the code, it's listed as an open question in §5, not guessed.

## 1. The model code

### 1.1 `RumikOSSForCausalLM` — what it actually is

```python
from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM

class RumikOSSForCausalLM(Cohere2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.stop_predictor = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden//4), nn.GELU(), nn.Linear(hidden//4, 1)
        )
```

This is the single most important fact for everything downstream:
**`RumikOSSForCausalLM` is not custom transformer code — it's the stock HF
`Cohere2ForCausalLM` class, subclassed with exactly two additions**: a small
`stop_predictor` head, and two new methods (`generate_audio`,
`audio_tokens_to_codes`) plus a helper (`audio_token_ids`). The attention
blocks, MLP blocks, layer norms, rotary embeddings — all of it is unmodified
Cohere2. `trust_remote_code=True` is needed only to pull in this thin
subclass, not because the transformer internals are bespoke.

### 1.2 `generate_audio()` — mechanically, step by step

```python
def generate_audio(self, input_ids, attention_mask, allowed_ids=None,
                    max_new_tokens=2048, min_new_tokens=8,
                    temperature=0.8, top_k=30, do_sample=True):
```

1. **Text conditioning enters here**: `input_ids`/`attention_mask` are the
   *already-tokenized prompt* — `<text>{SPEAKER}: <description="...">
   {TEXT}<audio>`. The caller (per the model card) builds this with a plain
   `tokenizer(prompt, return_tensors="pt")` call before `generate_audio` is
   ever invoked. Text conditioning is not a separate cross-attention
   pathway — it's just the first N tokens of one flat autoregressive
   sequence, exactly like a normal chat prompt in a causal LM.
2. One forward pass over the full prompt seeds the KV cache (`out =
   self(input_ids=input_ids, ..., use_cache=True)`).
3. **A custom decode loop** (not `model.generate()`/`GenerationMixin`)
   follows, one token at a time:
   - After `min_new_tokens` steps, a **learned stop head** decides whether
     to end: `stop = sigmoid(self.stop_predictor(hidden))`; if `stop > 0.5`
     for the whole batch, it appends `</audio>` and returns. This is a
     genuinely novel mechanism vs. most autoregressive TTS — the model
     decides utterance length itself via a small trained classifier over
     the last hidden state, not by waiting for an EOS token to win the
     softmax naturally.
   - Otherwise, logits are **hard-masked to the audio vocabulary only**
     (`restricted.index_copy_(1, allowed_ids, ...)`, everything else set to
     `-inf`) — the model is architecturally *forced* to emit only unit
     tokens (or `</audio>`) once past `<audio>`, it cannot drift back into
     text vocabulary mid-generation.
   - Below `min_new_tokens` steps, `</audio>` itself is masked out too, so
     the model can't stop instantly.
   - Sampling is a plain temperature + top-k multinomial (`_constrained_sample`,
     deliberately not named `_sample` to avoid silently shadowing
     `GenerationMixin._sample`, per an inline comment — a real gotcha the
     authors hit and left a note about).
   - The loop manually advances the KV cache turn by turn (`past_key_values=cache`),
     i.e. it's a hand-rolled generation loop, not `model.generate(...)`.
4. **Where Mimi enters relative to this file**: nowhere. `generate_audio`
   returns raw token ids only — no codec object is imported, referenced, or
   called anywhere in `modeling_rumik_oss.py`. The codec is entirely the
   caller's responsibility, loaded and invoked separately, confirming the
   model card's usage pattern is not a simplification for the README — it's
   architecturally required.

### 1.3 `audio_tokens_to_codes()` — the arithmetic bridge to Mimi

```python
def audio_tokens_to_codes(self, token_ids) -> torch.Tensor:  # -> [1, 8, T]
```

Converts the flat generated token-id stream into the `[batch, num_quantizers,
frames]` shape `MimiModel.decode` expects. Mechanically:
- For each token, if it's the `</audio>` id, stop.
- If a token falls outside `[first_unit_id, last_unit_id]` (277392 `<text>`
  or `<audio>` sneaking in, or noise), the **current partial frame is
  discarded** and it waits for a fresh `quantizer==0` token to resync.
- Otherwise `code, q = divmod(token_id - first_unit_id, num_quantizers)` —
  this is pure arithmetic, no lookup table, no tokenizer call, no codec call.
- A frame is only appended to the output once all 8 quantizer positions
  (`q == 0..7`) have arrived *in order*; an out-of-order token resets the
  in-progress frame (`frame = [code] if q == 0 else []`).
- **This confirms the earlier Phase-1 spec's "frames of 8, resync on
  quantizer==0, drop incomplete frames" description almost exactly** — that
  wasn't a generic assumption, it's a real, literal read of this function.

## 2. Architecture hyperparameters (from `config.json` + `configuration_rumik_oss.py`)

| Field | Value |
|---|---|
| Total parameters (from `model.safetensors.index.json` metadata) | **3,381,533,697** (~3.38B) |
| Total weight size (bf16) | 6,763,067,394 bytes (~6.76 GB) — matches `dtype: bfloat16` |
| Hidden size | 2048 |
| Layers | 36 |
| Attention heads / KV heads | 16 / 4 (GQA) |
| Head dim | 128 |
| MLP intermediate size | 11008, gated activation (SiLU) |
| Attention pattern | sliding-window (4096) every layer except every 4th (`layer_switch=4`), which is full attention — Cohere2's local/global interleave |
| Position embedding | `rope_gptj`, `rope_theta=50000` |
| Max context | `max_position_embeddings = 8192` |
| Codebooks / codebook size | 8 quantizers × 2048 codes each = 16,384 unit tokens |
| Mimi frame rate | 12.5 Hz → `audio_tokens_per_second = 12.5 × 8 = 100` tokens/sec of audio |
| Vocab size | 277,404 (base tiny-aya-fire 261,008 + 16,384 unit tokens + 3 structure tokens + 9 delivery/vocalization tags) |
| `stop_predictor` | Extra head, **already trained** — present as 6 tensors (`stop_predictor.0/1/3.{weight,bias}`) in the checkpoint, not randomly initialized at load time |
| Speakers (`config.speakers`) | `["Ira", "Aisha", "Siya", "Zoya"]` |

**Attention/MLP module names, read directly from `model.safetensors.index.json`
(layer 0 shown, identical pattern all 36 layers)**:
```
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.input_layernorm.weight
```
These are exactly standard HF Cohere2/Llama-family names — a direct
consequence of §1.1 (no renaming happened; it's genuinely stock Cohere2).

## 3. Tokenizer — special tokens, resolved to actual IDs

Read directly from `tokenizer.json`'s `added_tokens` list (16,419 total added
tokens):

| Token | ID | `special` flag |
|---|---|---|
| `<text>` | 277392 | `False` |
| `<audio>` | 277393 | `False` |
| `</audio>` | 277394 | `False` |
| `<laugh>` | 277401 | `True` |
| `<chuckle>` | 277402 | `True` |
| `<sigh>` | 277403 | `True` |
| `<0_0>` (first unit token) | 261008 | — |
| `<2047_7>` (last unit token) | 277391 | — |

These match `config.json` exactly: `text_start_token_id=277392`,
`audio_start_token_id=277393`, `audio_end_token_id=277394`,
`first_unit_id=261008`, `last_unit_id=277391`. Unit-token layout is
code-major/quantizer-minor as documented in `configuration_rumik_oss.py`'s
docstring: `token_id = first_unit_id + code*num_quantizers + quantizer`.

**Speaker names are NOT special tokens.** `Ira`/`Aisha`/`Siya`/`Zoya` do not
appear anywhere in `added_tokens`. They're plain text, tokenized by the base
(inherited tiny-aya-fire) subword vocabulary like any other word in the
prompt string. This directly answers the question posed: speakers are part
of the ordinary text prompt, not a reserved mechanism.

**A finding not documented in the public README**: IDs 277395–277400 are six
*additional* special tokens — `[happy]`, `[angry]`, `[neutral]`, `[sad]`,
`[whisper]`, `[excited]` — bracket-style, distinct from the free-text
`<description="...">` mechanism the README documents. The NOTICE file's
count ("9 delivery and vocalization tags, ids 277395-277403") matches
exactly (6 bracket tags + `<laugh>`/`<chuckle>`/`<sigh>` = 9), so these are
real, trained, in-vocabulary tokens — just not surfaced in the README's
public "delivery controls" table, which only documents `tone`/`accent`/`pace`
as free text inside `<description="...">`. **Whether `[whisper]`/`[neutral]`
are meant to be used directly in prompts, or are an internal/legacy
mechanism superseded by the free-text description, is not determinable from
code alone — flagged in §5.**

Also present, inherited unchanged from the base `tiny-aya-fire` tokenizer and
**irrelevant to TTS**: agentic/tool-use tokens at 261000–261007
(`<|START_ACTION|>`, `<|START_TOOL_RESULT|>`, `<|START_THINKING|>`, etc.) —
these are Cohere's Command-R-style tool-calling tokens, carried over because
Rumik extended the base vocabulary rather than replacing it. They play no
role in `generate_audio`'s masked-vocabulary decode loop.

## 4. The `codec/` subfolder

`codec/` contains only `config.json` and `preprocessor_config.json` — **no
custom code, no separate weights file bundled in that subfolder**. Per the
model card, weights for it are fetched via `MimiModel.from_pretrained(REPO,
subfolder="codec")`, which pulls the standard `transformers.MimiModel` class
(confirmed: `config.json`'s `"architectures": ["MimiModel"]`, a stock HF
class, not `trust_remote_code`). Config values: `hidden_size=512`, 8 encoder/
decoder transformer layers, **`num_quantizers: 32`** (Mimi's own full RVQ
capacity) — Rumik's LM only ever predicts/uses the **first 8** of Mimi's 32
available quantizers (`config.json`'s top-level `num_quantizers=8`), a
common pattern for LM-driven codec models (semantic + a handful of acoustic
residual levels, not the full high-fidelity residual stack). Also notable:
`"use_streaming": false` and `"use_cache": false` in the codec's *stored*
config — the shipped default is **not** configured for streaming decode
out of the box; streaming, if used, has to be requested by the caller at
call time (see §4.1), not assumed from the checkpoint's config.

**Confirmed frozen at inference**: `model.safetensors.index.json` (the main
LM checkpoint's tensor index) contains **zero** `codec`/`mimi`-prefixed keys
— the codec's weights live in a completely separate checkpoint, loaded into
a separate `MimiModel` instance. Fine-tuning `RumikOSSForCausalLM` (full or
LoRA) touches only the 296 tensors in the main index; **it structurally
cannot touch the codec**, since they're not even the same `nn.Module` tree.
This resolves the question cleanly: fine-tuning never needs to touch the
codec.

### 4.1 Streaming decode — correcting an assumption from the earlier spec

The earlier Phase-1 spec assumed a `decode_step`-named streaming API (from
`rustymimi`/Kyutai's Rust implementation). Checking the actual installed
`transformers==5.2.0` `MimiModel` class directly (not guessed): there is
**no method literally named `decode_step`**. The public streaming-relevant
surface is:
```python
MimiModel.encode(input_values, padding_mask=None, num_quantizers=None,
                  encoder_past_key_values=None, padding_cache=None,
                  use_streaming=None, return_dict=None)
MimiModel.decode(audio_codes, padding_mask=None,
                  decoder_past_key_values=None, return_dict=None)
```
i.e. incremental/streaming decode in this transformers version is done by
calling `decode()` repeatedly, passing forward `decoder_past_key_values`
between calls (the same KV-cache pattern as the LM itself), rather than a
distinct `decode_step` function. `encode` additionally takes an explicit
`use_streaming` flag. **Any Phase-1 implementation should target this real
API (`decode(..., decoder_past_key_values=...)`), not the `rustymimi`-style
naming assumed earlier** — that assumption was reasonable given Kyutai's
other public tooling, but doesn't match what this checkpoint's actual codec
class (stock `transformers.MimiModel`) exposes.

## 5. LoRA-fine-tunable surface

Because `RumikOSSForCausalLM` is unmodified `Cohere2ForCausalLM` plus one
small additive head (§1.1), **the standard LoRA target modules apply
directly, with real names, not assumptions**:

- **Attention**: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`,
  `self_attn.o_proj`
- **MLP**: `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- (Embeddings are tied — `tie_word_embeddings: true` — so LoRA on the
  embedding/LM-head matrix, if wanted, needs `modules_to_save` or careful
  handling since input and output embeddings share the same underlying
  277,404×2048 tensor.)

**PEFT compatibility**: `peft.get_peft_model()` works via `named_modules()`
introspection on whatever `nn.Module` it's handed — it does not care that
the top-level class was loaded through `trust_remote_code`/`auto_map`, only
that submodules have the names given in `target_modules`. Since those names
are literally stock Cohere2 names, **no custom `PeftConfig` wiring or custom
target-module resolution is needed** — `target_modules=["q_proj","k_proj",
"v_proj","o_proj"]` (optionally + the three MLP proj names) will resolve
correctly out of the box. This is a more favorable answer than the default
"custom trust_remote_code models often need manual LoRA wiring" caution —
grounded in the fact that this particular repo's customization is thin.

**Two real, code-grounded gotchas that do need handling, though**:
1. `PeftModel` (what `get_peft_model()` returns) wraps the base model and
   does **not** forward arbitrary custom methods. `generate_audio`,
   `audio_tokens_to_codes`, and `audio_token_ids` are defined on
   `RumikOSSForCausalLM`, not on `Cohere2ForCausalLM` or any PEFT base
   class, so a raw `peft_model.generate_audio(...)` call will fail with
   `AttributeError`. You need either `peft_model.base_model.model.generate_audio(...)`
   (reaching through PEFT's wrapper to the underlying module) or, more
   robustly, `peft_model.merge_and_unload()` after training to get back a
   plain `RumikOSSForCausalLM` instance with LoRA weights merged in, then
   call `generate_audio` normally.
2. The `stop_predictor` head is small (4 tensors, ~hidden×hidden/4 scale)
   and **already trained** — it's not something LoRA typically targets (LoRA
   targets large linear layers where a low-rank update is worthwhile).
   Whether a style/accent LoRA needs to touch it depends entirely on whether
   the adaptation changes typical utterance length/pacing enough to shift
   where "should I stop" decisions land — plausible for a pace-focused
   adaptation, unclear for a pure accent adaptation. If it needs updating,
   full fine-tuning of just that small head (via `modules_to_save` in the
   `PeftConfig`) is more appropriate than trying to LoRA a 4-tensor MLP.

## 6. Training data format — reverse-engineered from the code

**Confirmed feasible: an encode path exists.** The model card's usage
example only shows `mimi.decode(codes)`, but `MimiModel` (verified directly
against the installed `transformers==5.2.0` source, not assumed) also
exposes:
```python
MimiModel.encode(input_values: torch.Tensor,  # (batch, channels, samples), 24kHz per codec/preprocessor_config.json
                  num_quantizers: int | None = None) -> codes  # (batch, num_quantizers, frames)
```
So training-target codes can be produced with `mimi.encode(waveform,
num_quantizers=8)` — the same frozen codec used at inference, just run in
the opposite direction. This confirms end-to-end data prep is possible with
publicly available code (`transformers.MimiModel`), no missing Rumik-specific
encode tooling required.

**What one training example looks like, built entirely from confirmed
pieces above**:

1. Build the prompt string exactly as the model card shows:
   `<text>{SPEAKER}: <description="{DESCRIPTION}"> {TEXT}<audio>`, tokenize
   it → a sequence of ordinary text-vocab token ids, ending in the
   `<audio>` token (id 277393).
2. Take the matching ground-truth audio for that utterance, resample to
   24kHz mono (per `codec/preprocessor_config.json`: `sampling_rate: 24000,
   feature_size: 1`), run `mimi.encode(waveform, num_quantizers=8)` →
   `[1, 8, T]` integer codes.
3. Flatten those codes back into the flat unit-token stream — the exact
   **inverse** of `audio_tokens_to_codes` (§1.3): for each frame `t` in
   order, for each quantizer `q` in `0..7`, emit
   `token_id = first_unit_id + code[q, t] * num_quantizers + q`.
4. Append the `</audio>` token (277394) after the last frame.
5. Concatenate: `[prompt_ids] + [flattened_audio_unit_ids] + [audio_end_id]`
   → one flat sequence, exactly matching the single-autoregressive-sequence
   framing the README describes ("text conditioning and audio generation
   share a single autoregressive sequence").
6. Standard next-token teacher forcing across this flat sequence is the
   obvious training objective for the LM itself.

**What could NOT be determined from the code alone** — these are the real
open questions, not guessable from what's on disk:

1. **Loss masking.** Is cross-entropy computed over the *entire* flat
   sequence (prompt text included), or masked to the audio-token span only
   (`<audio>` onward), the common "instruction masking" pattern for
   decoder-only SFT? Nothing in `modeling_rumik_oss.py` (a pure inference
   file — no training loop, no loss function, no `Trainer` subclass) or the
   config indicates this either way.
2. **`stop_predictor` training.** Is it trained jointly with the LM's
   token-prediction loss (e.g. an auxiliary BCE term at every step, or only
   at the true stop position), or trained as a wholly separate stage/pass?
   The class only defines the head's *architecture*, not how it was
   supervised.
3. **Per-codebook loss weighting.** Are all 8 quantizer positions weighted
   equally in the loss, or is quantizer 0 (plausibly closer to semantic
   content, per typical RVQ hierarchies) upweighted, as some RVQ-LM papers
   do? Not visible anywhere in the shipped code.
4. **Optimizer/schedule.** Learning rate, warmup, batch size, number of
   epochs/steps, and whether LoRA rank/alpha were ever tried internally —
   entirely absent; this is genuinely training-run metadata, not something
   a model card or inference-only modeling file would carry.
5. **Data volume and balance.** The blog states "fewer than 70,000 hours of
   speech" in total, but nothing on disk indicates the per-speaker (Ira/
   Aisha/Siya/Zoya) or per-language (22 languages) split, or how code-switched
   examples were constructed/weighted.
6. **The `[happy]`/`[neutral]`/`[sad]`/`[angry]`/`[excited]`/`[whisper]`
   bracket tokens** (§3) — are these meant to be used directly in prompts
   (e.g. `<text>Ira: [happy] {TEXT}<audio>`) as an alternative/legacy path to
   the documented `<description="tone, accent, pace">` free-text mechanism,
   or purely an internal training-time conditioning signal not intended for
   end-user prompts at all? The README doesn't mention them; the tokenizer
   proves they're real, trained, special tokens.
7. **Whether `generate_audio`'s hand-rolled decode loop (§1.2) is also how
   training-time teacher forcing was framed**, or whether training used
   plain `Cohere2ForCausalLM.forward()` batched cross-entropy over
   pre-built sequences (far more likely for efficiency, but not confirmed).

---

## Questions for research@rumik.ai

Paste-ready — grounded in what we verified above, so these ask for the
genuinely missing pieces rather than things the code already answers:

> Hi — we've been reading through `rumik-oss-1`'s shipped code
> (`modeling_rumik_oss.py`, `configuration_rumik_oss.py`, the tokenizer) to
> plan a LoRA fine-tune on `rumik-oss-1-base` for accent/style adaptation,
> ahead of your training-code release. A few questions we couldn't resolve
> from the checkpoint alone:
>
> 1. **Loss masking**: during SFT, is cross-entropy computed over the full
>    `<text>...<audio>...</audio>` sequence, or masked to the audio-token
>    span only (prompt tokens excluded from the loss)?
> 2. **`stop_predictor` supervision**: is the stop head trained jointly with
>    the token-prediction loss (e.g. per-step auxiliary BCE), or in a
>    separate stage? Any guidance on whether a pace/accent LoRA needs to
>    touch it (e.g. via `modules_to_save`), or whether it transfers fine
>    frozen?
> 3. **Per-codebook loss weighting**: are all 8 quantizer positions weighted
>    equally, or is an earlier quantizer (closer to semantic content)
>    upweighted?
> 4. **Training hyperparameters**: any pointers on learning rate range,
>    effective batch size, and rough step count that worked for post-training
>    at this model size, even approximate, would help us avoid an expensive
>    blind search.
> 5. **The `[happy]`/`[sad]`/`[angry]`/`[excited]`/`[neutral]`/`[whisper]`
>    bracket-style special tokens** (IDs 277395-277400) — we found these in
>    the tokenizer but they're not documented in the model card's delivery-
>    controls table, which only shows the free-text `<description="...">`
>    mechanism. Are these an alternate/legacy conditioning path, or purely
>    internal to training?
> 6. **Per-speaker/language data balance**: roughly how was the ~70,000
>    hours split across the four voices and 22 languages, particularly for
>    code-switched examples — mainly so we can gauge how much fine-tuning
>    data we'd realistically need for a single-speaker style adaptation
>    without destabilizing the other three voices.
>
> Happy to share what we build once your training code lands — thanks for
> shipping this openly.
