"""SonoEdit final attempt: activation steering at the layer-11/27 readout heads.

    python scripts/sonoedit_steering.py respell   [--case postgresql|async]
    python scripts/sonoedit_steering.py vectors   --respell "TEXT" [--case ...]
    python scripts/sonoedit_steering.py steer     [--coeffs 0.5,1,1.5,2] [--seeds 11,12,13,14,15,16]
    python scripts/sonoedit_steering.py collateral --coeff 1.0
    python scripts/sonoedit_steering.py check     --case async   # is a single-mention sentence
                                                                   # actually mispronounced?

NO WEIGHT EDITS ANYWHERE IN THIS FILE. Steering is a forward-hook that adds a
fixed direction to two attention layers' output, only while explicitly
enabled, only at incremental-decoding steps (hidden_states.shape[1] == 1,
which `generate_audio` only reaches after the one-shot prompt prefill --
confirmed by reading `modeling_rumik_oss.py:generate_audio`). The model's
parameters are never written; turning steering off is `hooks.enabled = False`.

WHERE THE DIRECTION COMES FROM. Option 3 (`sonoedit_heads.py`) found the
word's identity is read into the audio at two content-addressed attention
layers: layer 11 (head 0 alone carries ~24-44% of the effect) and layer 27
(heads 3/4/6/7, distributed). Those five heads are `TARGET_HEADS` below. The
steering direction per head is (mean head-output on frames where the model
says the word CORRECTLY) minus (mean head-output on frames where it says the
word WRONG), both from the model's own generations, averaged over the
established seeds. This needs a correctly-pronounced reference, which this
model does not produce for "PostgreSQL" itself -- hence Step 1, finding an
alternate spelling it does pronounce correctly, verified by generation + ASR
exactly like every other claim in this investigation (assumed, not asserted).

CAUTION built into the design, not added after: `steer` always also runs
coeff=0 (hooks off) at the same seeds for a same-run baseline, and
`collateral` tests the trigger-word-derived direction against sentences that
do NOT contain the word, because a steering vector that only helps the target
word and quietly damages everything else is not a fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from sonoedit_trace import FRAME_S, OUT, SEEDS, load_provider  # noqa: E402
from wer_hindi_corrections import edit_counts, normalise  # noqa: E402

TARGET_HEADS = {11: [0], 27: [3, 4, 6, 7]}   # from sonoedit_heads.py Option 3

CASES = {
    "postgresql": dict(
        word="PostgreSQL",
        text="The team migrated the database from MySQL to PostgreSQL last week.",
        anchor_contains="sql", anchor_after="to", anchor_before="last",
        respell_candidates=[
            "Postgres sequel", "Postgres S Q L", "Postgre S Q L",
            "Post-gres sequel", "Poss-gres sequel", "Postgres SQL database",
        ],
    ),
    "async": dict(
        word="async",
        text="The function was refactored so every network call now runs in async mode instead of blocking.",
        anchor_contains="network", anchor_after="in", anchor_before="mode",
        respell_candidates=["a-sink", "ay-sink", "eh-sink mode", "asink"],
    ),
}

UNRELATED_SENTENCES = [
    "The weather today is sunny with a light breeze from the west.",
    "She finished her coffee and walked out into the busy morning street.",
    "Our quarterly meeting has been moved to Thursday afternoon.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "He practiced the guitar every evening after dinner.",
]

_ASR = None


def asr_model():
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel

        _ASR = WhisperModel("small", device="cpu", compute_type="int8")
    return _ASR


def transcribe(wav_path) -> dict:
    segs, _ = asr_model().transcribe(str(wav_path), language="en", beam_size=5, word_timestamps=True)
    words = [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]
    return {"words": words, "text": " ".join(w for w, _, _ in words)}


def find_span(words: list[tuple[str, float, float]], case: dict) -> dict | None:
    """Generalized version of sonoedit_trace.align_runs: anchor on a word
    containing `anchor_contains`, then the next `anchor_after`, then the span
    up to the next `anchor_before`. Works regardless of what's actually said
    in between, so it applies unchanged to a respelling."""
    clean = [w.strip(".,-").lower() for w, _, _ in words]
    try:
        i_anchor = next(i for i, w in enumerate(clean) if case["anchor_contains"] in w)
        i_after = next(i for i in range(i_anchor + 1, len(clean)) if clean[i] == case["anchor_after"])
        i_before = next(i for i in range(i_after + 1, len(clean)) if clean[i] == case["anchor_before"])
    except StopIteration:
        return None
    span = words[i_after + 1:i_before]
    if not span:
        return None
    t_s, t_e = span[0][1], span[-1][2]
    return {"heard": " ".join(w for w, _, _ in span), "start_s": t_s, "end_s": t_e,
            "frame_start": int(t_s / FRAME_S), "frame_end": int(np.ceil(t_e / FRAME_S))}


def gen_and_align(p, text: str, seed: int, case: dict, out_prefix: str):
    """Generate audio for `text` at `seed`, decode, transcribe, locate the
    word span. Returns None if alignment fails (caller must handle)."""
    import torch

    torch.manual_seed(seed)
    cfg = p._model.config
    first, last, q_n = int(cfg.first_unit_id), int(cfg.last_unit_id), int(cfg.num_quantizers)
    prompt = p._build_prompt(text)
    enc = p._tokenizer(prompt, return_tensors="pt").to(p._model.device)
    ids_out = p._model.generate_audio(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                      max_new_tokens=p.max_new_tokens, temperature=p.temperature,
                                      top_k=p.top_k, do_sample=True)
    gen = ids_out[0].tolist()[enc.input_ids.shape[1]:]
    units = []
    for t in gen:
        if not first <= t <= last:
            break
        units.append(t)
    units = units[: len(units) // q_n * q_n]
    codes = p._model.audio_tokens_to_codes(units)
    wav = p._decode(codes)
    wav_path = OUT / f"{out_prefix}_seed{seed}.wav"
    sf.write(str(wav_path), wav, 24000)
    tr = transcribe(wav_path)
    span = find_span(tr["words"], case)
    n_frames = len(units) // q_n
    return {"wav": str(wav_path), "asr_text": tr["text"], "span": span,
            "prompt_len": int(enc.input_ids.shape[1]), "units": units, "n_frames": n_frames, "q_n": q_n}


# ------------------------------------------------------------------ respell ----

def stage_respell(case_name: str, seeds: list[int]) -> int:
    case = CASES[case_name]
    p = load_provider()
    results = {}
    for cand in case["respell_candidates"]:
        text = case["text"].replace(case["word"], cand)
        rows = []
        for seed in seeds:
            r = gen_and_align(p, text, seed, case, f"respell_{case_name}_{cand[:12].replace(' ', '_')}")
            heard = (r["span"] or {}).get("heard", "").lower()
            success = case_name == "postgresql" and "postgres" in heard
            if case_name == "async":
                success = "sink" in heard or "async" in heard
            rows.append({"seed": seed, "heard": heard or None, "asr_text": r["asr_text"], "success": success})
            print(f"  {cand!r} seed {seed}: heard={heard!r} success={success}", flush=True)
        results[cand] = rows
        n_ok = sum(row["success"] for row in rows)
        print(f"{cand!r}: {n_ok}/{len(rows)} seeds sound right\n", flush=True)
    (OUT / f"steering_respell_{case_name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("Pick the candidate with the most consistent success across seeds for `vectors --respell`.")
    return 0


# ------------------------------------------------------------------ vectors ----

class HeadCapture:
    """Exact recompute of layers 11/27 (see sonoedit_heads.HeadLab for the
    verified math), capturing the pre-o_proj per-head output instead of
    intervening. No weights are touched."""

    def __init__(self, provider, layers=None):
        layers = list(TARGET_HEADS) if layers is None else layers
        import torch

        self.torch = torch
        self.model = provider._model
        cfg = self.model.config
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.captured: dict[int, object] = {}
        self.enabled = False   # only recompute (no-cache) attention while True -- see enabled_for_pass()
        self.originals = {}
        for L in layers:
            attn = self.model.model.layers[L].self_attn
            self.originals[L] = attn.forward
            attn.forward = self._make_forward(L, attn)

    def _make_forward(self, L, attn):
        torch = self.torch
        orig = self.originals[L]

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
            if not self.enabled:
                # generate_audio decodes with a KV cache; this recompute path
                # ignores past_key_values entirely, so it must stay off during
                # generation and only turn on for the explicit teacher-forced
                # capture pass below, or it silently corrupts generation.
                return orig(hidden_states, position_embeddings=position_embeddings,
                           attention_mask=attention_mask, past_key_values=past_key_values,
                           cache_position=cache_position, **kwargs)
            B, T, _ = hidden_states.shape
            H, K, D = self.n_heads, self.n_kv, self.head_dim
            q = attn.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2).float()
            k = attn.k_proj(hidden_states).view(B, T, K, D).transpose(1, 2).float()
            v = attn.v_proj(hidden_states).view(B, T, K, D).transpose(1, 2).float()
            k = k.repeat_interleave(H // K, dim=1)
            v = v.repeat_interleave(H // K, dim=1)
            scores = (q @ k.transpose(2, 3)) * attn.scaling
            causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            per_head = weights @ v                              # [B, H, T, D]
            self.captured[L] = per_head.detach()
            out = per_head.transpose(1, 2).reshape(B, T, H * D).to(hidden_states.dtype)
            return attn.o_proj(out), None

        return forward

    def restore(self):
        for L, fn in self.originals.items():
            self.model.model.layers[L].self_attn.forward = fn


def capture_word_vectors(p, capture: HeadCapture, run: dict) -> dict[tuple[int, int], object]:
    """Run a teacher-forced pass over `run`'s own generated sequence and
    average each target head's pre-o_proj output over the word's frames."""
    import torch

    span = run["span"]
    word_frames = list(range(span["frame_start"], min(span["frame_end"], run["n_frames"])))
    positions = [run["prompt_len"] + f * run["q_n"] + q for f in word_frames for q in range(run["q_n"])]
    # `run["prompt_ids"]` is set by gen_and_align_with_ids so we don't re-tokenize the prompt here.
    ids = torch.tensor([run["prompt_ids"] + run["units"]], device=p._model.device)
    mask = torch.ones_like(ids)
    embeds = p._model.model.embed_tokens(ids)
    capture.captured = {}
    capture.enabled = True
    try:
        with torch.inference_mode():
            p._model.model(inputs_embeds=embeds, attention_mask=mask)
    finally:
        capture.enabled = False
    out = {}
    for L, heads in TARGET_HEADS.items():
        per_head = capture.captured[L][0]                        # [H, T, D]
        for h in heads:
            out[(L, h)] = per_head[h, positions, :].mean(0).cpu().numpy()
    return out


def gen_and_align_with_ids(p, text, seed, case, prefix):
    r = gen_and_align(p, text, seed, case, prefix)
    prompt = p._build_prompt(text)
    r["prompt_ids"] = p._tokenizer(prompt)["input_ids"]
    return r


def stage_vectors(case_name: str, respell_text: str, seeds: list[int]) -> int:
    case = CASES[case_name]
    p = load_provider()
    capture = HeadCapture(p)
    diffs = {k: [] for L, hs in TARGET_HEADS.items() for k in [(L, h) for h in hs]}
    per_seed = {}
    for seed in seeds:
        wrong = gen_and_align_with_ids(p, case["text"], seed, case, f"vec_wrong_{case_name}")
        right = gen_and_align_with_ids(p, case["text"].replace(case["word"], respell_text), seed, case,
                                       f"vec_right_{case_name}")
        if not wrong["span"] or not right["span"]:
            print(f"seed {seed}: alignment failed (wrong={bool(wrong['span'])}, right={bool(right['span'])}), skipping",
                  flush=True)
            continue
        heard_wrong = wrong["span"]["heard"].lower()
        heard_right = right["span"]["heard"].lower()
        print(f"seed {seed}: wrong-text heard {heard_wrong!r} | respelled heard {heard_right!r}", flush=True)
        v_wrong = capture_word_vectors(p, capture, wrong)
        v_right = capture_word_vectors(p, capture, right)
        for k in diffs:
            diffs[k].append(v_right[k] - v_wrong[k])
        per_seed[seed] = {"heard_wrong": heard_wrong, "heard_right": heard_right}
    capture.restore()
    vectors = {f"{L}|{h}": np.mean(diffs[(L, h)], axis=0).tolist() for L, hs in TARGET_HEADS.items() for h in hs
              if diffs[(L, h)]}
    norms = {k: float(np.linalg.norm(v)) for k, v in vectors.items()}
    (OUT / f"steering_vectors_{case_name}.json").write_text(
        json.dumps({"case": case_name, "respell_text": respell_text, "seeds_used": list(per_seed),
                   "per_seed": per_seed, "vectors": vectors, "norms": norms}, indent=1), encoding="utf-8")
    print("head norms:", norms)
    return 0


# -------------------------------------------------------------------- steer ----

class SteeringHooks:
    """Adds a fixed hidden-size delta to layers 11/27's attention output,
    only while `enabled` and only on single-token incremental-decode steps
    (never on the prompt prefill). Pure runtime hook; restore() undoes it."""

    def __init__(self, provider, vectors: dict[str, list[float]], coeff: float):
        import torch

        self.torch = torch
        self.model = provider._model
        self.coeff = coeff
        self.enabled = False
        self.originals = {}
        by_layer: dict[int, dict[int, list[float]]] = {}
        for key, vec in vectors.items():
            L, h = (int(x) for x in key.split("|"))
            by_layer.setdefault(L, {})[h] = vec
        for L, heads in by_layer.items():
            attn = self.model.model.layers[L].self_attn
            self.originals[L] = attn.forward
            self._wrap(L, attn, heads)

    def _wrap(self, L, attn, heads):
        torch = self.torch
        orig = self.originals[L]
        W_o = attn.o_proj.weight.detach().float()
        head_dim = int(getattr(self.model.config, "head_dim",
                              self.model.config.hidden_size // self.model.config.num_attention_heads))
        delta = torch.zeros(W_o.shape[0], dtype=torch.float32, device=W_o.device)
        for h, vec in heads.items():
            v = torch.tensor(vec, dtype=torch.float32, device=W_o.device)
            Wh = W_o[:, h * head_dim:(h + 1) * head_dim]
            delta = delta + Wh @ v
        self._delta = getattr(self, "_delta", {})
        self._delta[L] = delta

        def forward(hidden_states, *args, **kwargs):
            out = orig(hidden_states, *args, **kwargs)
            if self.enabled and hidden_states.shape[1] == 1:
                attn_out = out[0] if isinstance(out, tuple) else out
                attn_out = attn_out + (self.coeff * self._delta[L]).to(attn_out.dtype)
                return (attn_out, *out[1:]) if isinstance(out, tuple) else attn_out
            return out

        attn.forward = forward

    def restore(self):
        for L, fn in self.originals.items():
            self.model.model.layers[L].self_attn.forward = fn


def pronounced_correctly(asr_text: str, word: str) -> bool:
    clean = [w.strip(".,").lower() for w in asr_text.split()]
    return word.lower() in clean


def stage_steer(case_name: str, coeffs: list[float], seeds: list[int]) -> int:
    case = CASES[case_name]
    vectors = json.loads((OUT / f"steering_vectors_{case_name}.json").read_text(encoding="utf-8"))["vectors"]
    p = load_provider()
    results = {}
    for coeff in [0.0, *coeffs]:
        hooks = SteeringHooks(p, vectors, coeff)
        rows = []
        for seed in seeds:
            hooks.enabled = coeff != 0.0
            r = gen_and_align(p, case["text"], seed, case, f"steer_{case_name}_c{coeff}")
            ok = pronounced_correctly(r["asr_text"], case["word"])
            rows.append({"seed": seed, "asr_text": r["asr_text"], "pronounced_correctly": ok})
            print(f"coeff {coeff} seed {seed}: correct={ok} | {r['asr_text']}", flush=True)
        hooks.restore()
        n_ok = sum(row["pronounced_correctly"] for row in rows)
        results[str(coeff)] = {"rows": rows, "n_correct": n_ok, "n_total": len(rows)}
        print(f"coeff {coeff}: {n_ok}/{len(rows)} correct\n", flush=True)
        (OUT / f"steering_steer_{case_name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------- collateral ----

def stage_collateral(case_name: str, coeff: float, seeds: list[int]) -> int:
    vectors = json.loads((OUT / f"steering_vectors_{case_name}.json").read_text(encoding="utf-8"))["vectors"]
    p = load_provider()
    hooks = SteeringHooks(p, vectors, coeff)
    results = []
    for text in UNRELATED_SENTENCES:
        for seed in seeds:
            for enabled in (False, True):
                import torch

                torch.manual_seed(seed)
                hooks.enabled = enabled
                prompt = p._build_prompt(text)
                enc = p._tokenizer(prompt, return_tensors="pt").to(p._model.device)
                ids_out = p._model.generate_audio(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                                  max_new_tokens=p.max_new_tokens, temperature=p.temperature,
                                                  top_k=p.top_k, do_sample=True)
                cfg = p._model.config
                first, last, q_n = int(cfg.first_unit_id), int(cfg.last_unit_id), int(cfg.num_quantizers)
                gen = ids_out[0].tolist()[enc.input_ids.shape[1]:]
                units = []
                for t in gen:
                    if not first <= t <= last:
                        break
                    units.append(t)
                units = units[: len(units) // q_n * q_n]
                codes = p._model.audio_tokens_to_codes(units)
                wav = p._decode(codes)
                tag = "on" if enabled else "off"
                wav_path = OUT / f"collateral_{case_name}_{tag}_seed{seed}_{abs(hash(text)) % 10000}.wav"
                sf.write(str(wav_path), wav, 24000)
                tr = transcribe(wav_path)
                ref, hyp = normalise(text), normalise(tr["text"])
                s, d, i = edit_counts(ref, hyp)
                wer = round((s + d + i) / max(len(ref), 1), 3)
                results.append({"text": text, "seed": seed, "steering": tag, "wer": wer, "asr_text": tr["text"]})
                print(f"[{tag}] seed {seed} wer={wer} | {text[:40]}... -> {tr['text']}", flush=True)
    hooks.restore()
    (OUT / f"steering_collateral_{case_name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    by_tag = {"off": [r["wer"] for r in results if r["steering"] == "off"],
             "on": [r["wer"] for r in results if r["steering"] == "on"]}
    print(f"\nmean WER off={np.mean(by_tag['off']):.3f}  on={np.mean(by_tag['on']):.3f}")
    return 0


# ------------------------------------------------------------------- check ----

def stage_check(case_name: str, seeds: list[int]) -> int:
    """Is this case's single-mention TEXT actually mispronounced? (Needed for
    async: the earlier check used a double-mention sentence and found it
    inconsistent, not confirmed failing.)"""
    case = CASES[case_name]
    p = load_provider()
    rows = []
    for seed in seeds:
        r = gen_and_align(p, case["text"], seed, case, f"check_{case_name}")
        heard = (r["span"] or {}).get("heard", "").lower()
        correct = heard == case["word"].lower()
        rows.append({"seed": seed, "heard": heard or None, "asr_text": r["asr_text"], "correct": correct})
        print(f"seed {seed}: heard={heard!r} correct={correct} | {r['asr_text']}", flush=True)
    (OUT / f"steering_check_{case_name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    n_wrong = sum(not row["correct"] for row in rows)
    print(f"\n{n_wrong}/{len(rows)} seeds mispronounced -- "
          f"{'CONFIRMED failing, safe to use as case 2' if n_wrong == len(rows) else 'NOT consistently wrong, do not treat as confirmed without a human listen'}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["respell", "vectors", "steer", "collateral", "check"])
    ap.add_argument("--case", default="postgresql", choices=list(CASES))
    ap.add_argument("--respell", help="the chosen correctly-pronounced respelling (stage vectors)")
    ap.add_argument("--coeffs", default="0.5,1,1.5,2")
    ap.add_argument("--coeff", type=float, default=1.0)
    ap.add_argument("--seeds", default=None, help="comma list, default varies by stage")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None

    if a.stage == "respell":
        return stage_respell(a.case, seeds or SEEDS)
    if a.stage == "vectors":
        if not a.respell:
            raise SystemExit("--respell TEXT is required for stage vectors")
        return stage_vectors(a.case, a.respell, seeds or SEEDS)
    if a.stage == "steer":
        coeffs = [float(c) for c in a.coeffs.split(",")]
        return stage_steer(a.case, coeffs, seeds or [*SEEDS, 14, 15, 16])
    if a.stage == "collateral":
        return stage_collateral(a.case, a.coeff, seeds or [11])
    return stage_check(a.case, seeds or SEEDS)


if __name__ == "__main__":
    sys.exit(main())
