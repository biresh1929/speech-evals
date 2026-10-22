"""Acoustic causal tracing on Rumik-OSS-1 (SonoEdit Phase 1, adapted).

    python scripts/sonoedit_trace.py generate          # seeded generations + word alignment
    python scripts/sonoedit_trace.py realign           # redo word alignment only (CPU)
    python scripts/sonoedit_trace.py effect            # go/no-go: does corrupting the word matter?
    python scripts/sonoedit_trace.py sweep [--noise N] # restore layer x component, checkpointed

See SONOEDIT_ADAPTATION_NOTES.md for why each choice was made. Summary:

TARGET WORD: "PostgreSQL" -- the only one of three candidates confirmed
mispronounced by generation + ASR (heard as "PaktiJoy SQL" both times).

WHAT IS TRACED (a deliberate, documented deviation from the paper). SonoEdit
measures recovery of the *correct* coarse token. That needs a correct token at
the same sequence position, which Rumik cannot provide: it mispronounces this
word, has no phoneme input, and a respelled prompt has different token
positions, so hidden states cannot be patched across the two. Instead this
traces the model's OWN generated tokens for the word: corrupt the input
embeddings of the word's text tokens, measure how much that lowers the
probability of the audio tokens generated for the word, then restore one
layer/component at a time and measure recovery. That localises where the
text -> pronunciation mapping for this word is computed, which is where an
edit would have to act; it does not show where a *correct* pronunciation would live.

MEASUREMENT: one teacher-forced forward pass over prompt + generated audio
tokens gives every position's prediction at once. Probabilities are a softmax
over the audio-unit vocabulary only (what generate_audio can sample), at
temperature 1. Reported per quantizer 0..7, for the frames where the word is
spoken (from Whisper word timestamps, 80 ms per Mimi frame) and for control
frames elsewhere in the utterance.

RESTORATION (Rumik's parallel blocks): the decoder layer's output (the combined
residual after attn+MLP), the attention output alone, or the MLP output alone,
patched in from the clean run at the word's text-token positions. Attention and
MLP use a 5-layer window (ROME uses 10 of 48 because single-layer component
effects are small). Layer type (sliding/full attention) is recorded.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from longform_trials import LOCAL_PATH  # noqa: E402

OUT = Path(DATA_ROOT + r"/rumik_samples/sonoedit")
WORD = "PostgreSQL"
TEXT = "The team migrated the database from MySQL to PostgreSQL last week."
SEEDS = [11, 12, 13]
FRAME_S = 1920 / 24000        # one Mimi frame = 80 ms
NOISE_MULT = 3.0              # ROME: 3x the std of token embeddings
WINDOW = 5                    # layers restored together for attn / mlp


def load_provider():
    from voicenarrator.providers.tts.rumik import RumikProvider

    p = RumikProvider(model_path=LOCAL_PATH, vram_mode="codec_cpu", speaker="Ira",
                      description="clear, moderate pace, neutral tone")
    p._load()
    return p


# --------------------------------------------------------------- generate ----

def stage_generate() -> int:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    p = load_provider()
    cfg = p._model.config
    first, last, q_n = int(cfg.first_unit_id), int(cfg.last_unit_id), int(cfg.num_quantizers)
    prompt = p._build_prompt(TEXT)
    enc = p._tokenizer(prompt, return_tensors="pt").to(p._model.device)

    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        ids = p._model.generate_audio(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                      max_new_tokens=p.max_new_tokens, temperature=p.temperature,
                                      top_k=p.top_k, do_sample=True)
        gen = ids[0].tolist()[enc.input_ids.shape[1]:]
        units = []
        for t in gen:
            if not first <= t <= last:
                break
            units.append(t)
        units = units[: len(units) // q_n * q_n]
        round_robin_ok = all((t - first) % q_n == j % q_n for j, t in enumerate(units))
        codes = p._model.audio_tokens_to_codes(units)
        wav = p._decode(codes)
        wav_path = OUT / f"seed{seed}.wav"
        sf.write(str(wav_path), wav, 24000)
        runs.append({"seed": seed, "units": units, "round_robin_ok": round_robin_ok,
                     "frames": len(units) // q_n, "audio_seconds": round(len(wav) / 24000, 2),
                     "wav": str(wav_path), "gen_seconds": round(time.perf_counter() - t0, 1)})
        print(f"seed {seed}: {runs[-1]['frames']} frames, round robin ok={round_robin_ok}, "
              f"{runs[-1]['audio_seconds']}s in {runs[-1]['gen_seconds']}s", flush=True)

    align_runs(runs)

    (OUT / "generations.json").write_text(json.dumps({"text": TEXT, "prompt": prompt, "runs": runs},
                                                     indent=1), encoding="utf-8")
    wrong = [r["seed"] for r in runs if r.get("alignment") and not r["pronounced_correctly"]
             and r["round_robin_ok"]]
    print(f"\nmispronounced + aligned + clean round robin: seeds {wrong} of {SEEDS}")
    return 0


def align_runs(runs: list[dict]) -> None:
    """Whisper word timestamps -> the frame span where the target word is spoken."""
    from faster_whisper import WhisperModel

    asr = WhisperModel("small", device="cpu", compute_type="int8")
    for r in runs:
        segs, _ = asr.transcribe(r["wav"], language="en", beam_size=5, word_timestamps=True)
        words = [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]
        r["asr_text"] = " ".join(w for w, _, _ in words)
        clean = [w.strip(".,").lower() for w, _, _ in words]
        r["pronounced_correctly"] = WORD.lower() in clean
        # The word sits between "MySQL to" and "last". Anchor on the "to" right
        # after MySQL: the mispronunciation itself can contain "to" (seed 13 was
        # heard as "pox to grow iSQL"), so "the last 'to'" clips the span.
        try:
            i_my = next(i for i, w in enumerate(clean) if "sql" in w)
            i_to = next(i for i in range(i_my + 1, len(clean)) if clean[i] == "to")
            i_last = next(i for i in range(i_to + 1, len(clean)) if clean[i] == "last")
        except StopIteration:
            r["alignment"] = None
            print(f"seed {r['seed']}: could not align ({r['asr_text']})", flush=True)
            continue
        span = words[i_to + 1:i_last]
        if not span:
            r["alignment"] = None
            continue
        t_s, t_e = span[0][1], span[-1][2]
        r["alignment"] = {"heard_as": " ".join(w for w, _, _ in span), "start_s": t_s, "end_s": t_e,
                          "frame_start": int(t_s / FRAME_S), "frame_end": int(np.ceil(t_e / FRAME_S))}
        print(f"seed {r['seed']}: heard '{r['alignment']['heard_as']}' {t_s:.2f}-{t_e:.2f}s "
              f"(frames {r['alignment']['frame_start']}-{r['alignment']['frame_end']}), "
              f"correct={r['pronounced_correctly']} | {r['asr_text']}", flush=True)


def stage_realign() -> int:
    path = OUT / "generations.json"
    gens = json.loads(path.read_text(encoding="utf-8"))
    align_runs(gens["runs"])
    path.write_text(json.dumps(gens, indent=1), encoding="utf-8")
    return 0


# ----------------------------------------------------------------- tracing ----

class Tracer:
    """Hooks on every layer: capture clean outputs, or patch them into a corrupted run."""

    COMPONENTS = ("layer", "attn", "mlp")

    def __init__(self, provider, positions: list[int]):
        import torch

        self.torch = torch
        self.model = provider._model
        self.layers = self.model.model.layers
        self.positions = positions
        self.mode = "off"                    # off | capture | patch
        self.cache: dict[tuple[int, str], object] = {}
        self.patch: set[tuple[int, str]] = set()
        self.patch_positions = positions
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(i, "layer")))
            self.handles.append(layer.self_attn.register_forward_hook(self._hook(i, "attn")))
            self.handles.append(layer.mlp.register_forward_hook(self._hook(i, "mlp")))

    def _hook(self, i: int, comp: str):
        def fn(_module, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.cache[(i, comp)] = tensor[:, self.positions].detach().clone()
                return None
            if self.mode == "patch" and (i, comp) in self.patch:
                idx = [self.positions.index(p) for p in self.patch_positions]
                new = tensor.clone()
                new[:, self.patch_positions] = self.cache[(i, comp)][:, idx]
                return (new, *out[1:]) if isinstance(out, tuple) else new
            return None
        return fn


def build_sequence(provider, gens: dict, seed: int):
    import torch

    run = next(r for r in gens["runs"] if r["seed"] == seed)
    enc = provider._tokenizer(gens["prompt"], return_offsets_mapping=True)
    prompt_ids = enc["input_ids"]
    c0 = gens["prompt"].index(WORD)
    c1 = c0 + len(WORD)
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > c0 and a < c1 and b > a]
    ids = torch.tensor([prompt_ids + run["units"]], device=provider._model.device)
    return ids, len(prompt_ids), span, run


def prob_table(provider, hidden, pred_positions: dict[str, list[int]], ids, allowed, first, q_n):
    """{frame_set: [mean prob of the generated token at quantizer q] for q in 0..7}."""
    torch = __import__("torch")
    model = provider._model
    out = {}
    with torch.inference_mode():
        w = model.lm_head.weight[allowed]
        return _prob_table(model, w, hidden, pred_positions, ids, first, q_n, out)


def _prob_table(model, w, hidden, pred_positions, ids, first, q_n, out):
    torch = __import__("torch")
    for name, positions in pred_positions.items():
        h = hidden[0, [p - 1 for p in positions]]
        logits = (h.float() @ w.float().T) * float(model.logit_scale)
        logp = torch.log_softmax(logits, dim=-1)
        tokens = ids[0, positions]
        col = tokens - first            # allowed = first..last then </audio>; unit index == column
        p = logp[torch.arange(len(positions)), col].exp().cpu().numpy()
        qs = [(int(t) - first) % q_n for t in tokens.tolist()]
        out[name] = [float(np.mean([p[k] for k in range(len(p)) if qs[k] == q])) for q in range(q_n)]
    return out


def run_pass(provider, embeds, mask):
    torch = __import__("torch")
    with torch.inference_mode():
        return provider._model.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state


def prepare(seed: int | None, p=None):
    import torch

    gens = json.loads((OUT / "generations.json").read_text(encoding="utf-8"))
    good = [r["seed"] for r in gens["runs"] if r.get("alignment") and not r["pronounced_correctly"]
            and r["round_robin_ok"]]
    if not good:
        raise SystemExit("no mispronounced, aligned generation to trace -- run `generate` again")
    seed = seed or good[0]
    p = p or load_provider()
    cfg = p._model.config
    first, q_n = int(cfg.first_unit_id), int(cfg.num_quantizers)
    ids, plen, span, run = build_sequence(p, gens, seed)
    al = run["alignment"]
    n_frames = run["frames"]
    word_frames = list(range(al["frame_start"], min(al["frame_end"], n_frames)))
    # Controls: an equal number of frames well before and well after the word.
    k = len(word_frames)
    before = [f for f in range(2, al["frame_start"] - 4)][-k:]
    after = [f for f in range(al["frame_end"] + 4, n_frames - 2)][:k]
    pred = {name: [plen + f * q_n + q for f in frames for q in range(q_n)]
            for name, frames in (("word", word_frames), ("before", before), ("after", after)) if frames}
    allowed = torch.arange(first, int(cfg.last_unit_id) + 1, device=ids.device)
    mask = torch.ones_like(ids)
    emb = p._model.model.embed_tokens
    with torch.inference_mode():
        clean_embeds = emb(ids)
        text_rows = emb.weight[torch.randint(0, first, (20000,), device=ids.device)].float()
        noise_scale = NOISE_MULT * float(text_rows.std())
    meta = {"seed": seed, "prompt_tokens": plen, "word_token_positions": span,
            "word_tokens": [p._tokenizer.decode([ids[0, i].item()]) for i in span],
            "frames": {"word": word_frames, "before": before, "after": after},
            "alignment": al, "noise_scale": noise_scale, "sequence_length": int(ids.shape[1])}
    return p, ids, mask, clean_embeds, span, pred, allowed, first, q_n, meta


def corrupt(clean_embeds, span, noise_scale, noise_seed):
    torch = __import__("torch")
    g = torch.Generator(device=clean_embeds.device).manual_seed(noise_seed)
    e = clean_embeds.clone()
    noise = torch.randn(e[:, span].shape, generator=g, device=e.device, dtype=torch.float32)
    e[:, span] = e[:, span] + (noise * noise_scale).to(e.dtype)
    return e


def stage_effect(seed: int | None, noise_n: int) -> int:
    p, ids, mask, clean_embeds, span, pred, allowed, first, q_n, meta = prepare(seed)
    torch = __import__("torch")
    print(f"seed {meta['seed']}: seq {meta['sequence_length']} tokens, word tokens "
          f"{meta['word_tokens']} at {span}, frames {meta['frames']['word']}", flush=True)

    # Sanity: the inputs_embeds + manual lm_head path must match the model's own logits.
    with torch.inference_mode():
        full = p._model(input_ids=ids, attention_mask=mask).logits[0, pred["word"][0] - 1].float()
        mine = (run_pass(p, clean_embeds, mask)[0, pred["word"][0] - 1].float()
                @ p._model.lm_head.weight.float().T) * float(p._model.logit_scale)
    sanity = float((full - mine).abs().max())
    print(f"sanity: max |logit diff| manual vs model = {sanity:.4f}", flush=True)

    t0 = time.perf_counter()
    clean = prob_table(p, run_pass(p, clean_embeds, mask), pred, ids, allowed, first, q_n)
    pass_s = time.perf_counter() - t0
    corrupted = []
    for n in range(noise_n):
        h = run_pass(p, corrupt(clean_embeds, span, meta["noise_scale"], 1000 + n), mask)
        corrupted.append(prob_table(p, h, pred, ids, allowed, first, q_n))
    corr = {k: np.mean([c[k] for c in corrupted], axis=0).tolist() for k in clean}
    result = {**meta, "sanity_max_logit_diff": sanity, "seconds_per_pass": round(pass_s, 2),
              "clean": clean, "corrupted_mean": corr, "noise_samples": noise_n}
    (OUT / "effect.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{'':8}" + "".join(f"   q{q}  " for q in range(q_n)))
    for k in clean:
        print(f"{k:<8}" + "".join(f"{clean[k][q]:.3f}->{corr[k][q]:.3f} "[:8] + " " for q in range(q_n)))
        print(f"{'  drop':<8}" + "".join(f"{clean[k][q] - corr[k][q]:+.3f}  " for q in range(q_n)))
    print(f"\n{pass_s:.1f}s per forward pass")
    return 0


def stage_sweep(seed: int | None, noise_n: int, only: set[str] | None = None) -> int:
    p, ids, mask, clean_embeds, span, pred, allowed, first, q_n, meta = prepare(seed)
    n_layers = len(p._model.model.layers)
    layer_types = list(p._model.config.layer_types)
    tracer = Tracer(p, span)
    tracer.mode = "capture"
    clean = prob_table(p, run_pass(p, clean_embeds, mask), pred, ids, allowed, first, q_n)
    tracer.mode = "off"
    noisy = [corrupt(clean_embeds, span, meta["noise_scale"], 1000 + n) for n in range(noise_n)]
    corr = [prob_table(p, run_pass(p, e, mask), pred, ids, allowed, first, q_n) for e in noisy]
    corr_mean = {k: np.mean([c[k] for c in corr], axis=0).tolist() for k in clean}

    ckpt = OUT / f"sweep_seed{meta['seed']}.json"
    done = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.is_file() else {}
    results = done.get("results", {})
    conditions = [(comp, L, "span") for comp in ("layer", "attn", "mlp") for L in range(n_layers)]
    conditions += [("layer", L, "last_token") for L in range(n_layers)]
    if only:
        conditions = [c for c in conditions if f"{c[0]}@{c[2]}" in only]
    t0 = time.perf_counter()
    for i, (comp, L, where) in enumerate(conditions):
        key = f"{comp}|{L}|{where}"
        if key in results:
            continue
        if comp == "layer":
            layers = [L]
        else:
            layers = list(range(max(0, L - WINDOW // 2), min(n_layers, L + WINDOW // 2 + 1)))
        tracer.patch = {(j, comp) for j in layers}
        tracer.patch_positions = span if where == "span" else [span[-1]]
        tracer.mode = "patch"
        tables = [prob_table(p, run_pass(p, e, mask), pred, ids, allowed, first, q_n) for e in noisy]
        tracer.mode = "off"
        results[key] = {k: np.mean([t[k] for t in tables], axis=0).tolist() for k in clean}
        if i % 6 == 5 or i == len(conditions) - 1:
            ckpt.write_text(json.dumps({**meta, "noise_samples": noise_n, "window": WINDOW,
                                        "layer_types": layer_types, "clean": clean,
                                        "corrupted_mean": corr_mean, "results": results}, indent=1),
                            encoding="utf-8")
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"{len(results)}/{len(conditions)} conditions, ~{rate * (len(conditions) - i - 1) / 60:.0f} "
                  f"min left", flush=True)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "realign", "effect", "sweep"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--noise", type=int, default=3)
    ap.add_argument("--only", help="comma list of conditions, e.g. layer@span,mlp@span (sweep only)")
    a = ap.parse_args()
    if a.stage == "generate":
        return stage_generate()
    if a.stage == "realign":
        return stage_realign()
    if a.stage == "effect":
        return stage_effect(a.seed, a.noise)
    only = set(a.only.split(",")) if a.only else None
    return stage_sweep(a.seed, a.noise, only)


if __name__ == "__main__":
    sys.exit(main())
