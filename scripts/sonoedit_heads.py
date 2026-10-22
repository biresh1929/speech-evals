"""SonoEdit Option 3: are the layer-11/27 readouts concentrated in a few heads?

    python scripts/sonoedit_heads.py patterns            # Step 1, all seeds
    python scripts/sonoedit_heads.py ablate [--seed N]   # Step 2, checkpointed per seed
    python scripts/sonoedit_heads.py extras [--seed N]   # write-norm-ranked and GQA-group knockouts

NO WEIGHT EDITS. Layers 11 and 27 get a replacement attention forward for the
duration of this process only: it recomputes attention exactly as the model
does, then optionally intervenes on activations. The model's parameters are
never written. A sanity pass checks the replacement reproduces the stock
forward before anything is measured.

Why an exact recompute is possible: these are full-attention layers, which in
Cohere2 apply NO rotary embedding. There is no q/k norm, no softcapping, no
bias; scaling is head_dim**-0.5; 16 query heads share 4 KV heads (GQA,
repeat_kv layout: query head i uses KV head i // 4).

STEP 1 (patterns), per layer x head x seed, over queries at the word's audio
frames (all 8 quantizers per frame):
  * attention mass on the word's text tokens (' Postgre', 'SQL');
  * that mass divided by the mean mass per other prompt text token ("x per
    token": 1.0 = no preference for the word);
  * WRITE NORM: the norm of what the head adds to the residual from the word's
    tokens, ||W_o[head] . sum_{j in word} a_ij v_j||. Attention mass alone is not
    causal: a head can stare at a token whose value it barely writes. This is
    the better Step-1 indicator.
  * the same mass measured from control frames before the word.

STEP 2 (ablate), two runtime interventions, each on a single head:
  * KNOCKOUT: that head may not attend from any position after the word to the
    word's tokens (scores set to -inf before softmax). This removes exactly the
    readout edge and nothing else the head does.
  * MEAN-ABLATION (as the prompt asked): that head's output at every position
    after the word is replaced by its mean over those positions in the clean
    run. This removes everything the head does there, so it is the blunter
    tool; the before-word control frames show its specificity.
Effect = fraction of the word's dependence removed:
  (p_clean - p_intervened) / (p_clean - p_corrupted), on word-frame q0 tokens,
with p_corrupted from that seed's causal-tracing sweep. 1.0 = as bad as
destroying the word's embedding. Plus all-heads knockouts at 11, 27 and both
(sanity: should reproduce the layer-level finding) and cumulative top-k.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sonoedit_trace import OUT, SEEDS, prepare, prob_table, run_pass  # noqa: E402

READOUT_LAYERS = (11, 27)


class HeadLab:
    """Exact recompute of full-attention layers with per-head interventions."""

    def __init__(self, provider, layers=READOUT_LAYERS):
        import torch

        self.torch = torch
        self.model = provider._model
        cfg = self.model.config
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.layers = layers
        self.enabled = False
        self.knockout: dict[int, set[int]] = {}      # layer -> heads
        self.ablate: dict[int, set[int]] = {}        # layer -> heads
        self.query_from = 0                          # intervene on queries at positions >= this
        self.word_keys: list[int] = []
        self.capture: dict | None = None             # set to a dict to collect Step-1 stats
        self.stat_queries: dict[str, list[int]] = {}
        self.head_means: dict[int, object] = {}      # layer -> [heads, head_dim] clean mean
        self.originals = {}
        for L in layers:
            attn = self.model.model.layers[L].self_attn
            assert attn.sliding_window is None, f"layer {L} is not full attention"
            self.originals[L] = attn.forward
            attn.forward = self._make_forward(L, attn)

    def _make_forward(self, L, attn):
        torch = self.torch

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
            if not self.enabled:
                return self.originals[L](hidden_states, position_embeddings, attention_mask,
                                         past_key_values=past_key_values,
                                         cache_position=cache_position, **kwargs)
            B, T, _ = hidden_states.shape
            H, K, D = self.n_heads, self.n_kv, self.head_dim
            q = attn.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2).float()
            k = attn.k_proj(hidden_states).view(B, T, K, D).transpose(1, 2).float()
            v = attn.v_proj(hidden_states).view(B, T, K, D).transpose(1, 2).float()
            k = k.repeat_interleave(H // K, dim=1)          # == repeat_kv layout
            v = v.repeat_interleave(H // K, dim=1)
            scores = (q @ k.transpose(2, 3)) * attn.scaling
            causal = torch.ones(T, T, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
            for h in self.knockout.get(L, ()):
                rows = torch.arange(self.query_from, T, device=scores.device)
                scores[:, h, rows[:, None], torch.tensor(self.word_keys, device=scores.device)] = float("-inf")
            weights = torch.softmax(scores, dim=-1)
            per_head = weights @ v                           # [B, H, T, D]

            if self.capture is not None:
                self._collect(L, attn, weights, v)
            if L in self.head_means and self.ablate.get(L):
                for h in self.ablate[L]:
                    per_head[:, h, self.query_from:] = self.head_means[L][h]
            out = per_head.transpose(1, 2).reshape(B, T, H * D).to(hidden_states.dtype)
            return attn.o_proj(out), None

        return forward

    def _collect(self, L, attn, weights, v):
        torch = self.torch
        w0 = weights[0]                                      # [H, T, T]
        word = self.word_keys
        other_text = [j for j in range(self.prompt_len) if j not in word]
        W_o = attn.o_proj.weight.float()                      # [hidden, H*D]
        stats = {}
        for name, qpos in self.stat_queries.items():
            qp = torch.tensor(qpos, device=w0.device)
            rows = w0[:, qp]                                  # [H, Q, T]
            mass_word = rows[:, :, word].sum(-1).mean(-1)
            mass_other = rows[:, :, other_text].sum(-1).mean(-1)
            per_tok = (mass_word / len(word)) / (mass_other / len(other_text)).clamp_min(1e-9)
            entry = {"mass_on_word": mass_word.cpu().tolist(),
                     "mass_on_other_text": mass_other.cpu().tolist(),
                     "word_vs_other_text_per_token": per_tok.cpu().tolist()}
            if name == "word_frames":
                # What each head writes from the word's tokens, through o_proj.
                contrib = torch.einsum("hqk,hkd->hqd", rows[:, :, word], v[0][:, word])  # [H, Q, D]
                norms, totals = [], []
                full = torch.einsum("hqk,hkd->hqd", rows, v[0])
                for h in range(self.n_heads):
                    Wh = W_o[:, h * self.head_dim:(h + 1) * self.head_dim]
                    norms.append(float((contrib[h] @ Wh.T).norm(dim=-1).mean()))
                    totals.append(float((full[h] @ Wh.T).norm(dim=-1).mean()))
                entry["write_norm_from_word"] = norms
                entry["head_output_norm"] = totals
            stats[name] = entry
        self.capture[L] = stats

    def record_means(self, positions_from):
        """Mean per-head output over positions >= positions_from, from a clean pass."""
        self.head_means = {}
        self._mean_from = positions_from
        torch = self.torch
        for L in self.layers:
            attn = self.model.model.layers[L].self_attn

            def hook(_m, args, kwargs, _L=L, _attn=attn):
                hs = args[0] if args else kwargs["hidden_states"]
                B, T, _ = hs.shape
                H, K, D = self.n_heads, self.n_kv, self.head_dim
                q = _attn.q_proj(hs).view(B, T, H, D).transpose(1, 2).float()
                k = _attn.k_proj(hs).view(B, T, K, D).transpose(1, 2).float().repeat_interleave(H // K, 1)
                v = _attn.v_proj(hs).view(B, T, K, D).transpose(1, 2).float().repeat_interleave(H // K, 1)
                s = (q @ k.transpose(2, 3)) * _attn.scaling
                s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
                ph = torch.softmax(s, -1) @ v
                self.head_means[_L] = ph[0, :, positions_from:].mean(1)       # [H, D]
            self._pre = getattr(self, "_pre", [])
            self._pre.append(attn.register_forward_pre_hook(hook, with_kwargs=True))

    def clear_mean_hooks(self):
        for hnd in getattr(self, "_pre", []):
            hnd.remove()
        self._pre = []


def stage_patterns() -> int:
    from sonoedit_trace import load_provider

    provider = load_provider()
    lab = HeadLab(provider)          # wraps layers 11/27 once for the whole process
    results = {}
    for seed in SEEDS:
        results[str(seed)] = _patterns_one(provider, lab, seed)
        (OUT / "heads_patterns.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def _patterns_one(provider, lab, seed):
    p = provider
    _, ids, mask, emb, span, pred, allowed, first, q_n, meta = prepare(seed, p)
    plen = meta["prompt_tokens"]
    f = meta["frames"]
    lab.prompt_len, lab.word_keys, lab.knockout, lab.ablate = plen, span, {}, {}
    lab.stat_queries = {
        "word_frames": [plen + fr * q_n + q for fr in f["word"] for q in range(q_n)],
        "before_frames": [plen + fr * q_n + q for fr in f["before"] for q in range(q_n)],
    }
    stock = prob_table(p, run_pass(p, emb, mask), pred, ids, allowed, first, q_n)
    lab.enabled, lab.capture = True, {}
    mine = prob_table(p, run_pass(p, emb, mask), pred, ids, allowed, first, q_n)
    lab.enabled = False
    capture, lab.capture = lab.capture, None
    sanity = max(abs(a - b) for k in stock for a, b in zip(stock[k], mine[k], strict=True))
    print(f"seed {seed}: replacement forward vs stock, max |prob diff| = {sanity:.5f}", flush=True)
    for L in READOUT_LAYERS:
        s = capture[L]["word_frames"]
        share = np.array(s["write_norm_from_word"]) / max(sum(s["write_norm_from_word"]), 1e-9)
        order = np.argsort(share)[::-1]
        print(f"  L{L} top heads by write-from-word: "
              + ", ".join(f"h{h} {share[h]*100:.0f}% (mass {s['mass_on_word'][h]:.2f}, "
                          f"x{s['word_vs_other_text_per_token'][h]:.1f}/tok)" for h in order[:4]), flush=True)
    return {"sanity_max_prob_diff": sanity, "layers": {str(k): v for k, v in capture.items()}, "meta": meta}


def effect(p_clean, p_corr, p_int):
    return (p_clean - p_int) / (p_clean - p_corr) if abs(p_clean - p_corr) > 1e-9 else float("nan")


def stage_ablate(seeds) -> int:
    from sonoedit_trace import load_provider

    provider = load_provider()
    lab = HeadLab(provider)
    for seed in seeds:
        _ablate_one(provider, lab, seed)
    return 0


def _ablate_one(p, lab, seed):
    ckpt = OUT / f"heads_ablate_seed{seed}.json"
    state = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.is_file() else {}
    res = state.get("results", {})
    _, ids, mask, emb, span, pred, allowed, first, q_n, meta = prepare(seed, p)
    sweep = json.loads((OUT / f"sweep_seed{seed}.json").read_text(encoding="utf-8"))
    p_corr = sweep["corrupted_mean"]["word"][0]
    lab.prompt_len, lab.word_keys, lab.capture = meta["prompt_tokens"], span, None
    lab.knockout, lab.ablate = {}, {}
    after_word = span[-1] + 1

    clean = prob_table(p, run_pass(p, emb, mask), pred, ids, allowed, first, q_n)
    lab.record_means(after_word)
    run_pass(p, emb, mask)
    lab.clear_mean_hooks()
    H = lab.n_heads

    def run(key, knockout=None, ablate=None, query_from=None):
        if key in res:
            return res[key]
        lab.knockout = knockout or {}
        lab.ablate = ablate or {}
        lab.query_from = after_word if query_from is None else query_from
        lab.enabled = True
        t = prob_table(p, run_pass(p, emb, mask), pred, ids, allowed, first, q_n)
        lab.enabled = False
        res[key] = {"word_q0": t["word"][0], "before_q0": t["before"][0],
                    "after_q0": t["after"][0] if "after" in t else None,
                    "effect": effect(clean["word"][0], p_corr, t["word"][0]),
                    "before_change": t["before"][0] - clean["before"][0]}
        ckpt.write_text(json.dumps({"seed": seed, "clean": clean, "p_corrupted_word_q0": p_corr,
                                    "results": res}, indent=1), encoding="utf-8")
        return res[key]

    t0 = time.perf_counter()
    allh = set(range(H))
    r = run("manual_no_intervention")
    print(f"seed {seed} sanity (replacement forward, no intervention): effect {r['effect']:+.3f}", flush=True)
    for tag, ko in (("ko_all_L11", {11: allh}), ("ko_all_L27", {27: allh}),
                    ("ko_all_both", {11: allh, 27: allh})):
        r = run(tag, knockout=ko)
        print(f"seed {seed} {tag}: effect {r['effect']:.2f}, before-frames change {r['before_change']:+.3f}",
              flush=True)
    r = run("ko_all_both_audio_only", knockout={11: allh, 27: allh}, query_from=meta["prompt_tokens"])
    print(f"seed {seed} ko_all_both_audio_only: effect {r['effect']:.2f}", flush=True)

    for L in READOUT_LAYERS:
        for h in range(H):
            run(f"ko|{L}|{h}", knockout={L: {h}})
            run(f"mean|{L}|{h}", ablate={L: {h}})
        ranked = sorted(range(H), key=lambda h, L=L: -res[f"ko|{L}|{h}"]["effect"])
        print(f"seed {seed} L{L} single-head knockout effects (top 5): "
              + ", ".join(f"h{h} {res[f'ko|{L}|{h}']['effect']:.2f}" for h in ranked[:5])
              + f" | {(time.perf_counter()-t0)/60:.0f} min so far", flush=True)

    for k in (1, 2, 4):
        ko = {L: set(sorted(range(H), key=lambda h, L=L: -res[f"ko|{L}|{h}"]["effect"])[:k])
              for L in READOUT_LAYERS}
        r = run(f"ko_top{k}_both", knockout=ko)
        print(f"seed {seed} knockout top-{k} per layer {ko}: effect {r['effect']:.2f}, "
              f"before change {r['before_change']:+.3f}", flush=True)


def stage_extras(seeds) -> int:
    """Follow-ups once the single-head sweep exists: rank heads by WRITE NORM
    (Step 1) instead of by their small single-knockout effects, knock out whole
    GQA key/value groups, and a low-write control group."""
    from sonoedit_trace import load_provider

    provider = load_provider()
    lab = HeadLab(provider)
    patterns = json.loads((OUT / "heads_patterns.json").read_text(encoding="utf-8"))
    for seed in seeds:
        _extras_one(provider, lab, seed, patterns[str(seed)])
    return 0


def _extras_one(p, lab, seed, pat):
    ckpt = OUT / f"heads_ablate_seed{seed}.json"
    state = json.loads(ckpt.read_text(encoding="utf-8"))
    res = state["results"]
    _, ids, mask, emb, span, pred, allowed, first, q_n, meta = prepare(seed, p)
    lab.prompt_len, lab.word_keys, lab.capture, lab.ablate = meta["prompt_tokens"], span, None, {}
    clean, p_corr = state["clean"], state["p_corrupted_word_q0"]
    rank = {L: [int(h) for h in np.argsort(pat["layers"][str(L)]["word_frames"]["write_norm_from_word"])[::-1]]
            for L in READOUT_LAYERS}

    def run(key, knockout):
        if key not in res:
            lab.knockout, lab.query_from, lab.enabled = knockout, span[-1] + 1, True
            t = prob_table(p, run_pass(p, emb, mask), pred, ids, allowed, first, q_n)
            lab.enabled = False
            res[key] = {"word_q0": t["word"][0], "before_q0": t["before"][0],
                        "effect": effect(clean["word"][0], p_corr, t["word"][0]),
                        "before_change": t["before"][0] - clean["before"][0], "heads": {
                            str(L): sorted(hs) for L, hs in knockout.items()}}
            ckpt.write_text(json.dumps({**state, "results": res}, indent=1), encoding="utf-8")
        r = res[key]
        print(f"seed {seed} {key}: effect {r['effect']:.2f} (before {r['before_change']:+.3f}) "
              f"heads {r['heads']}", flush=True)

    for k in (1, 2, 3, 4, 5, 6, 8, 12):
        run(f"wr_L27_top{k}", {27: set(rank[27][:k])})
    for k in (1, 2, 3, 5):
        run(f"wr_L11_top{k}", {11: set(rank[11][:k])})
    for k in (1, 2, 3, 4, 6):
        run(f"wr_both_top{k}", {11: set(rank[11][:k]), 27: set(rank[27][:k])})
    run("group_L27_kv1", {27: {4, 5, 6, 7}})
    run("group_L27_kv0", {27: {0, 1, 2, 3}})
    run("group_L11_kv0", {11: {0, 1, 2, 3}})
    run("control_kv3_both", {11: {12, 13, 14, 15}, 27: {12, 13, 14, 15}})


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["patterns", "ablate", "extras"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.stage == "patterns":
        return stage_patterns()
    seeds = [a.seed] if a.seed else SEEDS
    if a.stage == "extras":
        return stage_extras(seeds)
    return stage_ablate(seeds)


if __name__ == "__main__":
    sys.exit(main())
