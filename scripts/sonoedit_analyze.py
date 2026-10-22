"""Summarise a causal-tracing sweep checkpoint from sonoedit_trace.py.

    python scripts/sonoedit_analyze.py [--seed 11]

Recovery = (p_restored - p_corrupted) / (p_clean - p_corrupted) for the word
frames' quantizer-0 tokens: 0 means restoring that site did nothing, 1 means it
fully undid the corruption. Also reported: the same for q1-q7 (mean), and
whether full-attention layers differ from sliding-window ones. Works on a
partial checkpoint, so it can be read while the sweep is still running.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(DATA_ROOT + r"/rumik_samples/sonoedit")


def recovery(results, clean, corr, frames, q_idx):
    c = np.mean([clean[frames][q] for q in q_idx])
    x = np.mean([corr[frames][q] for q in q_idx])
    r = np.mean([results[frames][q] for q in q_idx])
    return float((r - x) / (c - x)) if abs(c - x) > 1e-6 else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    d = json.loads((OUT / f"sweep_seed{a.seed}.json").read_text(encoding="utf-8"))
    clean, corr, res, types = d["clean"], d["corrupted_mean"], d["results"], d["layer_types"]
    n = len(types)
    conds = [("layer", "span"), ("layer", "last_token"), ("attn", "span"), ("mlp", "span")]
    table = {c: [None] * n for c in conds}
    table_rest = {c: [None] * n for c in conds}
    for key, val in res.items():
        comp, layer, where = key.split("|")
        table[(comp, where)][int(layer)] = recovery(val, clean, corr, "word", [0])
        table_rest[(comp, where)][int(layer)] = recovery(val, clean, corr, "word", range(1, 8))

    print(f"seed {a.seed}: {len(res)}/{4 * n} conditions done; clean q0 {clean['word'][0]:.3f}, "
          f"corrupted q0 {corr['word'][0]:.3f}; attn/mlp windows of {d['window']} layers\n")
    print("recovery of word-frame q0 (1.0 = corruption fully undone)")
    print(f"{'L':>3} {'type':<5} {'layer@span':>10} {'layer@last':>10} {'attn@span':>10} {'mlp@span':>10}"
          f"   {'mlp q1-7':>8}")
    fmt = lambda v: f"{v:10.3f}" if v is not None else f"{'-':>10}"  # noqa: E731
    for L in range(n):
        row = [table[c][L] for c in conds]
        if all(v is None for v in row):
            continue
        print(f"{L:>3} {'FULL' if types[L] == 'full_attention' else 'slide':<5} "
              + " ".join(fmt(v) for v in row) + f"   {fmt(table_rest[('mlp', 'span')][L])[2:]}")

    summary = {}
    for c in conds:
        vals = [(L, v) for L, v in enumerate(table[c]) if v is not None]
        if not vals:
            continue
        ranked = sorted(vals, key=lambda t: -t[1])
        full = [v for L, v in vals if types[L] == "full_attention"]
        slide = [v for L, v in vals if types[L] != "full_attention"]
        summary[f"{c[0]}@{c[1]}"] = {
            "top5": [(L, round(v, 3)) for L, v in ranked[:5]],
            "mean_full_attention_layers": round(float(np.mean(full)), 3) if full else None,
            "mean_sliding_layers": round(float(np.mean(slide)), 3) if slide else None,
        }
    print()
    for k, s in summary.items():
        print(f"{k:<16} top layers {s['top5']}   full-attn mean {s['mean_full_attention_layers']}  "
              f"sliding mean {s['mean_sliding_layers']}")
    (OUT / f"sweep_summary_seed{a.seed}.json").write_text(json.dumps(
        {"table_q0": {f"{c[0]}@{c[1]}": table[c] for c in conds},
         "table_q1_7": {f"{c[0]}@{c[1]}": table_rest[c] for c in conds}, "summary": summary}, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
