"""Recompute real before/after latency + gate numbers from the committed
JSONL logs in data/. Run from the `code/` directory:

    python tools/resume_metrics_report.py

Every number printed is derived directly from logged rows (data/latency*.jsonl)
via observability.latency's own percentile math — nothing here is estimated.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability.latency import _percentile, compute_percentiles  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def report_file(label: str, path: Path) -> None:
    if not path.exists():
        print(f"{label}: MISSING ({path})")
        return
    print(f"\n--- {label} ({path.name}) ---")
    for r in compute_percentiles(log_path=path):
        print(
            f"  {r.node:22s} n={r.count:5d} err={r.error_count:3d} "
            f"p50={r.p50_ms:9.2f}ms p95={r.p95_ms:9.2f}ms p99={r.p99_ms:9.2f}ms "
            f"mean={r.mean_ms:9.2f}ms"
        )


def llm_swap_comparison() -> None:
    section("1. LLM model swap: Nemotron-3-Nano-4B -> Qwen3-4B-2507")
    report_file("before (Nemotron)", DATA / "latency.before-nemotron.jsonl")
    report_file("after (Qwen3)", DATA / "latency.after-qwen3.jsonl")


def stt_gpu_comparison() -> None:
    section("2. STT device: faster-whisper CPU -> CUDA")
    report_file("before (CPU, before-singleshot log)", DATA / "latency.before-singleshot.jsonl")
    report_file("before (CPU, before-nemotron log)", DATA / "latency.before-nemotron.jsonl")
    report_file("current production log (GPU-era only)", DATA / "latency.jsonl")


def completeness_gate_stats() -> None:
    section("3. Completeness gate decisions (data/latency.jsonl)")
    decisions: Counter = Counter()
    with (DATA / "latency.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("node") != "gate.completeness":
                continue
            decisions[(row.get("metadata") or {}).get("decision")] += 1
    total = sum(decisions.values())
    if not total:
        print("  no gate.completeness rows found")
        return
    for decision, count in decisions.most_common():
        print(f"  {decision:10s} {count:4d}  ({count / total * 100:.1f}%)")


def path_split_stats() -> None:
    section("4. Fast-path vs slow-path (safety-net) turn split")
    paths: Counter = Counter()
    with (DATA / "latency.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("node") != "voice.ttfa":
                continue
            paths[(row.get("metadata") or {}).get("path", "(none)")] += 1
    total = sum(paths.values())
    if not total:
        print("  no voice.ttfa rows found")
        return
    for path, count in paths.most_common():
        print(f"  {path:6s} {count:4d}  ({count / total * 100:.1f}%)")


def error_rate() -> None:
    section("5. Exception-based error rate, full current log")
    values: list[float] = []
    errors = 0
    with (DATA / "latency.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            values.append(1)
            if not row.get("success", True):
                errors += 1
    total = len(values)
    print(f"  rows={total} errors={errors} error_rate={errors / total * 100:.2f}%" if total else "  no rows")
    print("  NOTE: this only catches exceptions caught by track_latency/record_latency.")
    print("  It will NOT catch a model that streams successfully but returns empty")
    print("  content (the Nemotron ~89%-empty-completion failure mode noted in commit")
    print("  71f6fe4 was diagnosed by manual log inspection, not this counter).")


if __name__ == "__main__":
    llm_swap_comparison()
    stt_gpu_comparison()
    completeness_gate_stats()
    path_split_stats()
    error_rate()
