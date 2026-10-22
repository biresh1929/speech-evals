"""Latency observability — per-node p50/p95/p99, TTFT, histogram buckets.

Append-only JSONL log at `<code>/data/latency.jsonl`. One row per node
invocation:

    {
      "ts": "2026-05-16T12:34:56.789Z",
      "node": "llm.quick",
      "duration_ms": 1234.5,
      "success": true,
      "error": null,
      "metadata": {"model": "...", "ttft_ms": 480}
    }

Disable via env var `LATENCY_DISABLED=1`. Useful when running unit tests
that you don't want to pollute the log.

Adapted from the fiancial-assistant sibling project with finer low-end
histogram buckets — voice TTFA targets are sub-second, so the original
finance buckets (10-60s) were too coarse.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Voice-tuned histogram buckets — finer at the low end since TTFA targets are
# sub-second. Last bucket (10s) is the failure mode (LLM/TTS got stuck).
HISTOGRAM_BUCKETS_S: tuple[float, ...] = (
    0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0
)

# Log to <code>/data/latency.jsonl (relative to this file's location).
LATENCY_LOG = Path(__file__).resolve().parent.parent / "data" / "latency.jsonl"


def _enabled() -> bool:
    return os.environ.get("LATENCY_DISABLED", "0") != "1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_row(row: dict) -> None:
    if not _enabled():
        return
    try:
        LATENCY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LATENCY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.debug("latency log write failed: %s", exc)


def record_latency(
    node: str,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Append a single latency observation to the log."""
    _write_row({
        "ts": _utc_now_iso(),
        "node": node,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "error": error,
        "metadata": metadata or {},
    })


def track_latency(node: str, *, capture_args: bool = False) -> Callable:
    """Decorator that records duration + success of the wrapped callable.

    Works on both sync and async functions. `capture_args` adds a short
    `args_preview` to the metadata — only enable for nodes where it's useful.
    """
    def decorator(fn: Callable) -> Callable:
        is_coroutine = inspect.iscoroutinefunction(fn)

        if is_coroutine:
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                err: Optional[str] = None
                ok = True
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"[:200]
                    raise
                finally:
                    dur_ms = (time.perf_counter() - t0) * 1000
                    meta: dict = {}
                    if capture_args:
                        meta["args_preview"] = _arg_preview(args, kwargs)
                    record_latency(node, dur_ms, ok, err, meta)
            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            err: Optional[str] = None
            ok = True
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                ok = False
                err = f"{type(exc).__name__}: {exc}"[:200]
                raise
            finally:
                dur_ms = (time.perf_counter() - t0) * 1000
                meta: dict = {}
                if capture_args:
                    meta["args_preview"] = _arg_preview(args, kwargs)
                record_latency(node, dur_ms, ok, err, meta)
        return sync_wrapper
    return decorator


def _arg_preview(args, kwargs) -> str:
    bits = [repr(a)[:40] for a in args[:2]]
    bits += [f"{k}={repr(v)[:30]}" for k, v in list(kwargs.items())[:3]]
    return ", ".join(bits)


@contextmanager
def time_to_first_token(node: str, model: str = "", metadata: Optional[dict] = None):
    """Measure TTFT for a streaming call.

        with time_to_first_token("llm.quick", model="llama-3.1") as ttft:
            async for chunk in llm.astream(messages):
                ttft.mark()       # call on the first chunk only
                ...
    """
    state = {"first_token_ts": None, "t0": time.perf_counter()}

    class _Marker:
        def mark(self) -> None:
            if state["first_token_ts"] is None:
                state["first_token_ts"] = time.perf_counter()

    marker = _Marker()
    err: Optional[str] = None
    ok = True
    try:
        yield marker
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"[:200]
        raise
    finally:
        end = time.perf_counter()
        ttft_ms = (
            (state["first_token_ts"] - state["t0"]) * 1000
            if state["first_token_ts"] is not None
            else None
        )
        total_ms = (end - state["t0"]) * 1000
        meta = dict(metadata or {})
        if model:
            meta["model"] = model
        if ttft_ms is not None:
            meta["ttft_ms"] = round(ttft_ms, 2)
        record_latency(node, total_ms, ok, err, meta)


@dataclass
class LatencyReport:
    node: str
    count: int
    error_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    histogram: dict[float, int] = field(default_factory=dict)
    ttft_p95_ms: Optional[float] = None

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count else 0.0

    def is_slow(self, p95_threshold_ms: float = 1000) -> bool:
        # Voice default: anything over 1s p95 is a problem worth investigating.
        return self.p95_ms > p95_threshold_ms


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. pct in [0,100]. Empty → 0.0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _bucketize(values: list[float], buckets: Iterable[float]) -> dict[float, int]:
    """Cumulative histogram in Prometheus style: bucket value = count of
    observations <= bucket boundary. Buckets are in seconds, values in ms."""
    out: dict[float, int] = {}
    for b in sorted(buckets):
        out[b] = sum(1 for v in values if v <= b * 1000)
    return out


def _iter_rows(log_path: Path = LATENCY_LOG) -> Iterable[dict]:
    if not log_path.exists():
        return
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_percentiles(
    log_path: Path = LATENCY_LOG,
    since_iso: Optional[str] = None,
    nodes: Optional[set[str]] = None,
) -> list[LatencyReport]:
    """Compute per-node latency report from the JSONL log.

    Args:
        log_path: file to read; defaults to data/latency.jsonl
        since_iso: only include rows with ts >= this ISO timestamp
        nodes: restrict to this set of node names
    """
    by_node: dict[str, list[float]] = {}
    err_by_node: dict[str, int] = {}
    ttft_by_node: dict[str, list[float]] = {}

    for row in _iter_rows(log_path):
        node = row.get("node")
        if not node:
            continue
        if nodes and node not in nodes:
            continue
        if since_iso and row.get("ts", "") < since_iso:
            continue
        dur = float(row.get("duration_ms", 0))
        by_node.setdefault(node, []).append(dur)
        if not row.get("success", True):
            err_by_node[node] = err_by_node.get(node, 0) + 1
        ttft = (row.get("metadata") or {}).get("ttft_ms")
        if ttft is not None:
            ttft_by_node.setdefault(node, []).append(float(ttft))

    reports: list[LatencyReport] = []
    for node, values in by_node.items():
        ttft_values = ttft_by_node.get(node, [])
        reports.append(LatencyReport(
            node=node,
            count=len(values),
            error_count=err_by_node.get(node, 0),
            p50_ms=round(_percentile(values, 50), 2),
            p95_ms=round(_percentile(values, 95), 2),
            p99_ms=round(_percentile(values, 99), 2),
            mean_ms=round(sum(values) / len(values), 2),
            max_ms=round(max(values), 2),
            histogram=_bucketize(values, HISTOGRAM_BUCKETS_S),
            ttft_p95_ms=round(_percentile(ttft_values, 95), 2) if ttft_values else None,
        ))

    # Sort slowest p95 first — that's what you want to see at the top.
    reports.sort(key=lambda r: r.p95_ms, reverse=True)
    return reports
