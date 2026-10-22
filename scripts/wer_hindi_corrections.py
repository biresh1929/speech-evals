"""Word error rate of Whisper `small` on the human-corrected Hindi segments.

Reference  = the human correction (treated as ground truth)
Hypothesis = Whisper's original transcript
WER        = (substitutions + deletions + insertions) / reference words
             -- can exceed 100% when Whisper inserted more words than exist.

Three biases this number carries, reported alongside it rather than buried:

1. SURVIVOR BIAS. Only the 57 segments the reviewer could correct are here. The
   138 marked DROP were unusable -- effectively unrecoverable -- and are
   excluded. So this WER describes the *salvageable* output, and understates how
   badly `small` did on Hindi overall.

2. SCRIPT CONVENTION. The reviewer wrote English technical terms in Latin script
   ("swap", "pointer") where Whisper wrote them in Devanagari ("स्वाप"). Those
   count as substitutions even when Whisper heard the word correctly. Segments
   where this applies are counted so its size is visible.

3. ONE ANNOTATOR. A single reviewer's corrections are the reference; there is no
   inter-annotator agreement to bound how "true" that reference is.

English: there is NO ground truth for any English segment. No human reviewed
them. This script computes no English WER and none should be estimated.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

CSV_PATH = Path(DATA_ROOT + r"/voice_dataset/hindi_review/hindi_review.csv")
OUT_PATH = Path(DATA_ROOT + r"/voice_dataset/hindi_review/wer_report.json")

DROP_MARKERS = {"DROP", "DELETE", "X", "SKIP-BAD", "BAD"}
# Punctuation to strip before comparing: ASCII punctuation, danda and double
# danda, and common Unicode quote/dash forms. Word content is left untouched.
PUNCT = re.compile(r"[\u0964\u0965.,!?;:\"'()\[\]{}\-\u2013\u2014\u2018\u2019\u201c\u201d]")
LATIN_WORD = re.compile(r"[A-Za-z]")

T_975_DF56 = 2.0032  # two-sided 95% critical t for 56 degrees of freedom


def normalise(text: str) -> list[str]:
    """NFC, strip punctuation, lowercase Latin, split on whitespace.

    NFC matters for Devanagari: the same visible syllable can be encoded with
    different code-point sequences, and without normalising those would count as
    substitutions that no listener could hear.
    """
    t = unicodedata.normalize("NFC", text)
    t = PUNCT.sub(" ", t).lower()
    return t.split()


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) by Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, D, I) for ref[:i] vs hyp[:j]
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub, dele, ins = dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]
            dp[i][j] = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
    _, s, d, i_ = dp[n][m]
    return s, d, i_


def bootstrap_ci(values: list[float], iters: int = 20000, seed: int = 0) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    corrected = [
        r for r in rows
        if (r.get("corrected_transcript") or "").strip()
        and r["corrected_transcript"].strip().upper() not in DROP_MARKERS
    ]
    dropped = sum(1 for r in rows if r["corrected_transcript"].strip().upper() in DROP_MARKERS)

    per = []
    tot_err = tot_ref = 0
    for r in corrected:
        ref = normalise(r["corrected_transcript"])
        hyp = normalise(r["current_transcript"])
        s, d, i = edit_counts(ref, hyp)
        wer = (s + d + i) / len(ref) if ref else 0.0
        tot_err += s + d + i
        tot_ref += len(ref)
        per.append({
            "clip": r["audio_file"], "wer": round(wer, 4),
            "ref_words": len(ref), "hyp_words": len(hyp), "S": s, "D": d, "I": i,
            "reference_has_latin_words": any(LATIN_WORD.search(w) for w in ref),
            "hypothesis_has_latin_words": any(LATIN_WORD.search(w) for w in hyp),
        })

    wers = [p["wer"] for p in per]
    n = len(wers)
    mean = sum(wers) / n
    sd = math.sqrt(sum((w - mean) ** 2 for w in wers) / (n - 1))
    half = T_975_DF56 * sd / math.sqrt(n) if n == 57 else float("nan")
    boot = bootstrap_ci(wers)
    median = sorted(wers)[n // 2] if n % 2 else sum(sorted(wers)[n // 2 - 1:n // 2 + 1]) / 2

    buckets = {
        "win (<10%, minor fix)": sum(w < 0.10 for w in wers),
        "moderate (10-50%)": sum(0.10 <= w <= 0.50 for w in wers),
        "loss (>50%, major rewrite) -- total": sum(w > 0.50 for w in wers),
        "  loss, 50-100%": sum(0.50 < w <= 1.00 for w in wers),
        "  loss, >100% (more errors than reference words)": sum(w > 1.00 for w in wers),
    }
    script_mismatch = sum(p["reference_has_latin_words"] and not p["hypothesis_has_latin_words"] for p in per)

    print(f"Hindi segments: {len(rows)} reviewed, {n} corrected (scored), {dropped} dropped (excluded)\n")
    print(f"{'clip':<12}{'WER':>8}{'ref':>5}{'hyp':>5}{'S':>4}{'D':>4}{'I':>4}")
    for p in per:
        print(f"{p['clip']:<12}{p['wer']*100:>7.1f}%{p['ref_words']:>5}{p['hyp_words']:>5}"
              f"{p['S']:>4}{p['D']:>4}{p['I']:>4}")

    print(f"\nmean per-segment WER : {mean*100:.1f}%")
    print(f"std deviation        : {sd*100:.1f} pts")
    print(f"95% CI (t, df=56)    : {(mean-half)*100:.1f}% to {(mean+half)*100:.1f}%")
    print(f"95% CI (bootstrap)   : {boot[0]*100:.1f}% to {boot[1]*100:.1f}%")
    print(f"median               : {median*100:.1f}%")
    print(f"pooled corpus WER    : {tot_err/tot_ref*100:.1f}%  ({tot_err} errors / {tot_ref} reference words)")
    print("\nseverity:")
    for k, v in buckets.items():
        print(f"  {k:<70} {v:>3}")
    print(f"\nsegments where the reference uses Latin-script English terms the hypothesis "
          f"wrote in Devanagari: {script_mismatch} of {n}")
    print("\nEnglish: no ground truth exists (no English segment has been human-reviewed). "
          "No English WER is computed or estimated.")

    OUT_PATH.write_text(json.dumps({
        "model_evaluated": "faster-whisper-small",
        "segments_scored": n, "segments_dropped_excluded": dropped,
        "mean_wer": mean, "sd": sd, "ci95_t": [mean - half, mean + half],
        "ci95_bootstrap": list(boot), "median": median,
        "pooled_wer": tot_err / tot_ref, "severity": buckets,
        "script_convention_mismatch_segments": script_mismatch,
        "english": "no ground truth; not computed",
        "per_segment": per,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
