"""Merge reviewed Hindi transcripts back into manifest.jsonl.

Marks corrected rows `confidence: "human_corrected"`, which is the point of the
whole exercise: it distinguishes text a person actually verified from text
Whisper guessed at, so training code can weight or filter on it.

Safety properties, in order of how much they matter:

1. **Backs up the manifest first.** Timestamped copy next to it, every run.
2. **Previews before writing**, and asks. `--apply` skips the prompt,
   `--dry-run` never writes.
3. **Matches on (source, start), not row order.** Row indices would silently
   corrupt everything if the manifest were ever re-generated or re-sorted
   between export and merge -- which happens, because re-running the dataset
   prep changes counts.
4. **Blank means untouched.** Only rows with text typed into
   `corrected_transcript` are modified; everything else keeps `auto`.
5. **Detects a mangled encoding and refuses.** If the CSV was re-saved as
   cp1252, the Devanagari arrives as mojibake and merging it would destroy the
   very data being corrected.
"""

from __future__ import annotations
from _paths import DATA_ROOT  # speech-evals: where data and outputs live (see _paths.py)

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

MANIFEST = Path(DATA_ROOT + r"/voice_dataset/manifest.jsonl")
CSV_PATH = Path(DATA_ROOT + r"/voice_dataset/hindi_review/hindi_review.csv")

DROP_MARKERS = {"DROP", "DELETE", "X", "SKIP-BAD", "BAD"}
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Sequences that show up when UTF-8 Devanagari is decoded as cp1252/latin-1.
MOJIBAKE_HINTS = ("à¤", "à¥", "Ã ", "â€")


def looks_mojibake(text: str) -> bool:
    return any(h in text for h in MOJIBAKE_HINTS)


def _force_utf8_stdout() -> None:
    """Make the console able to print Devanagari.

    A Windows console defaults to cp1252, and printing Hindi to it raises
    UnicodeEncodeError mid-preview -- which is exactly where this script shows
    you what it is about to change. `errors="replace"` means a console that
    genuinely cannot render a glyph shows a placeholder instead of crashing;
    the data written to disk is unaffected either way.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main() -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--apply", action="store_true", help="Write without prompting.")
    ap.add_argument("--dry-run", action="store_true", help="Show changes, write nothing.")
    ap.add_argument(
        "--allow-non-devanagari",
        action="store_true",
        help="Accept corrections with no Devanagari (e.g. deliberate romanisation).",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"review CSV not found: {args.csv}")
        print("Run scripts/export_hindi_review.py first.")
        return 1
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}")
        return 1

    # utf-8-sig strips the BOM if present and is harmless if not.
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        review = list(csv.DictReader(f))

    required = {"current_transcript", "corrected_transcript", "source_recording", "start_s"}
    missing = required - set(review[0] if review else {})
    if missing:
        print(f"CSV is missing required column(s): {sorted(missing)}")
        return 1

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    # (source, rounded start) -> manifest row. Rounded because the CSV stores
    # start to 2dp and floats do not round-trip exactly through text.
    index: dict[tuple[str, float], dict] = {
        (r["source"], round(float(r["start"]), 2)): r for r in rows
    }

    corrections: list[tuple[dict, str, str]] = []
    drops: list[tuple[dict, str]] = []
    unmatched: list[str] = []
    suspicious: list[str] = []
    blank = 0

    for row in review:
        corrected = (row.get("corrected_transcript") or "").strip()
        if not corrected:
            blank += 1
            continue

        key = (row["source_recording"], round(float(row["start_s"]), 2))
        target = index.get(key)
        if target is None:
            unmatched.append(f"{row.get('audio_file', '?')} ({key[0]} @ {key[1]}s)")
            continue

        if corrected.upper() in DROP_MARKERS:
            drops.append((target, row.get("audio_file", "?")))
            continue

        if looks_mojibake(corrected):
            suspicious.append(
                f"{row.get('audio_file', '?')}: looks like mojibake -- the CSV was "
                f"probably re-saved in the wrong encoding: {corrected[:40]!r}"
            )
            continue
        if not DEVANAGARI.search(corrected) and not args.allow_non_devanagari:
            suspicious.append(
                f"{row.get('audio_file', '?')}: no Devanagari in {corrected[:40]!r} "
                f"(pass --allow-non-devanagari if this is intentional)"
            )
            continue

        corrections.append((target, row.get("audio_file", "?"), corrected))

    print(f"review rows:        {len(review)}")
    print(f"  corrections:      {len(corrections)}")
    print(f"  marked to drop:   {len(drops)}")
    print(f"  left blank:       {blank}")
    print(f"  unmatched:        {len(unmatched)}")
    print(f"  suspicious:       {len(suspicious)}")

    if suspicious:
        print("\nREFUSING these rows (fix and re-run):")
        for s in suspicious[:10]:
            print(f"  {s}")
    if unmatched:
        print("\nCould not match to a manifest row (was the manifest regenerated?):")
        for u in unmatched[:10]:
            print(f"  {u}")

    if not corrections and not drops:
        print("\nNothing to merge.")
        return 0

    print("\n--- changes ---")
    for target, audio_file, corrected in corrections[:15]:
        old = target["text"]
        verified = old.strip() == corrected
        label = "VERIFIED (unchanged text)" if verified else "CORRECTED"
        print(f"\n{audio_file}  [{label}]")
        print(f"   was: {old[:90]}")
        if not verified:
            print(f"   now: {corrected[:90]}")
    if len(corrections) > 15:
        print(f"\n... and {len(corrections) - 15} more corrections")
    for target, audio_file in drops[:10]:
        print(f"\n{audio_file}  [DROP] {target['text'][:70]}")
    if len(drops) > 10:
        print(f"... and {len(drops) - 10} more drops")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if not args.apply:
        print(
            f"\nApply {len(corrections)} correction(s) and remove {len(drops)} row(s)? [y/N] ",
            end="",
        )
        try:
            if input().strip().lower() not in ("y", "yes"):
                print("aborted, nothing written.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\naborted, nothing written.")
            return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = args.manifest.with_name(f"{args.manifest.stem}.backup-{stamp}.jsonl")
    shutil.copy2(args.manifest, backup)

    drop_ids = {id(t) for t, _ in drops}
    for target, _audio_file, corrected in corrections:
        target["text"] = corrected
        target["confidence"] = "human_corrected"

    kept = [r for r in rows if id(r) not in drop_ids]

    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections

    conf = collections.Counter(r["confidence"] for r in kept)
    print(f"\nwrote {args.manifest}")
    print(f"backup: {backup.name}")
    print(f"rows: {len(rows)} -> {len(kept)} ({len(drops)} dropped)")
    print(f"confidence: {dict(conf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
