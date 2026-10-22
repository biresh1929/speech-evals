# 1. Whisper-small on code-switched Hindi programming speech

**Question.** WapasBol's users speak Hindi with English technical terms mixed in. How well does an off-the-shelf
Whisper-small transcribe that, measured against human-corrected references rather than assumed?

## Data and method

- Source: my own recordings of explaining code in Hindi, segmented and auto-transcribed (faster-whisper).
- Review: all **195** segments in the Hindi review queue were checked by hand
  ([`HOW_TO_REVIEW.md`](../workspace/voice_dataset/hindi_review/HOW_TO_REVIEW.md)). For each, I typed the correct Hindi,
  confirmed a correct transcript, or marked it `DROP` (noise, silence, cut mid-word, or too wrong to correct).
- Result: **57 correctable**, **138 dropped**.
- Metric: WER = (S + D + I) / reference words, reference = my correction, hypothesis = Whisper's transcript
  ([`scripts/wer_hindi_corrections.py`](../scripts/wer_hindi_corrections.py)). WER can exceed 100%.
- Data: [`workspace/voice_dataset/hindi_review/hindi_review.csv`](../workspace/voice_dataset/hindi_review/hindi_review.csv)
  (transcripts only; the audio is not published).

## Results (`python scripts/wer_hindi_corrections.py`)

| | |
|---|---|
| pooled corpus WER | **77.2%** (355 errors / 460 reference words) |
| mean per-segment WER | 79.6% (std 45.7 pts, median 75.0%) |
| 95% CI of the mean | 67.5–91.8% (t, df = 56); 68.4–92.1% (bootstrap) |
| severity | 3 segments < 10%; 11 at 10–50%; 34 at 50–100%; 9 above 100% |

## What the number does and doesn't mean

1. **Survivor bias.** It describes only the 57 salvageable segments; the 138 dropped ones were worse, so the true WER on
   this speech is higher.
2. **Script convention.** In 28 of 57 segments I wrote English technical terms in Latin script ("pointer", "swap") where
   Whisper wrote them in Devanagari. Standard WER counts these as substitutions even when the word was heard right, so
   part of 77.2% is a scoring artefact, not a recognition error. A script-normalised WER is the obvious next step.
3. **One annotator.** My corrections are the reference; there is no inter-annotator agreement.
4. **No English WER.** No English segment was human-reviewed, so none is computed.

## Next

Script-normalised scoring (transliterate both sides before comparing); compare Whisper small / medium / large-v3 and
forced vs detected language on the same 57 segments; a second annotator on a subset.
