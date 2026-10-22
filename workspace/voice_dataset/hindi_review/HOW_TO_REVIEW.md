# Reviewing the Hindi transcripts

195 clips in `clips/`, ordered **shortest first** so you can warm up
before the hard ones.

## What to do

1. Open `hindi_review.csv`. It is UTF-8 with a BOM, so Excel, LibreOffice and
   Google Sheets will all read the Devanagari correctly. **If you see mojibake,
   stop** -- something re-saved it in the wrong encoding and corrections would
   be lost.
2. Play `clips/hi_001.wav`, read `current_transcript`, and:

   | what you heard | what to type in `corrected_transcript` |
   |---|---|
   | transcript is wrong | the correct Hindi text |
   | transcript is already right | retype or copy it (this marks it *verified*) |
   | clip is unusable -- noise, silence, cut off mid-word | `DROP` |
   | you want to skip and come back | leave blank |

   Blank means "not reviewed" and the row stays `auto`. Typing anything marks it
   `human_corrected`, which is what makes it trustworthy training data -- so
   confirming a correct transcript is just as valuable as fixing a wrong one.

3. Save as CSV (keep the encoding), then run:

   ```
   python scripts/merge_hindi_corrections.py
   ```

   It previews every change and asks before writing. `--apply` skips the prompt;
   `--dry-run` shows the diff and writes nothing.

## Why bother

These rows are Whisper `small` guesses at conversational Hindi and are the least
reliable part of the dataset. Everything else is auto-transcribed English that
happens to be fairly clean. Correcting these 197 turns the weakest slice into
the only slice with real ground truth.
