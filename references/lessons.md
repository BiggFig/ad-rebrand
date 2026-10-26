# Lessons learned (first run: Artuvate/FisioRest → Nookwave, 7 videos + 3 images)

## Audio
- **Whisper word times are not boundaries.** They were off by up to 0.5 s (one mention's "rest" ran 0.5 s past whisper's end and was heard as "nook wave arrest"; another mention's real span was 0.5 s *earlier* than triage said). Always re-measure on the demucs vocals stem: 20 ms frames, energy dB plus the >4 kHz share of the spectrum. "ph/f" is a low-energy fricative before the first vowel; the word ends at the release of "st"/"t".
- **Don't let the clip start after whisper's onset** — it clips the "N" ("called WAVE"). **Don't include the previous word's plosive release** either ("calle**d**|Nookwave" → "GWAVE").
- If an in-context word is mushy, **borrow a crisp sentence-initial "Nookwave"** from elsewhere in the take. Generate every line in one take with all mentions; Fish's web app returns 2 takes free, so download both.
- The new name was shorter than "FisioRest"; stretch with ffmpeg `atempo` only within 0.87–1.15, place at the slot start, and let the remainder be a pause. Crossfade the old word out (10 ms), RMS-match to the preceding 1 s.
- **Verify by transcription** of (new vocals + music) in a ~5 s window per mention, both with `--prompt NewName` and without. Full-file whisper can merge sentences — spot-check windows rather than trusting a total count.
- **Sung words**: speech TTS + WORLD pitch/loudness transfer onto the original notes works technically (0.05 st median error) but needs a human listen; produce alternatives.
- Some sources are quiet (one was −25.5 LUFS); the final should match the source unless the user asks to normalise.

## Voice cloning (Fish Audio)
- The API key (FISH_AUDIO_API_KEY in ~/.config/video-studio/.env) only spends **API credit**, which is separate from the plan's platform credit. If the API wallet is $0, use the **web app in the user's Chrome** (Claude in Chrome — the in-app browser can't upload files). See references/fish-audio-web.md.
- One private clone per distinct speaker. Jobs that share narration (same envelope, ~0.97 correlation) share a clone; pitch alone (e.g. 107 vs 128 Hz median F0) is enough to tell speakers apart.
- Clone reference: 25–35 s of the **vocals stem**, no brand names, starting and ending in pauses.

## Visual
- Burned-in captions: the most convincing fix **cuts glyphs from the video's own captions** (same font, outline, antialiasing); draw any missing letter to match stems. Re-centre the whole line. For pop-in animations, detect the box per frame and scale.
- Cut points in edits must respect burned captions: captions lead speech by ~0.15 s; at hard cuts, hold the correct caption box over the 3–10 frames that would flash removed text (render_edit.py `caption_fixes`).
- Leather tags on CG turntables: per-frame colour segmentation → oriented rect → warp a stitched pill tag with the new name (rebrand_job01.py `render_tag`). Curved plates: bent texture via remap (`render_bent_tag`). Zoom-in cuts need **multi-scale** template matching.
- White fabric bags: per-frame "dark text on mostly-white" detection + inpaint, two passes (`erase_text_on_white`) — tracking failed because collages jump and scroll diagonally. Re-lettering a tumbling bag is possible with hand-placed keyframes + motion blur but will be softer than the original.
- Web screenshots: multi-scale match the whole line, blank it, re-set the line centred in the page font (Futura matched one page). Wider names squeeze the line ~5–10%.
- Separate jobs that look identical may be offset by a frame (job-03 frame n = job-02 frame n+1). Detect the offset from a caption change.
- Large-occurrence jobs: agents can run out of context before the full render; make the render step robust and finish it yourself if needed.

## Things to surface to the user, not silently fix
- Other brands' marks on the product (e.g. an OEM emboss) — ask; remove only with approval.
- Third-party watermarks/@handles on stock clips — **never remove** (copyright/attribution); flag.
- Retail/endorsement claims (store staff, store branding) and inconsistent numbers between screens (12,000 vs 25,000 customers) — flag; edit only on the user's instruction.
- Script redundancy created by the rename ("a company called Nookwave… the result is Nookwave").
