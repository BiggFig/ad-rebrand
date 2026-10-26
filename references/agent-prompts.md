# Subagent prompt templates

Fill in `{OLD_PRODUCT}`, `{OLD_BRAND}`, `{NEW_NAME}`, `{JOB}`, and the job-specific facts from triage. Launch independent agents in one message with `run_in_background: true`. Every prompt must say: never modify `assets/source.mp4`, write only inside its own job folder(s), and where the environment lives:

> ffmpeg/ffprobe/whisper-cli at /opt/homebrew/bin (export PATH=/opt/homebrew/bin:$PATH) — this ffmpeg has NO drawtext, so all text/overlay work is Python (OpenCV/Pillow) piping raw BGR frames into ffmpeg. Python: ~/.cache/video-studio-venv/bin/python -I (numpy, opencv-contrib, pillow, soundfile, demucs, torch). Whisper model ~/.cache/whisper/ggml-small.en.bin. OpenCV seeking is unreliable on H.264 — decode sequentially for exact frames. Use the Read tool on crops and contact sheets constantly.

Also point each agent at the matching folder in this skill's `examples/` and at `references/lessons.md`.

---

## 1. Triage (one agent per video) — analysis only

```
You are triaging one ad video for a rebrand: the product was "{OLD_PRODUCT}" by the brand "{OLD_BRAND}"; both names must become "{NEW_NAME}". ANALYSIS ONLY — do not edit or render. Produce an exact inventory so another engineer can fix it.

Job folder: {JOB} (assets/source.mp4, {W}x{H}, {FPS} fps, ~{DUR} s). Write outputs under {JOB}/analysis/.
[environment block]

Find and report:
1. SPEECH: full transcript with sentence timings; every spoken old name (whisper mishears: "physio rest", "fizzy rest", "artivate", "art you vate"... — prompt whisper with the names, and check context). Refine each mention's true extent on the demucs vocals stem (20 ms frames, energy dB + >4 kHz share; "f/ph" onset to the final "st/t" release). Report whisper and refined times.
2. SPEAKERS: distinct voices; who says each mention; a 25–35 s clean solo clip per speaker with NO brand name (verify on the vocals stem) for voice cloning. Flag sung, overlapped, or on-camera lip-synced mentions.
3. ON-SCREEN: scan contact sheets every 0.5–1 s, then full-res crops. Every frame range where either old name is visible/legible (tags, device labels, packaging, bags, web screenshots, signs, end cards): first/last frame, bbox at a reference frame, motion type, legibility. List blank/illegible tags separately as no-fix.
4. CAPTIONS: burned in? style (box, font, weight, outline, y-range, per word/phrase); frame ranges of captions containing old names.
5. Anything else: URLs, @handles, other brands' marks, watermarks, claims that conflict between screens.

Deliverables: analysis/triage.json (speech_mentions[], speakers[], visual_mentions[], caption_mentions[], notes[]), analysis/triage.md, crops in analysis/triage_frames/. Frame numbers at {FPS} fps. Final reply ≤250 words.
```

## 2. Visual fix (one agent per video, or per group of identical edits)

```
Visual rebrand of one ad video. "{OLD_PRODUCT}"/"{OLD_BRAND}" → "{NEW_NAME}". Fix every on-screen occurrence so it reads {NEW_NAME} (or is cleanly blank where re-lettering would look worse). AUDIO IS NOT YOUR JOB — keep original audio only for the preview.

Job: {JOB}. Inventory: analysis/triage.json + triage.md + triage_frames/ — verify, don't trust blindly.
Items: {LIST FROM TRIAGE, with frame ranges and the approved extra removals}
Do NOT touch: {things the user said to leave, e.g. third-party watermarks}.

Deliverables: project/rebrand_job.py (re-runnable); project/intermediate/rebrand_video.mp4 (video-only, SAME resolution, fps and EXACT frame count as source, H.264 crf ≤16, yuv420p); renders/preview_visual_rebrand.mp4 (with ORIGINAL audio); analysis/visual_qc/ with before/after crops for every item (start/middle/end + every few frames for moving ones) and visual_qc.md listing anything imperfect.

Proven toolkit: {SKILL}/examples/caption-and-tag-library/ and {SKILL}/examples/web-and-sign-library/ (see the README notes in references/lessons.md). Match caption font/size/weight/outline/position — the best match is cutting glyphs from the video's OWN captions. Check for flicker, residual letters and patch halos. Final reply ≤250 words: per item what was done, what is imperfect, output paths.
```

Long jobs (200 s, 8+ items) can exhaust an agent before the full render. Tell it to write the full-render step early and to make the QC step not depend on files created mid-render; if it hands back unfinished, run its `rebrand_job.py` yourself.

## 3. Spoken-word splice (one agent per voice)

```
Splice cloned-voice "{NEW_NAME}" into the narration, replacing every spoken old name. AUDIO ONLY.
Job(s): {JOB}. Mentions: {from triage.json, with refined times}. Stems: analysis/stems/htdemucs/*/{vocals,no_vocals}.wav.
New voice material (private Fish Audio clone, already generated — DO NOT call any Fish API): {TAKE FILES}. Script: "{SCRIPT}" — the Nth {NEW_NAME} maps to mention N.

Reference: {SKILL}/examples/voice-splice/swap_voice_job01.py and references/lessons.md (audio section). Re-measure every old-word span yourself. VERIFY by transcribing a ~5 s window of (new vocals + no_vocals) around every mention, with and without the --prompt; neighbours intact, no leftover "rest"/"vate". Check for clicks.

Deliverables: analysis/stems/vocals_nookwave.wav (same length/rate as the original vocals stem), project/swap_voice.py, project/voice_swap.json (spans, take/word used, tempo, verification transcripts), renders/preview_audio_nookwave.wav. Final reply ≤200 words: per-mention pass/fail.
```

## 4. Sung-word swap

Use `{SKILL}/examples/sung-word/sung_swap.py` as the base: extract the original F0 contour (pyworld/WORLD), map the new word's syllables onto the original notes (e.g. first syllable on the first notes, the long vowel on the sustained note), impose F0 + loudness envelope with WORLD resynthesis (keeps formants, no chipmunk), blend with a short reverb, crossfade 20–30 ms. Ask for F0 error (median semitones), a whisper check of the isolated vocal, and 3–4 alternatives as short excerpts so a human can choose.

## 5. Still image

Per object: find the face corners → homography → cover old text with local background (gradient-aware, inpaint edges) → render new text flat with Pillow in the closest system font → warp back → match blur/JPEG softness. Assert that no pixel outside the replaced regions changed. Base: `{SKILL}/examples/still-image/rebrand_image.py`.
