---
name: ad-rebrand
description: Rebrand existing video ads and ad images after a product or brand rename — replace every spoken, captioned and on-screen mention of the old names (logos, leather tags, device labels, packaging, bags, web screenshots, signs, end cards) with the new name, using cloned voices for speech and a team of subagents. Use when the user hands over ad videos/images and says the brand or product was renamed ("we rebranded from X to Y", "swap the old name out", "fix these for the new brand"). Optionally also produces punched-up engagement cuts.
---

# Ad rebrand

Turn a batch of existing ads into the new brand: every **spoken** mention re-voiced in the original speaker's cloned voice, every **caption** re-lettered in the video's own caption font, every **on-screen** logo/tag/label re-lettered or cleanly blanked — with the rest of each ad untouched (same frames, timing, music).

Parallelise with subagents (Agent tool, `run_in_background: true`, one message launching all independent agents). Prompt templates: `references/agent-prompts.md`. Read `references/lessons.md` before starting — it records what broke last time.

## Inputs to pin down first
- Old names (product and brand) and the new name — and how whisper might mishear the old ones (e.g. "physio rest", "artivate").
- The files (videos, images). Work in a project folder the user can keep (e.g. ~/Downloads/<Brand> Video Edits); never edit the originals.

## Workflow

1. **Set up.** `scripts/setup_jobs.sh PROJECT START_N video1 video2 …` → `job-NN/{assets/source.mp4,analysis,project,renders}`. Copy images to `PROJECT/images/source/`. Environment: ffmpeg/whisper-cli via Homebrew, Python venv `~/.cache/video-studio-venv` (numpy, opencv-contrib, pillow, soundfile, demucs, torch; pyworld for sung words), whisper model `~/.cache/whisper/ggml-small.en.bin`. If missing, install (Homebrew needs the user's password in a terminal tab).

2. **Triage (parallel, one agent per video).** Template §1. Each writes `analysis/triage.json|md` with refined speech spans, speakers + clean clone-reference ranges, every visual/caption occurrence with frame ranges, and oddities. Check images yourself (zoom into tags/labels at full res) — many need nothing.

3. **Consolidate and ask once** (AskUserQuestion):
   - Voice rights: list every distinct voice that must be cloned (group videos that share narration — compare vocal-stem envelopes; compare median F0 to tell speakers apart). Cloning requires the user's explicit yes per voice.
   - Anything that is not the old brand but looks like a problem: other brands' marks on the product, retail/endorsement settings (store staff, store logos), inconsistent claims, script redundancy from the rename. Third-party watermarks: flag, never remove.
   - Sung mentions: explain they need re-pitching and will need a human listen.

4. **Start the visual team immediately** (template §2) — it doesn't depend on voices. Group videos that are the same edit into one agent. Image agent: template §5.

5. **Voices.** Export clone references from each job's vocals stem (25–35 s, no brand names) into `PROJECT/voice_refs/`. Prefer the Fish API (`examples/voice-splice/fish_tts_api.py`, key in `~/.config/video-studio/.env`) if it has API credit; otherwise create **private** clones in the Fish web app in the user's Chrome — `references/fish-audio-web.md`. Generate each voice's lines in one script, both takes, saved to `voice_refs/takes/`.

6. **Audio team (parallel, one agent per voice/job group).** Template §3 for speech, §4 for sung words. Output: `analysis/stems/vocals_nookwave.wav` per job, verified by transcription.

7. **Assemble each job** as soon as its picture and voice are both ready: `scripts/mux_final.py JOB OUT_NAME.mp4` (rebranded picture + new vocals + original music stem). If a visual agent hands back unfinished, run its `project/rebrand_job.py` yourself.

8. **QC every final yourself** — agents' reports are claims, not proof:
   `scripts/check_final.py source.mp4 final.mp4 --old "physio|fisio|artuvate|artivate" --new "nook" --frames <one frame per fixed item>` → frame-count match, clean decode, loudness vs source, zero spoken old names, and a source|final sheet you must look at. Spot-transcribe 5 s windows around each mention if totals look off (full-file whisper merges sentences). For sung lines expect whisper to struggle on the original too.

9. **Deliver.** Finals at the project root with descriptive names, a zip if asked (`zip -0`, videos are already compressed), and a summary: what changed per file, what needs a human listen (sung words, pause lengths after a shorter name), known imperfect frames, and the flagged non-brand issues.

## Optional: engagement cuts
For "make it more engaging" requests on top of the rebrand, `examples/engagement-edit/` holds the plan builder (sentence-level keep lists snapped to quiet points, callouts, punch-ins) and renderer (zoom alternation on jump cuts, branded callouts, progress bar, synthesized SFX, caption-hold fixes at cuts, loudnorm −14 LUFS). Reuse rather than rewrite.

## Bundled files
- `scripts/setup_jobs.sh`, `scripts/mux_final.py`, `scripts/check_final.py` — generic.
- `examples/` — working, job-specific code from the first run (paths and frame numbers are hard-coded; adapt, don't run blind):
  - `caption-and-tag-library/` — caption redraw, rotating leather tags, curved plates, bag eraser, multi-scale tracking (`rebrand_job01.py`); glyph-cut captions, engraved/debossed tags, tracked bag re-print (`caption_lib.py`, `text_lib.py`, `track_lib.py`, `rebrand_job05.py`).
  - `web-and-sign-library/` — own-glyph captions (`captions.py`), building-sign plate, web copy re-set, tag pressing, hand-keyed tumbling bag (`rbk.py`, `rebrand_job02.py`), laptop-screen page (`rebrand_job06_laptop.py`), review-page scroll, end-card logo, third-party emboss removal (`rebrand_job07.py`), QC sheets (`visual_qc.py`).
  - `voice-splice/` — measured-span word replacement with take overrides; Fish API client.
  - `sung-word/` — WORLD-based re-singing onto the original melody.
  - `still-image/` — homography text replacement on packaging.
