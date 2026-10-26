# ad-rebrand

A Claude Code skill for rebranding existing video ads and ad images after a product or brand rename. It replaces every spoken, captioned and on-screen mention of the old names with the new one, keeps everything else in the ad unchanged, and uses a team of subagents to work through a batch in parallel.

Built from a real batch: 7 ads and 3 images moved from "Artuvate / FisioRest" to "Nookwave".

## What it does

| Mention type | How it's fixed |
|---|---|
| Spoken (voiceover) | Private voice clone of the original speaker says the new name; the word is spliced into the separated voice stem at measured word edges and verified by transcription |
| Sung | Same clone, re-pitched onto the original melody with WORLD resynthesis |
| Burned-in captions | Re-lettered using glyphs cut from the video's own captions |
| Logos and labels | Tracked and re-lettered (leather tags on rotating renders, curved device plates, packaging, signs, web screenshots, end cards) or cleanly erased (fabric bags) |
| Still images | Homography text replacement on packaging |

It also flags things it should not fix silently: other brands' marks, third-party watermarks, retail and endorsement settings, and conflicting claims.

## Install

```bash
git clone https://github.com/BiggFig/ad-rebrand ~/.claude/skills/ad-rebrand
```

Requirements (macOS): Homebrew `ffmpeg` and `whisper-cpp` (model `~/.cache/whisper/ggml-small.en.bin`), and a Python 3.12 venv at `~/.cache/video-studio-venv` with `demucs soundfile opencv-contrib-python-headless pillow numpy requests ormsgpack pyworld`. For voices you need a Fish Audio account: the API (key in `~/.config/video-studio/.env` as `FISH_AUDIO_API_KEY`) or the web app driven through Claude in Chrome.

## Use

In Claude Code, attach the ads and say something like "we rebranded from X to Y, fix these", or run `/ad-rebrand`.

## Layout

- `SKILL.md`: the workflow (setup, triage, decisions, voices, visual and audio agent teams, assembly, QC, delivery).
- `references/`: subagent prompt templates, lessons learned, and the Fish Audio web cloning guide.
- `scripts/`: `setup_jobs.sh`, `mux_final.py` and `check_final.py` (frame count, decode, loudness, spoken old-name count, before/after sheet).
- `examples/`: working code from the first batch. Frame numbers and names are specific to those ads, so adapt them rather than running them as they are.

Voice cloning requires the speaker's consent or rights to the voice; the workflow asks for confirmation for each voice before creating a clone.
