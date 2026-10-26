# Creating voice clones in the Fish Audio web app (user's Chrome)

Use when the Fish API has no credit (check: `GET https://api.fish.audio/wallet/self/api-credit` with the key → `credit`). Platform (plan) credit can only be spent in the web app.

Before starting: the user must be signed in to fish.audio in Chrome with Claude in Chrome connected. If several Chrome browsers are connected, ask which (AskUserQuestion, list every browser + the "confirmation screen in every extension" option, then `switch_browser`/`select_browser`). Never type the user's password.

**Permission gate:** the form has a required box "I confirm I have permission to create and share this voice model" (also accepts Fish's Terms). Only tick it after the user has explicitly confirmed they have rights to clone **that specific voice**. Ask once per batch, listing every voice. Always set Type = **Private**.

## Flow (per voice; coordinates are for a 1512x711 viewport and may drift — screenshot first)
1. Navigate `https://fish.audio/app/voice-cloning/` → wait 3 s → `find` "audio file upload input" → `file_upload` the reference WAV (never click the Upload button; it opens a native picker).
2. Click **Analyze Audio** (right column) → wait ~20 s. A "good" badge means proceed. A "Multiple speakers detected" warning is normal for songs: tick "I understand…" and continue.
3. Click **Continue to details** → set Voice Name (triple-click the field, type e.g. "Brand Narrator A jobs02-03 (private)").
4. Scroll down 10 ticks → click **Private** → screenshot (layout shifts) → tick the permission box (only with user consent) → zoom to verify → **Create**.
5. On the voice page, click **Use Voice** → Text to Speech opens with the new voice in the first speaker block. Click the block, cmd+A, type the script containing every new-name line in order → **Generate Speech** → wait ~15 s.
6. Two takes appear (free). Before downloading, `touch` a marker file; click each take's download icon; then move `find ~/Downloads -newer MARKER -name "*.mp3"` into `voice_refs/takes/<voice>_take<N>.mp3`.
7. Close the tab group's tabs when done.

## Script writing
- Put every replacement in natural context, in mention order, e.g. "It's called Nookwave. Nookwave uses gentle traction. …".
- Add a spare sentence-initial "Nookwave" — crisp onsets make good substitutes for mushy in-context ones.
- For sung lines, generate the spoken line; the sung-word agent re-pitches it.
