"""Replace each spoken "FisioRest" in the separated voice stem with a cloned-voice "Nookwave".

Usage: swap_voice.py JOB_DIR [TAKE]
Uses assets/voice/tts/web_take<TAKE>.wav (all four lines generated in the Fish Audio web app with the
private "Nookwave Ad Doctor" clone) plus words_take<TAKE>.json from whisper-cli; TAKE defaults to 1.
Writes analysis/stems/vocals_nookwave.wav and project/voice_swap.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

J = Path(sys.argv[1])
SR = 44100
WHISPER_MODEL = Path('~/.cache/whisper/ggml-small.en.bin').expanduser()

# (source span of "physio rest", line to synthesise so the word gets natural prosody)
# Source span of the spoken "physio rest" (onset of "ph" to release of the final "t"), measured from
# the voice stem's energy and high-frequency (fricative) profile; whisper's word times were up to 0.5 s off.
MENTIONS = [
    ((133.05, 134.16), "It's called Nookwave."),
    ((155.81, 156.83), 'Nookwave uses gentle traction'),
    ((203.74, 204.65), 'are now using Nookwave to take the pressure off'),
    ((221.92, 222.75), 'Nookwave is now available'),
]


def run(*cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def snap_edges(v, a, b, win=0.08):
    """Move span edges to the quietest 10 ms within ±win so the splice lands between phonemes."""
    def q(t):
        i0, i1 = int((t - win) * SR), int((t + win) * SR)
        hop = int(SR * 0.01)
        e = [rms(v[i:i + hop]) for i in range(i0, i1, hop)]
        return (i0 + int(np.argmin(e)) * hop) / SR
    return q(a), q(b)


def word_span(wav, word='nookwave'):
    out = J / 'analysis/voice_tmp'
    run('ffmpeg', '-v', 'error', '-y', '-i', wav, '-ar', 16000, '-ac', 1, out / 'w16.wav')
    run('whisper-cli', '-m', WHISPER_MODEL, '-f', out / 'w16.wav', '-ml', 1, '-sow', '-ojf',
        '-of', out / 'w', '--prompt', 'Nookwave')
    segs = json.load(open(out / 'w.json'))['transcription']
    for i, s in enumerate(segs):
        t = s['text'].strip().lower().strip('.,')
        if t.startswith('nook') or t.startswith('nuke') or t.startswith('nu'):
            start = s['offsets']['from'] / 1000
            end = s['offsets']['to'] / 1000
            # whisper may split "Nook wave" into two tokens
            if i + 1 < len(segs) and segs[i + 1]['text'].strip().lower().startswith('wave'):
                end = segs[i + 1]['offsets']['to'] / 1000
            return start, end
    raise RuntimeError(f'"Nookwave" not found in {wav}: {[s["text"] for s in segs]}')


def fit(clip, target_len):
    """Time-stretch clip (mono) to target_len samples with ffmpeg atempo, bounded to keep it natural."""
    ratio = len(clip) / target_len
    ratio = min(1.35, max(0.75, ratio))
    tmp = J / 'analysis/voice_tmp'
    sf.write(tmp / 'c.wav', clip, SR)
    run('ffmpeg', '-v', 'error', '-y', '-i', tmp / 'c.wav', '-af', f'atempo={ratio:.4f}', tmp / 'c2.wav')
    out, _ = sf.read(tmp / 'c2.wav')
    if len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    return out[:target_len]


# take 1, mention 0: "called Nookwave" runs together in the take (the "Nook" comes out as "d-wave"),
# so borrow the crisp sentence-initial "Nookwave" (before "uses") for that slot.
WORD_OVERRIDES = {(1, 0): (1.32, 1.91), (1, 1): (1.32, 1.91)}


def take_words(take):
    segs = json.load(open(J / f'assets/voice/tts/words_take{take}.json'))['transcription']
    spans = []
    for s in segs:
        if s['text'].strip().lower().strip('.,').startswith('nookwave'):
            spans.append((s['offsets']['from'] / 1000, s['offsets']['to'] / 1000))
    assert len(spans) == len(MENTIONS), spans
    return spans


def trim_word(take, a, b):
    """Tighten whisper's word span to the quiet points around the word."""
    mono = take
    hop = int(SR * 0.005)

    def quiet(t, lo, hi):
        i0, i1 = int(max(0, t + lo) * SR), int((t + hi) * SR)
        e = [rms(mono[i:i + hop]) for i in range(i0, i1, hop)]
        return (i0 + int(np.argmin(e)) * hop) / SR
    a2 = quiet(a, -0.12, 0.0)   # never start after whisper's onset: that clips the "N"
    b2 = quiet(b, -0.06, 0.08)
    return a2, b2


def main():
    take_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    take, sr = sf.read(J / f'assets/voice/tts/web_take{take_no}.wav')
    assert sr == SR
    spans = take_words(take_no)
    vocals, sr = sf.read(J / 'analysis/stems/htdemucs/source_audio/vocals.wav', always_2d=True)
    assert sr == SR
    mono = vocals.mean(axis=1)
    out = vocals.copy()
    log = []
    tmp = J / 'analysis/voice_tmp'
    tmp.mkdir(exist_ok=True)
    for i, (((a, b), line), (ws, we)) in enumerate(zip(MENTIONS, spans)):
        ws, we = trim_word(take, ws, we)
        if (take_no, i) in WORD_OVERRIDES:  # hand-checked edges where the take runs words together
            ws, we = WORD_OVERRIDES[(take_no, i)]
        clip = take[int(ws * SR):int(we * SR)].copy()
        n = int((b - a) * SR)
        ratio = len(clip) / n
        if abs(ratio - 1) > 0.03:  # fit the slot, within ±15% so the word still sounds natural
            sf.write(tmp / 'c.wav', clip, SR)
            r = min(1.15, max(0.87, ratio))
            run('ffmpeg', '-v', 'error', '-y', '-i', tmp / 'c.wav', '-af', f'atempo={r:.4f}', tmp / 'c2.wav')
            clip, _ = sf.read(tmp / 'c2.wav')
        clip = clip[:n]
        # match loudness to the surrounding speech
        ctx = mono[int((a - 1.0) * SR):int(a * SR)]
        clip = clip * rms(ctx[np.abs(ctx) > 0.01]) / rms(clip[np.abs(clip) > 0.01])
        fade = int(SR * 0.010)
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
        # crossfade out the old word, then lay the new one at the slot start (remaining time = pause)
        i0 = int(a * SR)
        env = np.zeros(n)
        env[:fade] = np.linspace(1, 0, fade)
        env[-fade:] = np.linspace(0, 1, fade)
        out[i0:i0 + n] *= env[:, None]
        out[i0:i0 + len(clip)] += clip[:, None]
        log.append(dict(line=line, source_span=[round(a, 3), round(b, 3)], take=take_no,
                        take_word=[round(ws, 3), round(we, 3)], slot_s=round(n / SR, 3),
                        word_s=round(len(clip) / SR, 3), tempo=round(min(1.15, max(0.87, ratio)), 3)))
        print('swapped', line, log[-1])
    sf.write(J / 'analysis/stems/vocals_nookwave.wav', out, SR)
    json.dump(log, open(J / 'project/voice_swap.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
