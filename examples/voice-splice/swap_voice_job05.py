"""Replace the spoken "FisioRest" (job-05, mention S1) in the vocals stem with a cloned-voice "Nookwave".

Usage: swap_voice.py [CANDIDATE]   (CANDIDATE in CANDIDATES, default DEFAULT)
Inputs : ../voice_refs/takes/narratorB_take{1,2}.mp3 (private Fish Audio clone, made by the user)
         analysis/stems/htdemucs/source/{vocals,no_vocals}.wav
Outputs: analysis/stems/vocals_nookwave.wav, renders/preview_audio_nookwave.wav, project/voice_swap.json
Verification: mixes the result, transcribes 1.5-6.5 s with whisper-cli, checks the text.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

J = Path(__file__).resolve().parent.parent
TAKES = J.parent / 'voice_refs/takes'
STEMS = J / 'analysis/stems/htdemucs/source'
TMP = J / 'analysis/voice_tmp'
SR = 44100
WHISPER_MODEL = Path('~/.cache/whisper/ggml-small.en.bin').expanduser()
os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

# Old word "FisioRest", measured on the vocals stem (20 ms frames, energy + >4 kHz share):
# "bought" t-release dies by 3.80 (-42 dB); "f" fricative 3.82-3.94 (HF 0.6-0.8); "st" 4.50-4.56;
# final "t" release 4.60-4.66; "from" voicing starts at 4.68. Triage (3.82-4.66) confirmed.
OLD = (3.81, 4.67)

# "Nookwave" spans in the takes: whisper onset, tightened to start inside the "t" closure of "bought"
# (quietest frame, so no plosive release is carried in and the "N" is intact) and to end after the
# "v" fricative, before the next word's voicing.
CANDIDATES = {
    't1a': (1, 3.62, 4.24),   # whisper 3.65-4.08
    't1b': (1, 6.70, 7.47),   # whisper 6.87-7.25; v fricative runs 7.42-7.49
    't2a': (2, 3.63, 4.26),   # whisper 3.79-4.27
    't2b': (2, 6.91, 7.51),   # whisper 7.03-7.51
}
DEFAULT = 't1b'  # longest natural word: fills the slot so "from" follows with no gap
TEMPO_MIN, TEMPO_MAX = 0.87, 1.15
FADE = int(SR * 0.010)


def run(*cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def load_take(n):
    TMP.mkdir(exist_ok=True)
    out = TMP / f'take{n}.wav'
    run('ffmpeg', '-v', 'error', '-y', '-i', TAKES / f'narratorB_take{n}.mp3', '-ar', SR, '-ac', 1, out)
    x, sr = sf.read(out)
    assert sr == SR
    return x


def quiet_edge(x, t, lo, hi, hop=0.005):
    h = int(SR * hop)
    i0, i1 = int(max(0, t + lo) * SR), int((t + hi) * SR)
    e = [rms(x[i:i + h]) for i in range(i0, i1, h)]
    return (i0 + int(np.argmin(e)) * h) / SR


def transcribe(wav, a, b, tag):
    w16 = TMP / f'{tag}16.wav'
    run('ffmpeg', '-v', 'error', '-y', '-ss', a, '-to', b, '-i', wav, '-ar', 16000, '-ac', 1, w16)
    run('whisper-cli', '-m', WHISPER_MODEL, '-f', w16, '-ml', 1, '-sow', '-ojf', '-of', TMP / tag,
        '--prompt', 'Nookwave')
    segs = json.load(open(TMP / f'{tag}.json'))['transcription']
    words = [(round(a + s['offsets']['from'] / 1000, 2), s['text'].strip()) for s in segs if s['text'].strip()]
    return ' '.join(w for _, w in words), words


def click_check(x, t, win=0.004):
    """Largest sample-to-sample jump within ±win of t, relative to the local median jump."""
    i0, i1 = int((t - win) * SR), int((t + win) * SR)
    d = np.abs(np.diff(x[i0:i1]))
    ref = np.abs(np.diff(x[int((t - 0.05) * SR):int((t + 0.05) * SR)]))
    return round(float(d.max() / (np.percentile(ref, 99) + 1e-9)), 2)


def main():
    cand = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    take_no, ws, we = CANDIDATES[cand]
    take = load_take(take_no)
    ws = quiet_edge(take, ws, -0.01, 0.01)
    we = quiet_edge(take, we, -0.01, 0.01)
    clip = take[int(ws * SR):int(we * SR)].copy()

    vocals, sr = sf.read(STEMS / 'vocals.wav', always_2d=True)
    assert sr == SR
    mono = vocals.mean(axis=1)
    a, b = OLD
    n = int(round((b - a) * SR))

    # stretch toward the slot ("from" follows with no pause), bounded to stay natural
    ratio = len(clip) / n
    tempo = min(TEMPO_MAX, max(TEMPO_MIN, ratio))
    sf.write(TMP / 'c.wav', clip, SR)
    run('ffmpeg', '-v', 'error', '-y', '-i', TMP / 'c.wav', '-af', f'atempo={tempo:.4f}', TMP / 'c2.wav')
    clip, _ = sf.read(TMP / 'c2.wav')
    clip = clip[:n]

    # RMS-match to the preceding second of speech (voiced frames only)
    ctx = mono[int((a - 1.0) * SR):int(a * SR)]
    gain = rms(ctx[np.abs(ctx) > 0.01]) / rms(clip[np.abs(clip) > 0.01])
    clip = clip * gain
    clip[:FADE] *= np.linspace(0, 1, FADE)
    clip[-FADE:] *= np.linspace(1, 0, FADE)

    # crossfade out the old word; new word at slot start, leftover time = short pause before "from"
    out = vocals.copy()
    i0 = int(round(a * SR))
    env = np.zeros(n)
    env[:FADE] = np.linspace(1, 0, FADE)
    env[-FADE:] = np.linspace(0, 1, FADE)
    out[i0:i0 + n] *= env[:, None]
    out[i0:i0 + len(clip)] += clip[:, None]
    assert out.shape == vocals.shape

    dst = J / 'analysis/stems/vocals_nookwave.wav'
    sf.write(dst, out, SR, subtype='PCM_16' if sf.info(STEMS / 'vocals.wav').subtype == 'PCM_16' else 'FLOAT')
    nv, sr2 = sf.read(STEMS / 'no_vocals.wav', always_2d=True)
    assert sr2 == SR
    L = min(len(nv), len(out))
    mix = out[:L] + nv[:L]
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    (J / 'renders').mkdir(exist_ok=True)
    prev = J / 'renders/preview_audio_nookwave.wav'
    sf.write(prev, mix, SR)

    text, words = transcribe(prev, 1.5, 6.5, 'verify')
    norm = text.lower().replace('-', ' ')
    ok = ('nookwave from' in norm.replace('nook wave', 'nookwave')) and 'rest' not in norm
    om = out.mean(axis=1)
    log = dict(
        mention='S1', old_word='FisioRest', new_word='Nookwave',
        old_span_s=[a, b], old_span_frames_30fps=[int(a * 30), int(b * 30)],
        candidate=cand, take=f'voice_refs/takes/narratorB_take{take_no}.mp3',
        take_word_span_s=[round(ws, 3), round(we, 3)], word_s_raw=round(we - ws, 3),
        tempo=round(tempo, 4), word_s_placed=round(len(clip) / SR, 3), slot_s=round(n / SR, 3),
        pause_before_next_word_s=round((n - len(clip)) / SR, 3), gain=round(gain, 3), fade_ms=10,
        click_ratio_at_splices={str(a): click_check(om, a), str(round(a + len(clip) / SR, 3)):
                                click_check(om, a + len(clip) / SR), str(b): click_check(om, b)},
        verification=dict(window_s=[1.5, 6.5], transcript=text, words=words, passed=ok),
    )
    json.dump(log, open(J / 'project/voice_swap.json', 'w'), indent=1)
    print(json.dumps(log, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
