"""Build the edit plans (kept source intervals, punch-ins, callouts, SFX) for the full and short cuts."""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

J = Path(sys.argv[1])
words = json.load(open(J / 'analysis/words_clean.json'))

# Sentence table (index -> [start, end]) built from word timings.
sentences, cur = [], []
for a, b, t in words:
    cur.append((a, b, t))
    if t.endswith(('.', '?', '!')):
        sentences.append(cur)
        cur = []
if cur:
    sentences.append(cur)
S = [(s[0][0], s[-1][1]) for s in sentences]


def word_time(phrase, after=0.0, edge='start'):
    """Start (or end) time of the first occurrence of `phrase` (lowercase words) after `after`."""
    toks = phrase.lower().split()
    clean = [w[2].lower().strip('.,?!"') for w in words]
    for i in range(len(words) - len(toks) + 1):
        if words[i][0] >= after and clean[i:i + len(toks)] == toks:
            return words[i][0] if edge == 'start' else words[i + len(toks) - 1][1]
    raise KeyError(phrase)


# Voice energy, used to snap every cut to the quietest point near the boundary.
voc, sr = sf.read(J / 'analysis/stems/htdemucs/source_audio/vocals.wav')
voc = voc.mean(axis=1) if voc.ndim > 1 else voc
hop = int(sr * 0.01)
rms = np.sqrt(np.convolve(voc ** 2, np.ones(hop * 3) / (hop * 3), mode='same'))[::hop]


def snap(t, win=0.12):
    i0, i1 = max(0, int((t - win) * 100)), min(len(rms) - 1, int((t + win) * 100))
    return (i0 + int(np.argmin(rms[i0:i1 + 1]))) / 100.0


def quiet_spans(a, b, min_len=0.30, thresh_db=-38):
    """Pauses inside [a, b] long enough to tighten."""
    db = 20 * np.log10(rms + 1e-9)
    spans, start = [], None
    for i in range(int(a * 100), int(b * 100)):
        if db[i] < thresh_db:
            start = i if start is None else start
        else:
            if start is not None and (i - start) / 100 >= min_len:
                spans.append((start / 100, i / 100))
            start = None
    return spans


def intervals_from(keep):
    """keep: list of (start, end) source times -> merged, snapped, pause-tightened intervals."""
    out = []
    for a, b in keep:
        a, b = snap(a), snap(b)
        if out and abs(out[-1][1] - a) < 0.05:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    tightened = []
    for a, b in out:
        cursor = a
        for qa, qb in quiet_spans(a, b):
            tightened.append((cursor, qa + 0.06))
            cursor = qb - 0.06
        tightened.append((cursor, b))
    return [(round(a, 3), round(b, 3)) for a, b in tightened if b - a > 0.08]


def sent(*idx):
    return [S[i] for i in idx]


# FULL: every line except the weakest repeats/asides.
full_drop = {6, 17, 23, 24, 29, 45, 47, 60, 63}
full_keep = [S[i] for i in range(len(S)) if i not in full_drop and i != 62]
full_keep.insert(sum(1 for i in range(62) if i not in full_drop),
                 (S[62][0], word_time('off that nerve', after=200, edge='end')))
full_keep.sort()

# SHORT: hook -> failed fixes -> real cause -> product -> how -> proof -> offer -> CTA.
short_keep = (
    sent(0, 1, 10, 11, 12, 13, 14, 16, 20, 21, 22, 38)
    + [(word_time('what if you could', after=120), S[41][1])]
    + sent(43, 44, 51, 52)
    + [(S[62][0], word_time('off that nerve', after=200, edge='end'))]
    + sent(66, 68, 70)
)
short_keep.sort()

# Overlays keyed to source time (they follow their words through the edit; dropped if the words are cut).
CALLOUTS = [
    dict(t=0.15, end=4.1, text='SHOULDER BLADE KNOT?', style='hook'),
    dict(t=word_time('stretches every', after=30), text='Stretches', style='check', group='tried'),
    dict(t=word_time('massages for', after=30), text='Massages', style='check', group='tried'),
    dict(t=word_time('a tennis ball', after=30), text='Tennis ball', style='check', group='tried'),
    dict(t=word_time('a theragun', after=30), text='Theragun', style='check', group='tried'),
    dict(t=word_time('heating pads', after=30), end=S[16][1], text='Heating pads', style='check', group='tried'),
    dict(t=word_time('they start in your neck', after=55), dur=1.8, text='IT STARTS IN YOUR NECK', style='punch'),
    dict(t=word_time('your neck does', after=90), dur=1.4, text='YOUR NECK DOES.', style='punch'),
    dict(t=word_time('the harder you dig', after=100), dur=2.6, text='DIG HARDER = IT CLAMPS HARDER', style='punch'),
    dict(t=word_time("it's called", after=130), dur=2.2, text='NOOKWAVE', style='brand'),
    dict(t=word_time('under your neck', after=148), dur=2.0, text='UNDER THE NECK  —  NOT THE BLADE', style='punch'),
    dict(t=word_time('gentle traction', after=150), dur=3.0, text='26° GENTLE TRACTION', style='chip'),
    dict(t=word_time('targeted heat', after=160), dur=3.0, text='TARGETED HEAT', style='chip'),
    dict(t=word_time('focused vibration', after=165), dur=3.4, text='FOCUSED VIBRATION', style='chip'),
    dict(t=word_time('15 minutes', after=170), dur=2.4, text='15 MIN A DAY', style='chip'),
    dict(t=word_time('over 100,000', after=195), dur=3.0, text='100,000+ USERS', style='punch'),
    dict(t=word_time('55%', after=215), dur=2.8, text='55% OFF', style='offer'),
    dict(t=word_time('365', after=215), dur=3.2, text='365-DAY GUARANTEE', style='offer'),
    dict(t=word_time("you don't pay", after=230), dur=1.6, text="DOESN'T HELP? DON'T PAY.", style='punch'),
    dict(t=word_time('click below', after=235), end=242.9, text='TAP BELOW  ↓', style='cta'),
]
# Hard punch-ins on emphasis lines (source ranges), on top of the alternating jump-cut zoom.
PUNCHES = [
    (S[1][0], S[1][1]),                              # "You're making it worse."
    (S[22][0], S[22][1]),                            # "They start in your neck."
    (S[33][0], S[33][1]),                            # "Your neck does."
    (S[38][0], S[38][1]),                            # "The harder you dig..."
    (S[43][0], S[43][1]),                            # "It's called Nookwave."
    (S[68][0], S[68][1]),                            # "If it doesn't help, you don't pay."
]

for name, keep in (('full', full_keep), ('short', short_keep)):
    iv = intervals_from(keep)
    plan = dict(version=name, fps=30, intervals=iv, callouts=CALLOUTS, punches=PUNCHES,
                duration=round(sum(b - a for a, b in iv), 2))
    json.dump(plan, open(J / f'plan_{name}.json', 'w'), indent=1)
    print(name, len(iv), 'cuts', plan['duration'], 's')
