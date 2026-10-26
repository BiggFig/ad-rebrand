"""Render an engagement edit from plan_<version>.json.

Usage: render_edit.py JOB_DIR VERSION [--voice VOICE_WAV] [--preview OUTPUT_FRAME ...]
Picture comes from project/intermediate/rebrand_video.mp4 (already rebranded), audio from the separated stems.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from rebrand import caption_box  # noqa: E402

FPS = 30
OUT_W, OUT_H = 1080, 1350
SRC_W, SRC_H = 720, 900
CREAM, CHARCOAL, COPPER, RED = (250, 245, 234), (31, 29, 27), (184, 107, 60), (214, 69, 69)
FONT_PATH = '/System/Library/Fonts/SFNS.ttf'
_fonts = {}


def font(size, weight='Heavy'):
    key = (size, weight)
    if key not in _fonts:
        f = ImageFont.truetype(FONT_PATH, size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _fonts[key] = f
    return _fonts[key]


# ---------------------------------------------------------------- timeline

class Timeline:
    def __init__(self, intervals):
        self.iv = []
        out = 0
        for a, b in intervals:
            fa, fb = round(a * FPS), round(b * FPS)
            self.iv.append((fa, fb, out))
            out += fb - fa
        self.n = out

    def src_frame(self, k):
        for i, (fa, fb, o) in enumerate(self.iv):
            if o <= k < o + (fb - fa):
                return fa + (k - o), i
        raise IndexError(k)

    def to_out(self, t, clamp=False, lead=0.35):
        """Source seconds -> output frame (None if the moment was cut, unless clamp).

        A moment up to `lead` seconds before a kept interval maps to that interval's first frame,
        since cuts are snapped a few frames into the word.
        """
        f = t * FPS
        for fa, fb, o in self.iv:
            if fa - lead * FPS <= f < fb:
                return int(o + max(0, f - fa))
            if clamp and f < fa:
                return o
        return self.n if clamp else None


# ---------------------------------------------------------------- caption holds

def caption_fixes(tl, changes):
    """Frames at hard cuts that would flash a caption belonging to removed speech.

    Returns {output_frame: donor_source_frame} — the donor's caption box is pasted over the frame.
    """
    fixes = {}
    ch = np.array(changes)
    for i, (fa, fb, o) in enumerate(tl.iv):
        prev_end = tl.iv[i - 1][1] if i else None
        nxt_start = tl.iv[i + 1][0] if i + 1 < len(tl.iv) else None
        # start: old caption still up for a few frames after a real cut
        if prev_end is not None and fa - prev_end > 15:
            c = ch[(ch > fa) & (ch <= fa + 14)]
            if len(c):
                for k in range(fa, int(c[0])):
                    fixes[o + k - fa] = int(c[0]) + 1
        # end: next caption already up before a real cut
        if nxt_start is not None and nxt_start - fb > 15:
            c = ch[(ch < fb) & (ch >= fb - 10)]
            if len(c):
                for k in range(int(c[-1]), fb):
                    fixes[o + k - fa] = int(c[-1]) - 2
    return fixes


def paste_caption(frame, donor):
    bd, bf = caption_box(donor), caption_box(frame)
    if bd is None:
        return frame
    boxes = [bd] + ([bf] if bf is not None else [])
    x0 = max(0, min(b[0] for b in boxes) - 3)
    x1 = min(SRC_W, max(b[2] for b in boxes) + 4)
    y0 = max(0, min(b[1] for b in boxes) - 3)
    y1 = min(SRC_H, max(b[3] for b in boxes) + 4)
    if bf is not None and (bf[0] < bd[0] - 2 or bf[2] > bd[2] + 2):
        # new box is wider: paste the donor band so no edge of the wrong caption peeks out
        frame[y0:y1, x0:x1] = donor[y0:y1, x0:x1]
    else:
        frame[bd[1] - 2:bd[3] + 3, bd[0] - 2:bd[2] + 3] = donor[bd[1] - 2:bd[3] + 3, bd[0] - 2:bd[2] + 3]
    return frame


# ---------------------------------------------------------------- picture

def zoom_for(k, tl, punches_out):
    src, i = tl.src_frame(k)
    fa, fb, o = tl.iv[i]
    base = 1.0 if i % 2 == 0 else 1.07
    drift = 1.0 + 0.03 * (k - o) / max(1, fb - fa)
    punch = 1.10 if any(a <= k < b for a, b in punches_out) else 1.0
    return min(1.17, base * drift * punch)


def frame_to_out(frame, z, cy=450):
    cw, ch = SRC_W / z, SRC_H / z
    x0 = (SRC_W - cw) / 2
    y0 = min(max(cy - ch / 2, 0), SRC_H - ch)
    M = np.float32([[OUT_W / cw, 0, -x0 * OUT_W / cw], [0, OUT_H / ch, -y0 * OUT_H / ch]])
    return cv2.warpAffine(frame, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def ease_out_back(x):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def pill(text, size, fg, bg, pad=(34, 18), weight='Heavy', bg_alpha=245, icon=None):
    f = font(size, weight)
    l, t, r, b = f.getbbox(text)
    tw, th = r - l, b - t
    icon_w = int(th * 1.15) + 16 if icon else 0
    W, H = tw + 2 * pad[0] + icon_w, th + 2 * pad[1]
    im = Image.new('RGBA', (W + 16, H + 20), (0, 0, 0, 0))
    sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((8, 12, W + 8, H + 12), radius=H // 2, fill=(0, 0, 0, 70))
    im.alpha_composite(sh)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 6, W + 8, H + 6), radius=H // 2, fill=bg + (bg_alpha,))
    x = 8 + pad[0]
    if icon == 'x':
        r_ = th * 0.58
        cx, cy = x + r_, 6 + H / 2
        d.ellipse((cx - r_, cy - r_, cx + r_, cy + r_), fill=RED + (255,))
        q = r_ * 0.45
        d.line((cx - q, cy - q, cx + q, cy + q), fill=(255, 255, 255, 255), width=max(3, size // 9))
        d.line((cx - q, cy + q, cx + q, cy - q), fill=(255, 255, 255, 255), width=max(3, size // 9))
        x += icon_w
    d.text((x - l, 6 + pad[1] - t), text, font=f, fill=fg + (255,))
    return im


STYLES = {
    'hook': dict(size=66, fg=CHARCOAL, bg=CREAM, y=128),
    'punch': dict(size=54, fg=CREAM, bg=CHARCOAL, y=140),
    'brand': dict(size=84, fg=COPPER, bg=CREAM, y=128),
    'chip': dict(size=50, fg=(255, 255, 255), bg=COPPER, y=140),
    'offer': dict(size=70, fg=(255, 255, 255), bg=COPPER, y=124),
    'cta': dict(size=64, fg=(255, 255, 255), bg=COPPER, y=128),
    'check': dict(size=44, fg=CHARCOAL, bg=CREAM, y=120),
}


def schedule_callouts(callouts, tl):
    """Place callouts on the output timeline; drop ones whose words were cut."""
    out = []
    groups = {}
    for c in callouts:
        s = tl.to_out(c['t'])
        if s is None:
            continue
        if 'end' in c:
            e = tl.to_out(c['end'], clamp=True)
        else:
            e = s + int(c.get('dur', 2.0) * FPS)
        item = dict(c, s=s, e=max(e, s + 12))
        if c.get('group'):
            groups.setdefault(c['group'], []).append(item)
        out.append(item)
    for items in groups.values():  # checklist items share the group's end and stack in order
        end = max(i['e'] for i in items)
        for n, i in enumerate(items):
            i['e'], i['slot'] = end, n
    # one headline at a time: a new callout cuts the previous one short (offers stack instead)
    solo = sorted((c for c in out if c['style'] not in ('check', 'offer')), key=lambda c: c['s'])
    for a, b in zip(solo, solo[1:]):
        a['e'] = min(a['e'], b['s'])
    offers = [c for c in out if c['style'] == 'offer']
    for n, c in enumerate(offers):
        c['slot'] = 1 if n and offers[n - 1]['e'] > c['s'] else 0
    return out


def draw_overlays(frame_bgr, k, callouts, total, progress=True):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
    for c in callouts:
        if not c['s'] <= k < c['e']:
            continue
        st = STYLES[c['style']]
        age = k - c['s']
        left = c['e'] - k
        p = min(1.0, age / 6)
        scale = 0.6 + 0.4 * ease_out_back(p)
        alpha = min(1.0, age / 3, left / 4)
        if c['style'] == 'check':
            im = pill(c['text'], st['size'], st['fg'], st['bg'], icon='x', pad=(22, 14))
            x, y = 48, st['y'] + c.get('slot', 0) * 92
        else:
            text = c['text']
            im = pill(text, st['size'], st['fg'], st['bg'])
            x = (OUT_W - im.width) // 2
            y = st['y'] + c.get('slot', 0) * 128
            if c['style'] == 'cta':
                y += int(10 * np.sin(age / 4.0))
        if scale != 1.0:
            w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
            cx, cy = x + im.width / 2, y + im.height / 2
            im = im.resize((w, h), Image.LANCZOS)
            x, y = int(cx - w / 2), int(cy - h / 2)
        if alpha < 1:
            a = np.array(im)
            a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
            im = Image.fromarray(a)
        img.alpha_composite(im, (max(0, x), max(0, y)))
    if progress:
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, OUT_W, 9), fill=(0, 0, 0, 90))
        d.rectangle((0, 0, int(OUT_W * (k + 1) / total), 9), fill=COPPER + (255,))
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- audio

SR = 44100


def sfx_whoosh(dur=0.32, seed=1):
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    # sweep a resonant band-pass upward
    f = 300 + 3200 * t ** 1.5
    y1 = y2 = 0.0
    for i in range(n):
        w = 2 * np.pi * f[i] / SR
        r = 0.97
        y = noise[i] * (1 - r) + 2 * r * np.cos(w) * y1 - r * r * y2
        out[i], y2, y1 = y, y1, y
    env = np.sin(np.pi * t) ** 1.6
    out = out * env
    return out / (np.abs(out).max() + 1e-9)


def sfx_pop(freq=880, dur=0.09):
    t = np.arange(int(SR * dur)) / SR
    tone = np.sin(2 * np.pi * freq * t * (1 + 0.6 * np.exp(-t * 60)))
    return tone * np.exp(-t * 45)


def sfx_boom(dur=0.6):
    t = np.arange(int(SR * dur)) / SR
    body = np.sin(2 * np.pi * (48 + 60 * np.exp(-t * 18)) * t) * np.exp(-t * 6)
    click = np.random.default_rng(3).standard_normal(len(t)) * np.exp(-t * 120) * 0.35
    out = body + click
    return out / np.abs(out).max()


def db(x):
    return 10 ** (x / 20)


def build_audio(tl, voice_path, music_path, callouts, out_wav):
    voice, sr1 = sf.read(voice_path, always_2d=True)
    music, sr2 = sf.read(music_path, always_2d=True)
    assert sr1 == sr2 == SR
    fade = int(SR * 0.008)
    ramp_in = np.linspace(0, 1, fade)[:, None]
    parts_v, parts_m = [], []
    for fa, fb, _ in tl.iv:
        a, b = int(fa / FPS * SR), int(fb / FPS * SR)
        for src, parts in ((voice, parts_v), (music, parts_m)):
            seg = src[a:b].copy()
            seg[:fade] *= ramp_in
            seg[-fade:] *= ramp_in[::-1]
            parts.append(seg)
    v = np.concatenate(parts_v)
    m = np.concatenate(parts_m)
    n = int(tl.n / FPS * SR)
    v, m = v[:n], m[:n]
    fx = np.zeros((n, 2))

    def add(sig, at_frame, gain_db):
        i = int(at_frame / FPS * SR)
        s = sig[: max(0, n - i)] * db(gain_db)
        fx[i:i + len(s)] += s[:, None]

    whoosh = sfx_whoosh()
    for i in range(1, len(tl.iv)):
        if tl.iv[i][0] - tl.iv[i - 1][1] > 15:  # a real cut, not a tightened pause
            add(whoosh, max(0, tl.iv[i][2] - 5), -24)
    pop, boom = sfx_pop(), sfx_boom()
    for c in callouts:
        if c['style'] in ('brand', 'offer'):
            add(boom, c['s'], -14)
        elif c['style'] == 'check':
            add(sfx_pop(1200, 0.06), c['s'], -18)
        else:
            add(pop, c['s'], -19)
    mix = v + m + fx
    sf.write(out_wav, mix.astype(np.float32), SR)


# ---------------------------------------------------------------- main

def main():
    J, version = Path(sys.argv[1]), sys.argv[2]
    plan = json.load(open(J / f'plan_{version}.json'))
    voice = J / 'analysis/stems/htdemucs/source_audio/vocals.wav'
    if '--voice' in sys.argv:
        voice = Path(sys.argv[sys.argv.index('--voice') + 1])
    music = J / 'analysis/stems/htdemucs/source_audio/no_vocals.wav'
    picture = J / 'project/intermediate/rebrand_video.mp4'

    tl = Timeline(plan['intervals'])
    changes = json.load(open(J / 'analysis/caption_changes2.json'))
    fixes = caption_fixes(tl, changes)
    callouts = schedule_callouts(plan['callouts'], tl)
    punches_out = []
    for a, b in plan['punches']:
        s, e = tl.to_out(a), tl.to_out(b, clamp=True)
        if s is not None:
            punches_out.append((s, e))

    previews = None
    if '--preview' in sys.argv:
        previews = set(map(int, sys.argv[sys.argv.index('--preview') + 1:]))

    # Map output frames to source frames; read the source sequentially once.
    need = {}
    for k in range(tl.n):
        if previews is not None and k not in previews:
            continue
        src, _ = tl.src_frame(k)
        need.setdefault(src, []).append(k)
    donors = {d for k, d in fixes.items() if previews is None or k in previews}

    out_dir = J / 'renders'
    out_dir.mkdir(exist_ok=True)
    tmp_video = out_dir / f'_{version}_video.mp4'
    ff = None
    if previews is None:
        ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                               '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                               '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', str(tmp_video)],
                              stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(picture))
    donor_frames, pending = {}, {}
    next_k = 0 if previews is None else min(previews)
    src_i = 0
    last_needed = max(max(need), max(donors) if donors else 0)
    while src_i <= last_needed:
        ok, f = cap.read()
        if not ok:
            break
        if src_i in donors:
            donor_frames[src_i] = f.copy()
        for k in need.get(src_i, []):
            pending[k] = f.copy()
        # emit frames in order once their donors are available
        while next_k in pending and (fixes.get(next_k) is None or fixes[next_k] in donor_frames):
            fr = pending.pop(next_k)
            if next_k in fixes:
                fr = paste_caption(fr, donor_frames[fixes[next_k]])
            fr = frame_to_out(fr, zoom_for(next_k, tl, punches_out))
            fr = draw_overlays(fr, next_k, callouts, tl.n)
            if ff:
                ff.stdin.write(fr.tobytes())
            else:
                cv2.imwrite(str(out_dir / f'preview_{version}_{next_k}.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            next_k += 1
            while previews is not None and next_k < tl.n and next_k not in previews:
                next_k += 1
        src_i += 1
    if ff is None:
        return
    ff.stdin.close()
    ff.wait()
    assert next_k == tl.n, (next_k, tl.n)

    wav = out_dir / f'_{version}_mix.wav'
    build_audio(tl, voice, music, callouts, wav)
    final = out_dir / f'Nookwave_{version}.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(tmp_video), '-i', str(wav),
                    '-af', 'loudnorm=I=-14:TP=-1.0:LRA=11', '-ar', '48000',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
                    '-shortest', str(final)], check=True)
    json.dump(dict(version=version, frames=tl.n, seconds=round(tl.n / FPS, 2),
                   callouts=[{k: c[k] for k in ('text', 's', 'e')} for c in callouts],
                   caption_holds=len(fixes)), open(out_dir / f'_{version}_render.json', 'w'), indent=1)
    print('wrote', final, tl.n / FPS)


if __name__ == '__main__':
    main()
