"""Rebrand pass: remove Artuvate logo text and swap FisioRest -> Nookwave in burned-in captions.

Usage: rebrand.py SOURCE OUT_DIR [--preview FRAME ...]
Writes OUT_DIR/rebrand_video.mp4 (video only, same timing as source).
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/System/Library/Fonts/SFNS.ttf'
NEW = 'Nookwave'

# Burned-in caption ranges [first, last] frame and replacement text.
CAPTIONS = [
    (3973, 4028, "It's called Nookwave."),
    (4675, 4746, 'Nookwave uses gentle traction'),
    (6087, 6135, 'are now using Nookwave'),
    (6654, 6706, 'Nookwave is now available'),
]
CAPTION_BAND = (655, 750)

# Logo fixes: (first, last, kind, anchor_frame, bbox x0,y0,x1,y1 at anchor)
LOGOS = [
    (3970, 4028, 'hero_tag', None, None),
    (4675, 4746, 'plate', 4690, (300, 470, 460, 530)),
    (6030, 6100, 'bag', None, (0, 420, 260, 660)),  # detected per frame, not tracked
    (6740, 6860, 'webline', 6800, (140, 526, 578, 553)),  # guarantee screenshot body copy
]
WEBLINE = dict(text='Simply, get yourself a Nookwave today, use it for', size=20, cx=359, baseline=545,
               ink=(66, 66, 66), bg=(253, 253, 253))
FUTURA = '/System/Library/Fonts/Supplemental/Futura.ttc'


def font(size):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name('Semibold')
    except Exception:
        pass
    return f


def caption_box(frame):
    y0, y1 = CAPTION_BAND
    band = frame[y0:y1]
    white = (band > 232).all(axis=2)
    rows = np.where(white.sum(axis=1) > 60)[0]
    if len(rows) < 8:
        return None
    r0, r1 = rows.min(), rows.max()
    cols = np.where(white[r0:r1 + 1].sum(axis=0) > (r1 - r0) * 0.5)[0]
    if len(cols) < 20:
        return None
    return cols.min(), r0 + y0, cols.max(), r1 + y0


def draw_caption(frame, text):
    box = caption_box(frame)
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    h = y1 - y0
    scale = h / 62.0  # full-size box is ~62px tall; smaller during pop-in
    f = font(max(8, round(37 * scale)))
    pad = round(15 * scale)
    tw = f.getlength(text)
    cx = (x0 + x1) / 2
    nx0, nx1 = int(min(x0, cx - tw / 2 - pad)), int(max(x1, cx + tw / 2 + pad))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((nx0, y0, nx1, y1), radius=round(10 * scale), fill=(255, 255, 255))
    d.text((cx, (y0 + y1) / 2), text, font=f, fill=(0, 0, 0), anchor='mm')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def inpaint_dark(frame, region, rel=0.72, grow=2, close=15):
    """Erase dark lettering on a lighter plate inside `region` (uint8 mask)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if region.sum() == 0:
        return frame
    med = np.median(gray[region > 0])
    plate = ((gray >= med * rel) & (region > 0)).astype(np.uint8) * 255
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    plate = cv2.erode(plate, np.ones((3, 3), np.uint8))
    text = ((gray < med * rel) & (plate > 0)).astype(np.uint8) * 255
    text = cv2.dilate(text, np.ones((3, 3), np.uint8), iterations=grow)
    return cv2.inpaint(frame, text, 5, cv2.INPAINT_TELEA)


def erase_text_on_white(frame, roi, white=205, dark=195, cover=0.42):
    """Inpaint dark lettering that sits inside a mostly-white surface (the collage bag)."""
    x0, y0, x1, y1 = roi
    sub = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    frac = cv2.blur((gray > white).astype(np.float32), (41, 41))
    text = ((gray < dark) & (frac > cover)).astype(np.uint8) * 255
    text = cv2.dilate(text, np.ones((3, 3), np.uint8), iterations=2)
    if text.any():
        frame[y0:y1, x0:x1] = cv2.inpaint(sub, text, 5, cv2.INPAINT_TELEA)
    return frame


def hero_tag_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (5, 70, 60), (22, 255, 230))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 300:
        return None
    blob = (lab == i).astype(np.uint8) * 255
    hull = cv2.convexHull(cv2.findNonZero(blob))
    return hull


def plate_hull(frame, bbox, pad=12):
    """Light label plate inside a tracked bbox (shot 71 device label)."""
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    gray = cv2.cvtColor(frame[y0:y1 + pad, x0:x1 + pad], cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cv2.convexHull(cv2.findNonZero((lab == i).astype(np.uint8))) + np.array([x0, y0])


def tag_texture(CW, CH, base, ink, word='nookwave'):
    """Flat stitched pill tag (RGBA) with `word` debossed in `ink`."""
    b = tuple(int(x) for x in base[::-1])
    k = tuple(int(x) for x in ink[::-1])
    tex = Image.new('RGBA', (CW, CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(tex)
    d.rounded_rectangle((0, 0, CW - 1, CH - 1), radius=CH // 2, fill=b + (255,))
    stitch = Image.new('L', (CW, CH), 0)
    sd = ImageDraw.Draw(stitch)
    inset = max(6, CH // 7)
    sd.rounded_rectangle((inset, inset, CW - 1 - inset, CH - 1 - inset), radius=CH // 2 - inset,
                         outline=150, width=max(2, CH // 28))
    dash = max(8, CH // 6)
    for x in range(0, CW, dash * 2):
        sd.rectangle((x + dash, 0, x + 2 * dash - 1, CH), fill=0)
    tex.paste(Image.new('RGBA', (CW, CH), k + (255,)), (0, 0), stitch)
    f = font(int(CH * 0.5))
    while f.getlength(word) > CW * 0.66 and f.size > 8:
        f = font(f.size - 2)
    d = ImageDraw.Draw(tex)
    d.text((CW / 2 + 1, CH / 2 + 2), word, font=f, fill=(255, 255, 255, 50), anchor='mm')
    d.text((CW / 2, CH / 2), word, font=f, fill=k + (235,), anchor='mm')
    return np.array(tex)


def composite(frame, rgba):
    a = cv2.GaussianBlur(rgba[..., 3].astype(np.float32) / 255, (3, 3), 0.8)[..., None]
    col = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    return (frame * (1 - a) + col * a).astype(np.uint8)


def render_tag(frame, hull, base, ink, grow=1.08):
    """Stitched tag warped onto the oriented rectangle of `hull` (rotating hero shot)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    if w < h:
        w, h, ang = h, w, ang + 90
    w, h = w * grow, h * grow
    if w < 12 or h < 5:
        return frame
    u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v
    c = np.array([cx, cy])
    dst = np.float32([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2,
                      c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])
    CW = 480
    CH = max(40, int(CW * h / w))
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), dst)
    H, W = frame.shape[:2]
    return composite(frame, cv2.warpPerspective(tag_texture(CW, CH, base, ink), M, (W, H),
                                                flags=cv2.INTER_AREA))


# Device label in shot 71 at the anchor frame (4690): arched pill, ends sag ~6px below the centre.
PLATE = dict(x0=326, top=475, w=102, h=25, bend=6.0)


def plate_geometry(scale, lx, ly):
    ax, ay = next(L[4] for L in LOGOS if L[2] == 'plate')[:2]
    return dict(x0=lx + (PLATE['x0'] - ax) * scale, top=ly + (PLATE['top'] - ay) * scale,
                w=int(round(PLATE['w'] * scale)), h=int(round(PLATE['h'] * scale)),
                bend=PLATE['bend'] * scale)


def render_bent_tag(frame, g, base, ink):
    S = 4
    CW, CH = g['w'] * S, g['h'] * S
    flat = tag_texture(CW, CH, base, ink)
    pad = int(g['bend'] * S) + 2
    flat = np.pad(flat, ((0, pad), (0, 0), (0, 0)))
    xs = np.arange(CW, dtype=np.float32)
    shift = g['bend'] * S * ((xs - CW / 2) / (CW / 2)) ** 2
    mapx = np.tile(xs, (CH + pad, 1))
    mapy = np.arange(CH + pad, dtype=np.float32)[:, None] - shift[None, :]
    bent = cv2.remap(flat, mapx, mapy, cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    bent = cv2.resize(bent, (g['w'], (CH + pad) // S), interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    canvas = np.zeros((H, W, 4), np.uint8)
    x0, y0 = int(round(g['x0'])), int(round(g['top']))
    bh, bw = bent.shape[:2]
    canvas[y0:y0 + bh, x0:x0 + bw] = bent[:max(0, min(bh, H - y0)), :max(0, min(bw, W - x0))]
    return composite(frame, canvas)


def redraw_webline(frame, scale, lx, ly):
    """Blank the matched body-copy line and re-set it with the new product name."""
    w = WEBLINE
    ax, ay, bx, by = next(L[4] for L in LOGOS if L[2] == 'webline')
    x0, y0 = int(lx), int(ly)
    x1, y1 = int(lx + (bx - ax) * scale), int(ly + (by - ay) * scale)
    f = ImageFont.truetype(FUTURA, max(6, round(w['size'] * scale)), index=0)
    tw = f.getlength(w['text'])
    cx = lx + (w['cx'] - ax) * scale
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.rectangle((min(x0, cx - tw / 2) - 2, y0, max(x1, cx + tw / 2) + 2, y1 - 1), fill=w['bg'])
    d.text((cx, ly + (w['baseline'] - ay) * scale), w['text'], font=f, fill=w['ink'], anchor='ms')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def tag_colors(frame, hull, ink_ratio):
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    px = frame[m > 0].astype(np.float32)
    if len(px) == 0:
        return None
    base = np.percentile(px, 70, axis=0)
    return base, base * ink_ratio


class Tracker:
    def __init__(self, frame, bbox):
        x0, y0, x1, y1 = bbox
        self.tpl = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        self.pos = (x0, y0)
        self.size = (x1 - x0, y1 - y0)

    def update(self, frame, search=40):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = self.pos
        w, h = self.size
        H, W = gray.shape
        sx0, sy0 = max(0, x - search), max(0, y - search)
        sx1, sy1 = min(W, x + w + search), min(H, y + h + search)
        roi = gray[sy0:sy1, sx0:sx1]
        if roi.shape[0] < h or roi.shape[1] < w:
            return None
        res = cv2.matchTemplate(roi, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < 0.35:
            return None
        self.pos = (sx0 + loc[0], sy0 + loc[1])
        return (*self.pos, self.pos[0] + w, self.pos[1] + h)


def plan_tracks(cap):
    """Precompute per-frame bboxes for tracked logos (forward and backward from anchor)."""
    tracks = {}
    for first, last, kind, anchor, bbox in LOGOS:
        if anchor is None:
            continue
        boxes = {anchor: bbox}
        if kind in ('plate', 'webline'):
            # Zoom-in at the cut, then slow drift: multi-scale match of the anchor template per frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
            ok, f = cap.read()
            x0, y0, x1, y1 = bbox
            tpl = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            boxes = {}
            for n in range(first, last + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                ok, f = cap.read()
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                best = (0, None)
                for sc in (np.arange(0.85, 2.05, 0.05) if kind == 'plate' else np.arange(0.40, 1.21, 0.02)):
                    t = cv2.resize(tpl, None, fx=sc, fy=sc)
                    if t.shape[0] >= g.shape[0] or t.shape[1] >= g.shape[1]:
                        continue
                    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED))
                    if score > best[0]:
                        best = (score, (float(sc), *loc))
                if best[0] > 0.6:
                    boxes[n] = best[1]
            tracks[kind] = boxes
            continue
        for direction in (1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
            ok, f = cap.read()
            tr = Tracker(f, bbox)
            n = anchor + direction
            while first <= n <= last:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                ok, f = cap.read()
                if not ok:
                    break
                b = tr.update(f)
                if b is None or b[0] <= 2 or b[2] >= f.shape[1] - 2:
                    break  # lost, or template pinned at the frame edge
                boxes[n] = b
                tr = Tracker(f, b)  # re-anchor to follow slow appearance change
                n += direction
        # Past the frame edge the template can't match; continue at the last measured velocity.
        # The collage jumps up once (~6050-6054) then only pans left, so trust tracking up to 6054.
        known = sorted(k for k in boxes if kind != 'bag' or k <= 6054)
        boxes = {k: boxes[k] for k in known}
        if len(known) >= 6 and known[-1] < last:
            k1, k0 = known[-1], known[-6]
            vx = (boxes[k1][0] - boxes[k0][0]) / (k1 - k0)
            vy = 0.0  # vertical tracking is unreliable once the template is clipped
            x0, y0, x1, y1 = boxes[k1]
            for k in range(k1 + 1, last + 1):
                d = k - k1
                boxes[k] = tuple(int(round(v)) for v in (x0 + vx * d, y0 + vy * d, x1 + vx * d, y1 + vy * d))
                if boxes[k][2] < 0 or boxes[k][3] < 0:
                    break
        tracks[kind] = boxes
    return tracks


def process(frame, n, tracks):
    for first, last, kind, _, _ in LOGOS:
        if not first <= n <= last:
            continue
        if kind == 'hero_tag':
            hull = hero_tag_mask(frame)
            if hull is not None:
                cols = tag_colors(frame, hull, 0.42)
                if cols:
                    frame = render_tag(frame, hull, *cols, grow=1.10)
        elif kind == 'webline':
            t = tracks.get(kind, {}).get(n)
            if t is not None:
                frame = redraw_webline(frame, *t)
        elif kind == 'bag':
            for _ in range(2):  # second pass catches faint antialiased edges
                frame = erase_text_on_white(frame, next(L[4] for L in LOGOS if L[2] == 'bag'))
        elif kind == 'plate':
            t = tracks.get(kind, {}).get(n)
            if t is not None:
                g = plate_geometry(*t)
                x0, y0 = int(g['x0']), int(g['top'])
                px = frame[y0 + 2:y0 + g['h'], x0 + 4:x0 + g['w'] - 4].reshape(-1, 3).astype(np.float32)
                base = np.percentile(px, 75, axis=0)
                frame = render_bent_tag(frame, g, base, base * 0.38)
        else:
            b = tracks.get(kind, {}).get(n)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            m = np.zeros(frame.shape[:2], np.uint8)
            m[max(0, y0 - 6):max(0, y1 + 6), max(0, x0 - 8):max(0, x1 + 8)] = 255
            frame = inpaint_dark(frame, m, rel=0.75 if kind == 'bag' else 0.70, grow=2)
    for first, last, text in CAPTIONS:
        if first <= n <= last:
            frame = draw_caption(frame, text)
    return frame


def main():
    src, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(src)
    tracks = plan_tracks(cap)
    if '--preview' in sys.argv:
        for n in map(int, sys.argv[sys.argv.index('--preview') + 1:]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, f = cap.read()
            cv2.imwrite(str(out_dir / f'preview_{n}.png'), np.hstack([f, process(f.copy(), n, tracks)]))
        return
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '14',
                           '-preset', 'slow', '-pix_fmt', 'yuv420p', str(out_dir / 'rebrand_video.mp4')],
                          stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ff.stdin.write(process(f, n, tracks).tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    print('frames', n)


if __name__ == '__main__':
    main()
