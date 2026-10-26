"""job-06 visual rebrand: FisioRest / Artuvate -> Nookwave.

Fixes
  1. Laptop product web page (frames 1760-1941): garbled "artuvate" wordmark -> "nookwave",
     "FisoRest" product title -> "Nookwave". The page is a flat screen under a slow push-in, so
     it is tracked with a planar homography (ECC, every frame registered directly to anchor
     frame 1941, initialised from the neighbouring frame). The text is edited once in anchor
     space and the edit is warped into every frame, blurred and colour-fitted to that frame.
  2. Burned-in caption (frames 1958-2042) "It's called FisioRest," -> "It's called Nookwave,".
     The caption is static (no pop/scale). Old line is erased with a flow-warped clean plate
     (caption-free frames 1942-1957 of the same shot), then the line is re-set centred in
     Times New Roman Bold with the fitted size, yellow fill, black outline and soft shadow.

Usage
  rebrand_job.py                      full render -> project/intermediate/rebrand_video.mp4 (video only)
                                      + renders/preview_visual_rebrand.mp4 (muxed with ORIGINAL audio)
  rebrand_job.py --preview N [N ...]  write before|after PNGs to analysis/work/preview/
  rebrand_job.py --qc                 before/after crops from the rendered file -> analysis/visual_qc/
Run with: ~/.cache/video-studio-venv/bin/python -I project/rebrand_job.py
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JOB = Path(__file__).resolve().parents[1]
SRC = JOB / 'assets' / 'source.mp4'
OUT_VIDEO = JOB / 'project' / 'intermediate' / 'rebrand_video.mp4'
OUT_PREVIEW = JOB / 'renders' / 'preview_visual_rebrand.mp4'
QC_DIR = JOB / 'analysis' / 'visual_qc'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
W, H = 720, 1280

# ---------------------------------------------------------------- web page shot
WEB_FIRST, WEB_LAST, ANCHOR = 1760, 1941, 1941
TRACK_ROI = (0, 480, 420, 660)          # anchor-frame region used for ECC registration
AVENIR_C = str(Path.home() / 'Library/Fonts/Avenir Next Condensed.ttc')
DIN_C = '/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf'
# Each web item (anchor-frame coords, measured on frame 1941):
#   erase box, new text, font (path, index), target x-height ('xh') or cap height ('cap'),
#   left x and baseline y at the left end, baseline angle (deg, CCW+ = rising to the right),
#   max_x = right limit (next nav item / title badge). Width is fitted by x-squeeze (>=0.86)
#   and then by size reduction. bold = extra stroke in 1/4-px units to match the old weight.
WEB_ITEMS = [
    # garbled "artuvate" wordmark: humanist medium sans, x 18-66, x-height ~8.5, nav starts at x=82
    dict(name='wordmark', erase=(15, 500, 71, 520), text='nookwave', font=(AVENIR_C, 0), bold=1,
         xh=8.3, x0=18.0, base=516.0, angle=2.7, max_x=73.0),
    # "FisoRest" title: bold condensed grotesque, x 281-328, cap ~11.5, badge outline at x=332
    dict(name='title', erase=(278, 548, 331, 569), text='Nookwave', font=(DIN_C, 0), bold=1,
         cap=11.6, x0=281.5, base=565.6, angle=2.7, max_x=329.0),
]
WEB_BLUR_GAIN = 0.4    # scale on the measured residual defocus (see Rebrander)
WEB_INK_PCT = 6        # ink colour = median of darkest N% pixels of the old word

# ---------------------------------------------------------------- caption
CAP_FIRST, CAP_LAST = 1958, 2042
PLATE_FRAMES = (1942, 1957)             # caption-free frames of the same shot
CAP_OLD = "It's called FisioRest,"
CAP_NEW = "It's called Nookwave,"
TNR = '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'
CAP = dict(size=46.0, sx=0.98, track=-0.5, base=914, cx=359.5,
           fill_erode=0.5, outline=3.2, shadow_sigma=1.6, shadow_alpha=0.55, shadow_dxy=(1, 1))
CAP_ZONE = (130, 860, 590, 948)         # x0, y0, x1, y1 working box for the caption


# ============================================================ helpers
SWS = 'accurate_rnd+full_chroma_int'
# Source is BT.709 limited range. Decode/encode with the same explicit matrix so untouched pixels
# round-trip without a colour shift (OpenCV's decoder + ffmpeg's default BT.601 encode shifted
# luma by ~-1.7 levels).
ENC_COLOUR = ['-vf', f'scale=out_color_matrix=bt709:out_range=tv:flags={SWS}', '-colorspace', 'bt709',
              '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']


def iter_frames(path=SRC):
    """Sequential exact-frame decode via ffmpeg -> BGR uint8 frames."""
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', str(path), '-map', '0:v:0', '-vf',
                          f'scale=in_color_matrix=bt709:in_range=tv:flags={SWS}', '-f', 'rawvideo',
                          '-pix_fmt', 'bgr24', '-fps_mode', 'passthrough', '-'], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    size = W * H * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def read_frames(first, last):
    """Returns {n: BGR frame} for first..last (sequential decode, exact indices)."""
    out = {}
    for n, f in enumerate(iter_frames()):
        if n >= first:
            out[n] = f
        if n >= last:
            break
    return out


# ============================================================ web page
def track_web(frames):
    """Homography anchor->frame for every frame of the web shot."""
    ga = cv2.cvtColor(frames[ANCHOR], cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, y0, x1, y1 = TRACK_ROI
    tpl = ga[y0:y1, x0:x1]
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    Hs, prev = {}, np.eye(3)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    for n in range(WEB_LAST, WEB_FIRST - 1, -1):
        gi = cv2.cvtColor(frames[n], cv2.COLOR_BGR2GRAY).astype(np.float32)
        Wl = (prev @ T).astype(np.float32)
        cc, Wl = cv2.findTransformECC(tpl, gi, Wl, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        Hn = Wl.astype(np.float64) @ np.linalg.inv(T)
        Hn /= Hn[2, 2]
        Hs[n] = (Hn, cc)
        prev = Hn
    return Hs


def fit_font(item):
    """Font size + x-squeeze so the new word matches the old letter height and fits before max_x."""
    path, idx = item['font']
    ref = ImageFont.truetype(path, 200, index=idx)
    key = 'xh' if 'xh' in item else 'cap'
    glyph = 'x' if key == 'xh' else 'H'
    bb = ref.getbbox(glyph)
    size = item[key] * 200 / (bb[3] - bb[1])
    width = ImageFont.truetype(path, 200, index=idx).getlength(item['text']) * size / 200
    avail = (item['max_x'] - item['x0']) / np.cos(np.radians(item['angle']))
    sx = min(1.0, avail / width)
    if sx < 0.86:
        size *= sx / 0.86
        sx = 0.86
    return size, sx


def text_layer(item, ss=4):
    """Alpha layer (anchor-frame size) of the new web text, x-squeezed, rotated, anti-aliased."""
    path, idx = item['font']
    size, sx = fit_font(item)
    f = ImageFont.truetype(path, round(size * ss), index=idx)
    im = Image.new('L', (W * ss, 140 * ss), 0)
    oy = int(item['base']) - 100  # canvas row 0 == frame row oy
    by = (item['base'] - oy) * ss
    ImageDraw.Draw(im).text((0, by), item['text'], font=f, fill=255, anchor='ls',
                            stroke_width=item.get('bold', 0), stroke_fill=255)  # faux-bold in 1/4 px
    a = np.array(im, np.float32) / 255
    a = cv2.resize(a, (int(round(W * ss * sx)), a.shape[0]), interpolation=cv2.INTER_AREA)
    pad = np.zeros((a.shape[0], W * ss), np.float32)
    x0 = int(round(item['x0'] * ss))
    w = min(W * ss - x0, a.shape[1])
    pad[:, x0:x0 + w] = a[:, :w]
    # rotate about the start of the baseline (text rises to the right on this screen)
    M = cv2.getRotationMatrix2D((x0, by), item['angle'], 1.0)
    pad = cv2.warpAffine(pad, M, (pad.shape[1], pad.shape[0]), flags=cv2.INTER_LINEAR)
    a = cv2.resize(pad, None, fx=1 / ss, fy=1 / ss, interpolation=cv2.INTER_AREA)
    full = np.zeros((H, W), np.float32)
    full[oy:oy + a.shape[0]] = a[:H - oy]
    return full


def build_web_edit(anchor):
    """Edited anchor image + soft replacement mask (both in anchor coordinates)."""
    E = anchor.copy()
    mask = np.zeros((H, W), np.uint8)
    for it in WEB_ITEMS:
        x0, y0, x1, y1 = it['erase']
        g = cv2.cvtColor(anchor[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        bg = np.percentile(g, 90)
        ink = (g < bg - 40).astype(np.uint8) * 255
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = cv2.dilate(ink, np.ones((5, 5), np.uint8))
        E = cv2.inpaint(E, m, 4, cv2.INPAINT_TELEA)
        # flatten the erased box to the local page white (removes inpaint smudges)
        sub = E[y0:y1, x0:x1].astype(np.float32)
        page = cv2.GaussianBlur(sub, (0, 0), 6)
        sel = m[y0:y1, x0:x1] > 0
        sub[sel] = page[sel]
        E[y0:y1, x0:x1] = np.clip(sub, 0, 255).astype(np.uint8)
        # ink colour from the darkest old pixels
        dark = anchor[y0:y1, x0:x1][g < np.percentile(g, WEB_INK_PCT)]
        ink_col = np.median(dark, axis=0).astype(np.float32)
        a = text_layer(it)
        a = cv2.GaussianBlur(a, (0, 0), 0.45)  # anchor frame is itself slightly soft
        E = (E * (1 - a[..., None]) + ink_col[None, None, :] * a[..., None]).astype(np.uint8)
        mm = np.zeros((H, W), np.uint8)
        mm[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 255
        mm |= cv2.dilate((a > 0.02).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        mask = np.maximum(mask, mm)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.2)
    return E, soft


def prefilter(img, Hn):
    """Anti-alias the anchor image before warping it down to this frame's (smaller) scale."""
    det = abs(np.linalg.det(Hn[:2, :2])) ** 0.5  # ~linear scale anchor->frame (<=1)
    pre = max(0.0, 0.5 * (1 / max(det, 1e-3) - 1))
    return cv2.GaussianBlur(img, (0, 0), pre) if pre > 0.05 else img


def web_blur_sigma(frame, Hn, A):
    """Residual defocus of this frame vs the warped anchor, measured over the whole page ROI."""
    ga = cv2.cvtColor(prefilter(A, Hn), cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = TRACK_ROI
    roi[y0 + 5:y1 - 10, x0:x1] = 1
    Aw = cv2.warpPerspective(ga, Hn, (W, H))
    M = cv2.erode(cv2.warpPerspective(roi, Hn, (W, H), flags=cv2.INTER_NEAREST), np.ones((7, 7))) > 0
    errs = []
    for s in np.arange(0, 2.01, 0.1):
        ab = cv2.GaussianBlur(Aw, (0, 0), s) if s > 0 else Aw
        X, Y = ab[M], F[M]
        g, o = np.polyfit(X, Y, 1)
        errs.append(((g * X + o - Y) ** 2).mean())
    return 0.1 * int(np.argmin(errs))


def _web_prep(frame, Hn, sigma, A, E, soft):
    Aw = cv2.warpPerspective(prefilter(A, Hn), Hn, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    Ew = cv2.warpPerspective(prefilter(E, Hn), Hn, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    if sigma > 0:
        Aw = cv2.GaussianBlur(Aw, (0, 0), sigma)
        Ew = cv2.GaussianBlur(Ew, (0, 0), sigma)
    Mw = cv2.warpPerspective(soft, Hn, (W, H), flags=cv2.INTER_LINEAR)
    n, _, st, _ = cv2.connectedComponentsWithStats((Mw > 0.01).astype(np.uint8))
    boxes = []
    for i in sorted(range(1, n), key=lambda i: st[i, 0]):  # left to right: wordmark, title
        x, y, w, h = st[i, :4]
        p = 10
        boxes.append((max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)))
    return Aw, Ew, Mw, boxes


def web_levels(frame, Hn, sigma, A, E, soft):
    """Per item, per channel (gain, frame white, anchor white) mapping anchor levels to this frame.

    Two robust levels: page white (ring around the item) and ink (darkest old-glyph pixels,
    present in both images at the same place and blur). No regression: the AI glyphs morph
    frame to frame, which would flatten the slope and grey the patch. The per-frame values are
    noisy (glyph morphing), so the caller smooths them over time.
    """
    Aw, _, Mw, boxes = _web_prep(frame, Hn, sigma, A, E, soft)
    F = frame.astype(np.float32)
    lv = []
    for x0, y0, x1, y1 in boxes:
        f, a, m = F[y0:y1, x0:x1], Aw[y0:y1, x0:x1], Mw[y0:y1, x0:x1]
        ring = (cv2.dilate((m > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (m <= 0.01)
        core = m > 0.99
        L = []
        for c in range(3):
            fw, aw = np.percentile(f[..., c][ring], 60), np.percentile(a[..., c][ring], 60)
            fd, ad = np.percentile(f[..., c][core], 2), np.percentile(a[..., c][core], 2)
            L.append(((fw - fd) / max(aw - ad, 1.0), fw, aw))
        lv.append(L)
    return np.array(lv, np.float64)  # (items, 3, 3)


def apply_web(frame, Hn, sigma, A, E, soft, levels):
    """Warp the anchor edit into this frame (blur-matched), map colour levels, composite."""
    _, Ew, Mw, boxes = _web_prep(frame, Hn, sigma, A, E, soft)
    out = frame.astype(np.float32)
    for (x0, y0, x1, y1), L in zip(boxes, levels):
        f, e, m = out[y0:y1, x0:x1], Ew[y0:y1, x0:x1], Mw[y0:y1, x0:x1]
        res = np.empty_like(e)
        for c in range(3):
            g, fw, aw = L[c]
            res[..., c] = (e[..., c] - aw) * g + fw
        out[y0:y1, x0:x1] = f * (1 - m[..., None]) + res * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================ caption
def caption_sdf(text, ss=4):
    """Signed distance (px at 1x, +inside) of the caption glyphs on a full-frame 4x canvas."""
    c = CAP
    f = ImageFont.truetype(TNR, round(c['size'] * ss))
    adv = [f.getlength(text[:i + 1]) - f.getlength(text[:i]) + c['track'] * ss for i in range(len(text))]
    width = sum(adv[:-1]) + f.getbbox(text[-1])[2]
    x0 = 0
    zx0, zy0, zx1, zy1 = CAP_ZONE
    cw, ch = (zx1 - zx0) * ss, (zy1 - zy0) * ss
    im = Image.new('L', (int(width / c['sx']) + 200 * ss, ch), 0)
    d = ImageDraw.Draw(im)
    x = 50 * ss
    for chh, a in zip(text, adv):
        d.text((x, (c['base'] - zy0) * ss), chh, font=f, fill=255, anchor='ls')
        x += a
    g = (np.array(im) > 127).astype(np.uint8)
    g = cv2.resize(g, (int(g.shape[1] * c['sx']), g.shape[0]), interpolation=cv2.INTER_NEAREST)
    cols = np.nonzero(g.any(0))[0]
    gx0, gx1 = cols.min(), cols.max() + 1
    canvas = np.zeros((ch, cw), np.uint8)
    left = int(round(c['cx'] * ss - zx0 * ss - (gx1 - gx0) / 2))
    canvas[:, left:left + (gx1 - gx0)] = g[:, gx0:gx1]
    din = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 5)
    return (din - dout) / ss


def caption_layers(text, ss=4):
    sd = caption_sdf(text, ss)
    c = CAP

    def down(m):
        return cv2.resize(m.astype(np.float32), None, fx=1 / ss, fy=1 / ss, interpolation=cv2.INTER_AREA)

    fill = down(sd > c['fill_erode'])
    outer = down(sd > -c['outline'])
    sh = cv2.GaussianBlur(outer, (0, 0), c['shadow_sigma'])
    dx, dy = c['shadow_dxy']
    sh = np.roll(np.roll(sh, dy, 0), dx, 1) * c['shadow_alpha']
    return fill, outer, sh


def caption_colours(frame):
    zx0, zy0, zx1, zy1 = CAP_ZONE
    z = frame[zy0:zy1, zx0:zx1]
    hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (15, 90, 120), (40, 255, 255)) > 0
    core = cv2.erode(yel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.median(z[core], axis=0).astype(np.float32)


def draw_caption(plate_zone, text, yellow):
    fill, outer, sh = caption_layers(text)
    z = plate_zone.astype(np.float32)
    z = z * (1 - sh[..., None])                     # soft shadow
    z = z * (1 - outer[..., None]) + 8.0 * outer[..., None]  # black outline
    z = z * (1 - fill[..., None]) + yellow[None, None, :] * fill[..., None]
    return np.clip(z, 0, 255).astype(np.uint8)


def old_caption_mask(frame):
    """Erase mask for the old line: fitted glyph outline+shadow, unioned with measured yellow/dark ink."""
    _, outer, sh = caption_layers(CAP_OLD)
    zx0, zy0, zx1, zy1 = CAP_ZONE
    hsv = cv2.cvtColor(frame[zy0:zy1, zx0:zx1], cv2.COLOR_BGR2HSV)
    yel = (cv2.inRange(hsv, (15, 90, 120), (40, 255, 255)) > 0).astype(np.uint8)
    m = ((outer > 0.02) | (sh > 0.05) | (cv2.dilate(yel, np.ones((11, 11), np.uint8)) > 0)).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def fill_flow(flow, hole):
    """Fill flow vectors inside `hole` by normalised convolution from the surroundings."""
    w = (1 - hole).astype(np.float32)
    out = flow.copy()
    for s in (4, 8, 16, 32):
        num = cv2.GaussianBlur(flow * w[..., None], (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)[..., None]
        est = num / np.maximum(den, 1e-4)
        sel = (hole > 0) & (den[..., 0] > 0.05)
        out[sel] = est[sel]
        if (den[..., 0][hole > 0] > 0.05).all():
            break
    return out


class CaptionPlate:
    def __init__(self, ref_frame):
        zx0, zy0, zx1, zy1 = CAP_ZONE
        self.pad = 40
        self.box = (zx0 - self.pad, zy0 - self.pad, zx1 + self.pad, zy1 + self.pad)
        x0, y0, x1, y1 = self.box
        self.ref = ref_frame[y0:y1, x0:x1].copy()
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def plate(self, frame, hole_zone):
        """Clean background for the caption zone of `frame`, warped from the caption-free ref."""
        x0, y0, x1, y1 = self.box
        cur = frame[y0:y1, x0:x1]
        hole = np.zeros(cur.shape[:2], np.uint8)
        p = self.pad
        hole[p:p + hole_zone.shape[0], p:p + hole_zone.shape[1]] = hole_zone
        hole_big = cv2.dilate(hole, np.ones((9, 9), np.uint8))
        gc = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
        # hide the caption in the current frame so it does not drag the flow
        gc_f = cv2.inpaint(gc, hole_big, 5, cv2.INPAINT_TELEA)
        flow = self.dis.calc(gc_f, gr, None)  # cur -> ref
        flow = fill_flow(flow, hole_big)
        gx, gy = np.meshgrid(np.arange(cur.shape[1], dtype=np.float32), np.arange(cur.shape[0], dtype=np.float32))
        warped = cv2.remap(self.ref, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        # match local brightness of the warped plate to the current frame around the hole
        ring = (cv2.dilate(hole_big, np.ones((15, 15), np.uint8)) > 0) & (hole_big == 0)
        wf, cf = warped.astype(np.float32), cur.astype(np.float32)
        gain = (cf[ring].mean(0) + 1) / (wf[ring].mean(0) + 1)
        warped = np.clip(wf * gain, 0, 255)
        return warped[p:p + hole_zone.shape[0], p:p + hole_zone.shape[1]]


def apply_caption(frame, plater, yellow):
    zx0, zy0, zx1, zy1 = CAP_ZONE
    hole = old_caption_mask(frame)
    plate = plater.plate(frame, hole)
    z = frame[zy0:zy1, zx0:zx1].astype(np.float32)
    a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.5)[..., None]
    a = np.maximum(a, hole[..., None].astype(np.float32))
    clean = z * (1 - a) + plate * a
    out = frame.copy()
    out[zy0:zy1, zx0:zx1] = draw_caption(clean, CAP_NEW, yellow)
    return out


# ============================================================ driver
class Rebrander:
    def __init__(self):
        pre = read_frames(WEB_FIRST, PLATE_FRAMES[1])
        self.Hs = track_web(pre)
        sig = {n: web_blur_sigma(pre[n], self.Hs[n][0], pre[ANCHOR]) for n in self.Hs}
        ks = sorted(sig)  # light temporal smoothing (3-tap) to avoid any shimmer
        # The page-wide estimate over-reads (the AI page text morphs between frames, which looks
        # like blur to the fit); 0.4x matched the old glyph edges best in side-by-side tests.
        self.sigma = {k: WEB_BLUR_GAIN * float(np.mean([sig[j] for j in ks if abs(j - k) <= 1])) for k in ks}
        self.A = pre[ANCHOR]
        self.E, self.soft = build_web_edit(self.A)
        raw = {n: web_levels(pre[n], self.Hs[n][0], self.sigma[n], self.A, self.E, self.soft) for n in ks}
        # temporal smoothing of the colour levels: median over +-6 frames, then mean over +-3
        med = {k: np.median([raw[j] for j in ks if abs(j - k) <= 6], axis=0) for k in ks}
        self.levels = {k: np.mean([med[j] for j in ks if abs(j - k) <= 3], axis=0) for k in ks}
        self.plater = CaptionPlate(pre[PLATE_FRAMES[1]])
        self.yellow = None
        log = {str(k): dict(H=v[0].tolist(), ecc=float(v[1]), blur_sigma=self.sigma[k]) for k, v in sorted(self.Hs.items())}
        (JOB / 'analysis' / 'work').mkdir(parents=True, exist_ok=True)
        json.dump(log, open(JOB / 'analysis' / 'work' / 'web_homographies.json', 'w'))

    def process(self, frame, n):
        if WEB_FIRST <= n <= WEB_LAST:
            frame = apply_web(frame, self.Hs[n][0], self.sigma[n], self.A, self.E, self.soft, self.levels[n])
        if CAP_FIRST <= n <= CAP_LAST:
            if self.yellow is None:
                self.yellow = caption_colours(frame)
            frame = apply_caption(frame, self.plater, self.yellow)
        return frame


def frame_count(path):
    r = subprocess.run(['/opt/homebrew/bin/ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                        '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True)
    return int(r.stdout.strip())


def render():
    rb = Rebrander()
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                           '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                           *ENC_COLOUR, '-pix_fmt', 'yuv420p', '-r', '30', str(OUT_VIDEO)], stdin=subprocess.PIPE)
    n = 0
    for n, f in enumerate(iter_frames()):
        ff.stdin.write(rb.process(f, n).tobytes())
    n += 1
    ff.stdin.close()
    ff.wait()
    print('frames written', n)
    OUT_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(OUT_VIDEO), '-i', str(SRC), '-map', '0:v:0',
                    '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy', str(OUT_PREVIEW)], check=True)
    print('source frames', frame_count(SRC), 'output frames', frame_count(OUT_VIDEO))


def preview(frames):
    rb = Rebrander()
    d = JOB / 'analysis' / 'work' / 'preview'
    d.mkdir(parents=True, exist_ok=True)
    src = read_frames(min(frames), max(frames))
    for n in sorted(set(frames)):
        rb_out = rb.process(src[n].copy(), n)
        cv2.imwrite(str(d / f'p{n}.png'), np.hstack([src[n], rb_out]))


QC_CROPS = {  # frame: crop box (x0, y0, x1, y1), zoom
    'web': ([1760, 1790, 1820, 1850, 1880, 1910, 1941], None),
    'cap': ([1958, 1980, 2000, 2020, 2042], (120, 850, 600, 950)),
}


def web_crop_box(Hn):
    pts = np.array([[5, 490, 1], [380, 490, 1], [5, 585, 1], [380, 585, 1]], np.float64).T
    q = Hn @ pts
    q = q[:2] / q[2]
    x0, y0 = q.min(1) - 6
    x1, y1 = q.max(1) + 6
    return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))


def qc():
    """Before/after crops from the RENDERED video (sequential decode of both files)."""
    QC_DIR.mkdir(parents=True, exist_ok=True)
    Hs = {int(k): np.array(v['H']) for k, v in
          json.load(open(JOB / 'analysis' / 'work' / 'web_homographies.json')).items()}
    want = set(QC_CROPS['web'][0]) | set(QC_CROPS['cap'][0])
    last = max(want)
    for n, (fa, fb) in enumerate(zip(iter_frames(SRC), iter_frames(OUT_VIDEO))):
        if n > last:
            break
        if n not in want:
            continue
        if n in QC_CROPS['web'][0]:
            x0, y0, x1, y1 = web_crop_box(Hs[n])
            tag = 'item1_web'
        else:
            x0, y0, x1, y1 = QC_CROPS['cap'][1]
            tag = 'item2_caption'
        ca, cb = fa[y0:y1, x0:x1], fb[y0:y1, x0:x1]
        z = max(1, int(round(760 / (x1 - x0))))
        ca = cv2.resize(ca, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        cb = cv2.resize(cb, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        sep = np.full((6, ca.shape[1], 3), 255, np.uint8)
        img = np.vstack([ca, sep, cb])
        cv2.putText(img, f'f{n} BEFORE', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, 'AFTER', (6, ca.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
        cv2.imwrite(str(QC_DIR / f'{tag}_f{n}.png'), img)
        if n in (1941, 2000):
            cv2.imwrite(str(QC_DIR / f'fullframe_f{n}_before_after.png'),
                        cv2.resize(np.hstack([fa, fb]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


if __name__ == '__main__':
    if '--preview' in sys.argv:
        preview([int(x) for x in sys.argv[sys.argv.index('--preview') + 1:]])
    elif '--qc' in sys.argv:
        qc()
    else:
        render()
