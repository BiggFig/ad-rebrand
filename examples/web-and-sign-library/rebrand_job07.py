"""job-07 visual rebrand: FisioRest / Artuvate -> Nookwave (+ remove COSSEAL emboss).

Usage:
  rebrand_job.py                      full render -> project/intermediate/rebrand_video.mp4 + renders/preview
  rebrand_job.py --only VH,CAP ...    process only some items (dev)
  rebrand_job.py --qc                 also write analysis/visual_qc before/after crops

Pipeline: pass 1 decodes the source sequentially (ffmpeg rawvideo pipe -> exact frame indices) and keeps
only the frames inside item ranges; each item fixer gets the whole range at once (so it can track/smooth
over time) and returns edited frames; pass 2 decodes again and streams every frame to libx264, swapping in
the edited ones.
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
W, H, NFRAMES = 720, 1280, 1923
SFNS = '/System/Library/Fonts/SFNS.ttf'
FFMPEG = '/opt/homebrew/bin/ffmpeg'


# ----------------------------------------------------------------------------------------------- io
def decode(keep=None):
    """Yield (n, frame) sequentially; frames outside `keep` (set) are skipped cheaply if keep given."""
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-i', str(SRC), '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-fps_mode', 'passthrough', '-'], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        if keep is None or n in keep:
            yield n, np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        n += 1
    p.wait()


def load_ranges(ranges):
    keep = set()
    for a, b in ranges:
        keep.update(range(a, b + 1))
    return dict(decode(keep))


def sfont(size, wdth=105, wght=600):
    f = ImageFont.truetype(SFNS, max(4, int(round(size))))
    f.set_variation_by_axes([wdth, 28, 400, wght])
    return f


def ss_line_mask(shape, segs, width, S=4):
    """Antialiased thick line segments (supersampled), returns float alpha of `shape`."""
    h, w = shape
    m = np.zeros((h * S, w * S), np.uint8)
    for (x0, y0), (x1, y1) in segs:
        d = np.array([x1 - x0, y1 - y0], float)
        d /= np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]]) * width / 2
        p0, p1 = np.array([x0, y0], float), np.array([x1, y1], float)
        poly = np.array([p0 + nrm, p1 + nrm, p1 - nrm, p0 - nrm]) * S
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


# ------------------------------------------------------------------------------- wordmark (V-H, V-E1)
ENDCARD_REF = 1880
LOGO_BOX = (150, 260, 570, 360)          # x0,y0,x1,y1 around the end-card logo (static from f1870)


def build_wordmark(ref_gray):
    """Re-letter 'nookwave' from the end-card artuvate glyphs (Century-Gothic-like monoline).

    n = u rotated 180, o = left half of the 'a' bowl mirrored, w = two v's sharing the middle apex,
    k = t stem (crossbar removed) + synthesised arm/leg at the measured stroke width, a/v/e reused as-is.
    Returns (alpha float32, baseline_y_in_alpha, x-height top) at the end-card scale.
    """
    x0, y0, x1, y1 = LOGO_BOX
    A = (255 - ref_gray[y0:y1, x0:x1].astype(np.float32)) / 255
    n, lab, st, _ = cv2.connectedComponentsWithStats((A > 0.5).astype(np.uint8))
    comps = sorted(range(1, n), key=lambda i: st[i, 0])
    glyph = {}
    for name, i in zip('artuvaTe', comps):   # T = second t
        gx, gy, gw, gh = st[i, :4]
        own = cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8))
        crop = (A * own)[:, max(0, gx - 2):gx + gw + 2]
        glyph[name] = crop                  # full-height strip (keeps baseline registration)
    base = 345.5 - y0                         # baseline (row) inside the strip
    sw = 10.3                                 # stroke width (px)
    xh_top = 290.5 - y0

    def trim(g):
        cols = np.where(g.max(0) > 0.02)[0]
        return g[:, cols.min():cols.max() + 1]

    a, v, e, u, t = (trim(glyph[c]) for c in 'aveut')
    # n: rotate u by 180 about its x-height band centre
    hh = A.shape[0]
    nn = np.zeros_like(u)
    band = slice(int(xh_top) - 2, int(np.ceil(base)) + 2)
    nn[band] = u[band][::-1, ::-1]
    # o: mirror the left half of the a bowl (bowl diameter == glyph height)
    rows = np.where(a.max(1) > 0.02)[0]
    diam = rows.max() - rows.min() + 1
    half = a[:, :diam // 2 + 1]
    o = np.concatenate([half, half[:, ::-1][:, 1:]], axis=1) if diam % 2 else np.concatenate([half[:, :-1], half[:, ::-1]], axis=1)
    o = trim(o)
    # w: two v's, second starts where the first's right stroke begins at the top
    top_row = int(xh_top) + 3
    vr = np.where(v[top_row] > 0.5)[0]
    gaps = np.where(np.diff(vr) > 1)[0]
    right_stroke_start = vr[gaps[0] + 1] if len(gaps) else vr.max() - 10
    off = right_stroke_start
    w_ = np.zeros((hh, off + v.shape[1]), np.float32)
    w_[:, :v.shape[1]] = v
    w_[:, off:] = np.maximum(w_[:, off:], v)
    # k: t stem (crossbar removed) + arm and leg
    stem_cols = np.where(t[int(base) - 10] > 0.5)[0]
    sc0, sc1 = stem_cols.min() - 1, stem_cols.max() + 2
    stem_profile = t[int(base) - 10, sc0:sc1]
    kw = int(round(0.74 * (base - xh_top))) + (sc1 - sc0)
    k = np.zeros((hh, kw), np.float32)
    trow = np.where(t.max(1) > 0.5)[0].min()
    k[trow:int(base) + 1, :sc1 - sc0] = stem_profile[None, :]
    k[trow - 1, :sc1 - sc0] = t[trow - 1, sc0:sc1]                       # antialiased top edge
    k[int(base) + 1, :sc1 - sc0] = t[int(base) + 1, sc0:sc1] if int(base) + 1 < hh else 0
    sx = (sc1 - sc0) - 1.5
    jy = base - 0.40 * (base - xh_top)
    arm_end = (kw - 3.5, xh_top - 8)
    arm = ss_line_mask(k.shape, [((sx - 2, jy + 4), arm_end)], sw * 1.02)
    pj = (sx + 0.30 * (arm_end[0] - sx), jy + 0.30 * (arm_end[1] - jy))
    leg = ss_line_mask(k.shape, [(pj, (kw - 2.0, base + 8))], sw * 1.02)
    yy = np.arange(hh)[:, None]
    clip = np.clip(np.minimum(yy - xh_top + 0.5, base + 0.5 - yy), 0, 1).astype(np.float32)
    k = np.maximum(k, np.maximum(arm, leg) * clip)
    k = trim(k)

    seq = [(nn, 9), (o, 6), (o, 9), (k, 1), (w_, 2), (a, 6), (v, 2), (e, 0)]
    total = sum(g.shape[1] + gap for g, gap in seq)
    out = np.zeros((hh, total), np.float32)
    x = 0
    for g, gap in seq:
        out[:, x:x + g.shape[1]] = np.maximum(out[:, x:x + g.shape[1]], g)
        x += g.shape[1] + gap
    return np.clip(out, 0, 1), base, xh_top


def fix_endcard(frames):
    """V-H: logo slides in from the left while fading grey->black (f1856-1869), static to the end."""
    ref = frames[ENDCARD_REF]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = LOGO_BOX
    A = (255 - ref_gray[y0:y1, x0:x1].astype(np.float32)) / 255
    wm, base, _ = build_wordmark(ref_gray)
    # centre the new wordmark on the old logo's centre, same baseline
    cols = np.where(A.max(0) > 0.05)[0]
    old_cx = x0 + (cols.min() + cols.max()) / 2
    out = {}
    for n, f in frames.items():
        if n < 1856:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        D = (255 - g) / 255
        best = None
        for dx in range(-60, 3):
            sl = D[y0:y1, x0 + dx:x1 + dx]
            denom = (A * A).sum()
            c = (sl * A).sum() / denom      # darkness scale = 1 - colour/255
            err = ((sl - c * A) ** 2)[:, 40:-40].sum()
            if best is None or err < best[0]:
                best = (err, dx, c)
        _, dx, c = best
        if c < 0.01:
            continue
        # erase old logo: inpaint the (shifted) glyph footprint
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0 + dx:x1 + dx] = (A > 0.01).astype(np.uint8) * 255
        m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
        f2 = cv2.inpaint(f, m, 3, cv2.INPAINT_TELEA)
        # composite the new wordmark in the same colour / position
        ww = wm.shape[1]
        nx = int(round(old_cx + dx - ww / 2))
        alpha = np.zeros((H, W), np.float32)
        alpha[y0:y1, nx:nx + ww] = wm
        colour = 255 * (1 - c)            # grey level of the fading logo (black when c->1)
        if c > 1:
            colour = 0
        f2 = f2.astype(np.float32)
        a3 = alpha[..., None]
        out[n] = np.clip(f2 * (1 - a3) + colour * a3, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------------ V-E website
FUTURA = '/System/Library/Fonts/Supplemental/Futura.ttc'
WEB_REF = 1500


def ecc_track(frames, ref_n, mask=None, motion=cv2.MOTION_AFFINE, order=None):
    """Per-frame 2x3 warp mapping ref coords -> frame coords (chained init for robustness)."""
    ref = cv2.cvtColor(frames[ref_n], cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = {ref_n: np.eye(2, 3, dtype=np.float32)}
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    for seq in (sorted(n for n in frames if n > ref_n), sorted((n for n in frames if n < ref_n), reverse=True)):
        Wm = np.eye(2, 3, dtype=np.float32)
        for n in seq:
            g = cv2.cvtColor(frames[n], cv2.COLOR_BGR2GRAY).astype(np.float32)
            _, Wm = cv2.findTransformECC(ref, g, Wm, motion, crit, mask, 5)
            out[n] = Wm.copy()
    return out


def fit_font_to_box(fontfile, index, old_text, box_w, box_h):
    """Font size + horizontal scale so that `old_text` ink bbox == (box_w, box_h)."""
    size = 40
    for _ in range(4):
        f = ImageFont.truetype(fontfile, size * 4, index=index)
        b = f.getbbox(old_text)
        size = size * box_h / ((b[3] - b[1]) / 4)
    f = ImageFont.truetype(fontfile, int(round(size * 4)), index=index)
    b = f.getbbox(old_text)
    xscale = box_w / ((b[2] - b[0]) / 4)
    return size, xscale


def put_text(img, text, fontfile, index, size, xscale, x_left, y_top_ref_text, ref_text, colour, cx=None, gamma=1.0):
    """Draw `text` on float BGR img so its ink aligns with where `ref_text` would sit (same baseline)."""
    S = 4
    f = ImageFont.truetype(fontfile, int(round(size * S)), index=index)
    rb = f.getbbox(ref_text)
    tb = f.getbbox(text)
    pad = 20 * S
    wpx = tb[2] - tb[0] + 2 * pad
    im = Image.new('L', (wpx, max(rb[3], tb[3]) - min(rb[1], tb[1]) + 2 * pad), 0)
    top = min(rb[1], tb[1])
    ImageDraw.Draw(im).text((pad - tb[0], pad - top), text, font=f, fill=255)
    a = np.array(im, np.float32) / 255
    nw = int(round(a.shape[1] * xscale / S))
    a = cv2.resize(a, (nw, int(round(a.shape[0] / S))), interpolation=cv2.INTER_AREA) ** gamma
    # ink of ref_text starts at (pad - top + rb[1])/S rows into the canvas
    oy = int(round(y_top_ref_text - (pad - top + rb[1]) / S))
    ox = int(round(x_left - pad * xscale / S)) if cx is None else int(round(cx - a.shape[1] / 2))
    h, w = a.shape
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(img.shape[0], oy + h), min(img.shape[1], ox + w)
    aa = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox][..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - aa) + np.array(colour, np.float32) * aa
    return img


def ink_bbox(gray, box, thr=128):
    x0, y0, x1, y1 = box
    ys, xs = np.where(gray[y0:y1, x0:x1] < thr)
    return xs.min() + x0, ys.min() + y0, xs.max() + x0, ys.max() + y0


def fix_website(frames):
    """V-E: header logo, product title and the tiny 'Without/With FisioRest' labels, edited once in the
    f1500 reference and carried to every frame of the zoom/drift with an ECC affine track (edited reference composited inside feathered masks)."""
    ref = frames[WEB_REF]
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    E = ref.astype(np.float32).copy()
    M = np.zeros((H, W), np.float32)              # where the edited reference replaces the frame
    # 1) header wordmark
    hx0, hy0, hx1, hy1 = ink_bbox(gray, (255, 5, 465, 62))
    E[hy0 - 3:hy1 + 4, hx0 - 4:hx1 + 5] = 255
    M[hy0 - 6:hy1 + 7, hx0 - 30:hx1 + 31] = 1
    wm, _, _ = WORDMARK
    rows = np.where(wm.max(1) > 0.3)[0]
    cols = np.where(wm.max(0) > 0.3)[0]
    old_end = ink_bbox(WORDMARK_REF_GRAY, LOGO_BOX)
    s = ((hx1 - hx0) / (old_end[2] - old_end[0]) + (hy1 - hy0) / (old_end[3] - old_end[1])) / 2
    wm_c = np.pad(wm, 4)[rows.min() + 2:rows.max() + 7, cols.min() + 2:cols.max() + 7]
    small = cv2.resize(wm_c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, swd = small.shape
    cx, top = (hx0 + hx1) / 2, hy0 - 2 * s
    ox, oy = int(round(cx - swd / 2)), int(round(top))
    ink = np.median(ref[hy0:hy1, hx0:hx1][gray[hy0:hy1, hx0:hx1] < 40], axis=0)
    a = small[..., None]
    E[oy:oy + sh, ox:ox + swd] = E[oy:oy + sh, ox:ox + swd] * (1 - a) + ink * a
    # 2) product title (Futura Bold, condensed to the old ink box: matches the site's Futura Demi)
    tx0, ty0, tx1, ty1 = ink_bbox(gray, (15, 1040, 230, 1095))
    size, xs = fit_font_to_box(FUTURA, 2, 'FisioRest', tx1 - tx0 + 1, ty1 - ty0 + 1)
    tink = np.median(ref[ty0:ty1, tx0:tx1][gray[ty0:ty1, tx0:tx1] < 60], axis=0)
    E[ty0 - 3:ty1 + 4, tx0 - 3:tx1 + 4] = 255
    M[ty0 - 6:ty1 + 7, tx0 - 6:tx1 + 70] = 1
    put_text(E, 'Nookwave', FUTURA, 2, size, xs, tx0, ty0, 'FisioRest', tink)
    # 3) tiny gallery labels under the 5th/6th thumbnails
    for old, new, box in (('Without FisioRest', 'Without Nookwave', (496, 902, 542, 911)),
                          ('With FisioRest', 'With Nookwave', (554, 902, 594, 911))):
        lx0, ly0, lx1, ly1 = ink_bbox(gray, box, thr=200)
        bg = np.percentile(ref[ly0 - 3:ly1 + 4, lx0 - 3:lx1 + 4].reshape(-1, 3), 90, axis=0)
        M[ly0 - 3:ly1 + 4, lx0 - 4:lx1 + 5] = 1
        lab_ink = ref[ly0:ly1 + 1, lx0:lx1 + 1].reshape(-1, 3)[gray[ly0:ly1 + 1, lx0:lx1 + 1].ravel().argsort()[:6]].mean(0)
        E[ly0 - 2:ly1 + 3, lx0 - 2:lx1 + 3] = bg
        size_l, xs_l = fit_font_to_box(FUTURA, 2, old, lx1 - lx0 + 1, ly1 - ly0 + 1)
        put_text(E, new, FUTURA, 2, size_l, xs_l, 0, ly0, old, lab_ink, cx=(lx0 + lx1) / 2, gamma=0.55)
    M = cv2.GaussianBlur(M, (0, 0), 1.2)
    mask = np.ones((H, W), np.uint8)
    mask[895:980] = 0                       # burned-in caption is not part of the screenshot
    warps = ecc_track(frames, WEB_REF, mask)
    out = {}
    for n, f in frames.items():
        e = cv2.warpAffine(E, warps[n], (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(M, warps[n], (W, H), flags=cv2.INTER_LINEAR)[..., None]
        out[n] = np.clip(f.astype(np.float32) * (1 - m) + e * m, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------------- V-F review
REVIEW_REF = 1569
# Ink boxes (x0, x1) of single glyphs in the static f1569 page (Poppins body copy), with the text line
# band they live in: harvested so the new words are set in the page's own font.
REV_LINES = {'L2': (276, 314, 302), 'L3': (316, 349, 338), 'L5': (386, 419, 409), 'L7': (458, 491, 480)}  # y0,y1,baseline
REV_GLYPHS = {'o': ('L3', 364, 377), 'o2': ('L3', 380, 393), 'k': ('L3', 396, 406), 'w': ('L3', 93, 111),
              'a': ('L7', 375, 389), 'v': ('L7', 392, 404), 'e': ('L7', 406, 419), 'I': ('L3', 52, 53)}


def glyph_alpha(gray, line, x0, x1, bg, ink):
    y0, y1, _ = REV_LINES[line]
    sub = gray[y0:y1, x0 - 1:x1 + 2].astype(np.float32)
    a = np.clip((bg - sub) / (bg - ink), 0, 1)
    a[a < 0.04] = 0
    return a


def synth_N(h_band, base, cap_top, stem, width):
    """Poppins-style N from the measured stem width (no N on the page to harvest)."""
    S = 8
    m = np.zeros((h_band * S, int(width + 4) * S), np.uint8)
    xl, xr = 1.0, width + 1.0
    def P(x, y):
        return [int(round(x * S)), int(round(y * S))]
    for xa in (xl, xr - stem):
        cv2.fillPoly(m, [np.array([P(xa, cap_top), P(xa + stem, cap_top), P(xa + stem, base), P(xa, base)])], 255)
    d = stem * 1.12
    cv2.fillPoly(m, [np.array([P(xl, cap_top), P(xl + d, cap_top), P(xr, base), P(xr - d, base)])], 255)
    return cv2.resize(m, (m.shape[1] // S, h_band), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def compose_word(parts, h_band):
    """parts: list of (alpha, gap_before). Returns alpha strip."""
    width = sum(a.shape[1] + g for a, g in parts)
    out = np.zeros((h_band, width), np.float32)
    x = 0
    for a, g in parts:
        x += g
        out[:, x:x + a.shape[1]] = np.maximum(out[:, x:x + a.shape[1]], a)
        x += a.shape[1]
    return out


def fix_review(frames):
    """V-F: f1569-1592 the review page scrolls up (pure vertical translation, no motion blur). Edit the
    static f1569 page once (re-set 'Fisiorest'/'Artuvate's' from harvested page glyphs, erase the bag
    print in the reviewer photo), then paste the edited page regions at each frame's fitted scroll offset."""
    ref = frames[REVIEW_REF]
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgc = np.median(ref[268:282, 60:300].reshape(-1, 3), axis=0).astype(np.float32)
    bg = float(np.median(gray[268:282, 60:300]))
    ink_g = 22.0
    inkc = np.array([ink_g] * 3, np.float32)
    E = ref.astype(np.float32).copy()
    M = np.zeros((H, W), np.float32)
    G = {k: glyph_alpha(gray, ln, x0, x1, bg, ink_g) for k, (ln, x0, x1) in REV_GLYPHS.items()}

    def band_strip(line):
        return REV_LINES[line][1] - REV_LINES[line][0]

    def word(line):
        y0, y1, base = REV_LINES[line]
        hb = y1 - y0
        stem = 2.2
        N = synth_N(hb, base - y0, base - y0 - 17.3, stem, 12.6)
        # glyph strips were cut from other lines; re-register them to this band's baseline
        def reg(k):
            ln = REV_GLYPHS[k][0]
            dy = (REV_LINES[ln][2] - REV_LINES[ln][0]) - (base - y0)
            a = G[k]
            outa = np.zeros((hb, a.shape[1]), np.float32)
            for r in range(hb):
                rr = r + dy
                if 0 <= rr < a.shape[0]:
                    outa[r] = a[rr]
            return outa
        return [(N, 0), (reg('o'), 1), (reg('o2'), 0), (reg('k'), 0), (reg('w'), -1), (reg('a'), 0), (reg('v'), 0)], reg

    def replace(line, x_old0, x_old1, keep_from, extra_parts, x_line_end=470, x_line_start=50, limit=467):
        """Replace ink [x_old0, x_old1] on `line` by the composed word and slide the rest of the line along.
        If the line would run into the photo column it is condensed horizontally to fit (<=5%)."""
        y0, y1, _ = REV_LINES[line]
        parts, _ = word(line)
        alpha = compose_word(parts + extra_parts, y1 - y0)       # 1px margin each side
        orig = E[y0:y1].copy()
        inkcols = np.where((gray[y0:y1, :475] < 150).any(0))[0]
        x_line_end = int(inkcols.max()) + 3
        new_ink_end = x_old0 - 1 + alpha.shape[1] - 2
        shift = new_ink_end - x_old1
        width = x_line_end + shift + 1
        strip = np.tile(bgc, (y1 - y0, width, 1)).astype(np.float32)
        strip[:, :x_old0 - 1] = orig[:, :x_old0 - 1]
        rest = orig[:, keep_from:x_line_end + 1]
        strip[:, keep_from + shift:keep_from + shift + rest.shape[1]] = rest
        a = alpha[..., None]
        seg = strip[:, x_old0 - 1:x_old0 - 1 + alpha.shape[1]]
        strip[:, x_old0 - 1:x_old0 - 1 + alpha.shape[1]] = seg * (1 - a) + inkc * a
        if width > limit:   # condense [x_line_start, width) into [x_line_start, limit)
            part = strip[:, x_line_start:]
            part = cv2.resize(part, (limit - x_line_start, y1 - y0), interpolation=cv2.INTER_AREA)
            strip = np.concatenate([strip[:, :x_line_start], part], axis=1)
            print(f'  {line}: condensed x{(limit - x_line_start) / (width - x_line_start):.3f}')
        E[y0:y1, :strip.shape[1]] = strip
        M[y0 - 1:y1 + 1, x_old0 - 4 if width <= limit else x_line_start:strip.shape[1] + 2] = 1

    # L2: "buying Fisiorest, mainly because" -> "buying Nookwave, mainly because"
    _, reg2 = word('L2')
    replace('L2', 144, 241, 242, [(reg2('e'), 0)])
    # L5: "out to Artuvate's support team to": keep the original "e's", rebuild "Artuvat" -> "Nookwav"
    replace('L5', 130, 219, 220, [])
    # reviewer photo: erase the bag print
    px0, py0, px1, py1 = 524, 298, 632, 344
    sub = E[py0:py1, px0:px1]
    sg = cv2.cvtColor(np.clip(sub, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.dilate(sg, np.ones((9, 9), np.uint8))
    local = cv2.GaussianBlur(local, (0, 0), 3)
    txt = ((local - sg) > 14).astype(np.uint8)
    txt = cv2.dilate(txt, np.ones((3, 3), np.uint8), iterations=1)
    E[py0:py1, px0:px1] = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8), txt * 255, 4,
                                      cv2.INPAINT_TELEA).astype(np.float32)
    M[py0:py1, px0:px1] = np.maximum(M[py0:py1, px0:px1], cv2.dilate(txt, np.ones((7, 7), np.uint8)).astype(np.float32))
    M = cv2.GaussianBlur(M, (0, 0), 1.0)

    # per-frame scroll offset (triage values refined to 0.05 px against the f1569 page)
    tri = dict(json.load(open(JOB / 'analysis' / 'triage.json'))['review_scroll_offsets_px'])
    refg = gray
    out = {}
    for n, f in frames.items():
        if n == REVIEW_REF:
            dy = 0.0
        else:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            best = None
            for d in np.arange(tri[n] - 1.5, tri[n] + 1.5, 0.05):
                Mt = np.float32([[1, 0, 0], [0, 1, d]])
                s = cv2.warpAffine(refg, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                lo = max(0, int(-d) + 100) if d < 0 else 100
                lo = 60
                hi = min(H, int(H + d) - 2)
                e = np.mean((s[lo:hi, 40:520] - g[lo:hi, 40:520]) ** 2)
                if best is None or e < best[0]:
                    best = (e, d)
            dy = best[1]
        Mt = np.float32([[1, 0, 0], [0, 1, dy]])
        e = cv2.warpAffine(E, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(M, Mt, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        if m.max() < 1e-3:
            continue
        out[n] = np.clip(f.astype(np.float32) * (1 - m) + e * m, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------- V-D printed card
CARD_REF = 834
CARD_ROI = (260, 330, 700, 640)       # x0,y0,x1,y1 tracked region around the card text
CARD_WM_SCALE = 0.85                  # new wordmark size relative to the old print (it is a longer word)


def roi_h_to_full(Hr, roi):
    x0, y0 = roi[:2]
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    return T @ Hr.astype(np.float64) @ np.linalg.inv(T)


def motion_blur(img, vx, vy):
    """Linear motion blur along (vx, vy) px (box kernel)."""
    L = float(np.hypot(vx, vy))
    if L < 1.0:
        return img
    n = int(np.ceil(L)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    for t in np.linspace(-0.5, 0.5, 4 * n):
        k[int(round(c + t * vy)), int(round(c + t * vx))] += 1
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def glyph_points(alpha_mask, min_area=40, max_area=6000):
    """Centroids of glyph components sorted left->right (plus the top point of each component)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(alpha_mask.astype(np.uint8))
    idx = [i for i in range(1, n) if min_area <= st[i, 4] <= max_area]
    idx.sort(key=lambda i: cen[i, 0])
    return [cen[i] for i in idx], lab, idx, st


_CARD_CACHE = {}


def card_edit(ref):
    """Edited f834 card (print erased, nookwave re-lettered) + change mask + card mask (ROI)."""
    if 'v' in _CARD_CACHE:
        return _CARD_CACHE['v']
    x0, y0, x1, y1 = CARD_ROI
    g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    roi = g[y0:y1, x0:x1]
    # card surface mask (largest bright component)
    card = (roi > 165).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(card)
    card = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    # old print = dark glyph components enclosed by the card, inside the text band
    cf = cv2.morphologyEx(card, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    dark = ((roi < 175) & (cf > 0)).astype(np.uint8)
    n, labd, std, cen = cv2.connectedComponentsWithStats(dark)
    keep = [i for i in range(1, n) if std[i, 4] > 300 and std[i, 0] > 40 and 90 < cen[i, 1] < 240]
    ink = np.isin(labd, keep).astype(np.uint8)
    ink_full = np.zeros((H, W), np.uint8)
    ink_full[y0:y1, x0:x1] = ink
    core = cv2.erode(ink_full, np.ones((3, 3), np.uint8))
    inkcol = np.percentile(ref[core > 0], 30, axis=0).astype(np.float32)
    # clean plate: inpaint the dilated print
    hole = cv2.dilate(ink_full, np.ones((3, 3), np.uint8), iterations=3) * 255
    clean = cv2.inpaint(ref, hole, 7, cv2.INPAINT_TELEA).astype(np.float32)
    # homography end-card logo -> card print: similarity init from the ink extents, refined by ECC
    eg = WORDMARK_REF_GRAY
    lx0, ly0, lx1, ly1 = LOGO_BOX
    T = (255 - eg[ly0:ly1, lx0:lx1].astype(np.float32)) / 255
    (ccx, ccy), (cw, ch), cang = cv2.minAreaRect(cv2.findNonZero(ink))
    if cw < ch:
        cw, ch, cang = ch, cw, cang + 90
    if cang > 90:
        cang -= 180
    tb = ink_bbox(eg, LOGO_BOX)
    tw = tb[2] - tb[0]
    tcx, tcy = (tb[0] + tb[2]) / 2 - lx0, (tb[1] + tb[3]) / 2 - ly0
    s0 = cw / tw
    R = cv2.getRotationMatrix2D((0, 0), -cang, s0)
    init = np.vstack([R, [0, 0, 1]]).astype(np.float64)
    init[0, 2] = ccx - (R[0, 0] * tcx + R[0, 1] * tcy)
    init[1, 2] = ccy - (R[1, 0] * tcx + R[1, 1] * tcy)
    Iimg = (255 - roi.astype(np.float32)) / 255 * cv2.dilate(ink, np.ones((9, 9), np.uint8))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    Wg = init.astype(np.float32)
    for sig in (3.0, 1.5, 0.8):
        _, Wg = cv2.findTransformECC(cv2.GaussianBlur(T, (0, 0), sig), cv2.GaussianBlur(Iimg, (0, 0), sig * s0),
                                     Wg, cv2.MOTION_HOMOGRAPHY, crit, None, 1)
    Hg = (np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64) @ Wg.astype(np.float64)
          @ np.array([[1, 0, -lx0], [0, 1, -ly0], [0, 0, 1]], np.float64))
    # wordmark in end-card coords, centred on the old logo and scaled
    wm, _, _ = WORDMARK
    old = ink_bbox(eg, LOGO_BOX)
    cxe, cye = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
    rows = np.where(wm.max(1) > 0.02)[0]
    cols = np.where(wm.max(0) > 0.02)[0]
    wmc = wm[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    # place: the strip rows map to y0 of LOGO_BOX + row index (wordmark shares the logo baseline)
    S = 2
    canvas = np.zeros((H * S, W * S), np.float32)
    sc = CARD_WM_SCALE
    # A maps wordmark-strip coords (col, row) -> end-card coords; row r sits at y = ly0 + r
    A = np.array([[sc, 0, cxe - sc * (cols.min() + cols.max()) / 2],
                  [0, sc, cye - sc * (rows.min() + rows.max()) / 2]], np.float64)
    A3 = np.vstack([A, [0, 0, 1]])
    Sm = np.diag([S, S, 1.0])
    Hfull = Sm @ Hg @ A3
    big = cv2.warpPerspective(wm, Hfull, (W * S, H * S), flags=cv2.INTER_LINEAR)
    alpha = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    # match the print's softness (measured on the old ink edges)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    E = clean * (1 - alpha[..., None]) + inkcol * alpha[..., None]
    M = cv2.dilate(np.maximum(hole.astype(np.float32) / 255, (alpha > 0.01).astype(np.float32)),
                   np.ones((5, 5), np.uint8))
    M = cv2.GaussianBlur(M, (0, 0), 1.5)
    _CARD_CACHE['v'] = (E, M, card)
    return _CARD_CACHE['v']


def fix_card(frames):
    """V-D: the 'artuvate' print on the white card (f806-834). The card is near-static from f812; f806-811
    is the whip-in. Erase the print in the sharp f834 reference, re-letter it with the nookwave wordmark
    mapped through the glyph-to-glyph homography (end-card logo -> card print, so perspective and stroke
    style match), then carry the edited card to every frame with an ECC homography track; the two whip
    frames get the overlay motion-blurred to the measured blur."""
    ref = frames[CARD_REF]
    E, M, card = card_edit(ref)
    x0, y0, x1, y1 = CARD_ROI
    roi = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]

    # --- track
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    track_mask = cv2.erode(cv2.morphologyEx(card, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), np.ones((7, 7), np.uint8))
    refg = cv2.GaussianBlur(roi, (0, 0), 1.2).astype(np.float32)
    Hs = {CARD_REF: np.eye(3)}
    Wm = np.eye(3, dtype=np.float32)
    for n in sorted((k for k in frames if k < CARD_REF), reverse=True):
        gi = cv2.GaussianBlur(cv2.cvtColor(frames[n], cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1], (0, 0), 1.2).astype(np.float32)
        try:
            _, Wm = cv2.findTransformECC(refg, gi, Wm, cv2.MOTION_HOMOGRAPHY, crit, track_mask, 5)
        except cv2.error:
            Hs[n] = None
            break
        Hs[n] = roi_h_to_full(Wm, CARD_ROI)
    out = {}
    for n, f in frames.items():
        Hn = Hs.get(n)
        if Hn is None:
            continue
        e = cv2.warpPerspective(E, Hn, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpPerspective(M, Hn, (W, H), flags=cv2.INTER_LINEAR)
        fl = f.astype(np.float32)
        if n in CARD_BLUR:
            # whip frames: fit a linear motion blur on the unedited reference, then add only the
            # (blurred) change so the frame keeps its own blur and noise
            r = cv2.warpPerspective(ref.astype(np.float32), Hn, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            ys, xs = np.where(m > 0.01)
            by0, by1, bx0, bx1 = max(0, ys.min() - 40), min(H, ys.max() + 40), max(0, xs.min() - 40), min(W, xs.max() + 40)
            rc, fc, wc = r[by0:by1, bx0:bx1], fl[by0:by1, bx0:bx1], (m[by0:by1, bx0:bx1] > 0.3)
            best = None
            for vx in range(-40, 41, 2):
                for vy in range(-24, 25, 2):
                    err = np.mean(((motion_blur(rc, vx, vy) - fc) ** 2)[wc])
                    if best is None or err < best[0]:
                        best = (err, vx, vy)
            _, vx, vy = best
            rb = motion_blur(r, vx, vy)
            gain = np.clip(cv2.GaussianBlur(fl, (0, 0), 12) / np.maximum(cv2.GaussianBlur(rb, (0, 0), 12), 1),
                           0.8, 1.25)
            e = motion_blur(e, vx, vy) * gain
            m = cv2.GaussianBlur(cv2.dilate((motion_blur(m, vx, vy) > 0.02).astype(np.float32),
                                            np.ones((9, 9), np.uint8)), (0, 0), 2.5)
        m = m[..., None]
        out[n] = np.clip(fl * (1 - m) + e * m, 0, 255).astype(np.uint8)
    # f806 (heavy whip blur, untrackable): close the dark letter streaks out of the card
    if 806 in frames:
        out[806] = erase_blurred_band(frames[806], [(343, 312), (720, 345), (720, 497), (343, 386)])
    return out


def erase_blurred_band(frame, poly, ring=25, bright=175, feather=5):
    """Fill a polygon of blurred lettering on a white card with a plane fitted (per channel) to the
    bright card pixels in a ring around it, with the frame's noise kept low (heavily blurred source)."""
    m8 = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(m8, [np.int32(poly)], 255)
    ringm = cv2.dilate(m8, np.ones((ring, ring), np.uint8)) & ~m8
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sel = (ringm > 0) & (g > bright)
    ys, xs = np.where(sel)
    A = np.c_[xs, ys, np.ones_like(xs)].astype(np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    fill = np.zeros(frame.shape, np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, frame[ys, xs, c].astype(np.float64), rcond=None)
        fill[..., c] = coef[0] * xx + coef[1] * yy + coef[2]
    m = cv2.GaussianBlur(m8.astype(np.float32) / 255, (0, 0), feather)[..., None]
    return np.clip(frame * (1 - m) + fill * m, 0, 255).astype(np.uint8)


def erase_dark_closing(frame, poly, ksize, feather=6):
    """Remove dark (blurred) lettering on a light surface inside `poly` with a grey closing
    (max then min filter larger than the strokes), blended through a feathered polygon mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, k)
    closed = cv2.GaussianBlur(closed, (0, 0), 3)
    m = np.zeros(frame.shape[:2], np.float32)
    cv2.fillPoly(m, [np.int32(poly)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), feather)[..., None]
    out = frame.astype(np.float32) * (1 - m) + np.maximum(closed, frame).astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


CARD_BLUR = {807, 808, 809}


# ------------------------------------------------------------------------- V-G blurred card shot
VG_SIGMA = 10.0


def fix_blurred_card(frames):
    """V-G f1614-1632: the same card, static and heavily defocused behind 'RIGHT NOW'. Register the
    sharp f834 card to it (ECC homography on the blurred reference), then add the blurred
    (edited - original) card difference, so the blur shows the nookwave word shape instead."""
    ref = CARD_REF_FRAME
    E, M, _ = card_edit(ref)
    x0, y0, x1, y1 = 280, 380, 680, 600
    refg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_n = 1625
    tg = cv2.GaussianBlur(cv2.cvtColor(frames[tgt_n], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1)
    W0 = np.float32([[0.95, 0, x0 - 30], [0, 0.95, y0 - 10], [0, 0, 1]])
    _, Wm = cv2.findTransformECC(cv2.GaussianBlur(refg[y0:y1, x0:x1], (0, 0), VG_SIGMA), tg, W0,
                                 cv2.MOTION_HOMOGRAPHY, (3, 200, 1e-6), None, 1)
    Hm = Wm.astype(np.float64) @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    delta = (E - ref.astype(np.float32)) * (M[..., None] > 0.01)
    d = cv2.warpPerspective(delta, Hm, (W, H), flags=cv2.INTER_LINEAR)
    d = cv2.GaussianBlur(d, (0, 0), VG_SIGMA)
    return {n: np.clip(f.astype(np.float32) + d, 0, 255).astype(np.uint8) for n, f in frames.items()}


# ---------------------------------------------------------------------------- caption "FisioRest"
CAP_BOX = (240, 905, 480, 980)        # search box of the burned-in single-word caption (f806-834)
CAP_AXES = (105, 650)                 # SF Pro variable width / weight that matches the caption face
CAP_STROKE = 1.8                      # dark outline width (px)


def caption_mask(frames):
    """White fill of the old caption: pixels that are near-white in (almost) every frame of the range."""
    x0, y0, x1, y1 = CAP_BOX
    stack = np.stack([cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) for f in frames.values()])
    white = (np.percentile(stack, 20, axis=0) > 200).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    keep = [i for i in range(1, n) if st[i, 4] >= 6]
    white = np.isin(lab, keep).astype(np.uint8)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = white
    return full


def render_caption(text, size, xscale, cx, top_ref, ref_text):
    """RGBA-ish (fill alpha, stroke alpha) of `text` in the caption face, aligned like `ref_text`."""
    S = 4
    f = ImageFont.truetype(SFNS, int(round(size * S)))
    f.set_variation_by_axes([CAP_AXES[0], 28, 400, CAP_AXES[1]])
    sw = int(round(CAP_STROKE * S))
    rb = f.getbbox(ref_text)
    tb = f.getbbox(text)
    pad = 12 * S
    wpx = tb[2] - tb[0] + 2 * pad
    hpx = max(rb[3], tb[3]) - min(rb[1], tb[1]) + 2 * pad
    top = min(rb[1], tb[1])
    fill = Image.new('L', (wpx, hpx), 0)
    stroke = Image.new('L', (wpx, hpx), 0)
    ImageDraw.Draw(stroke).text((pad - tb[0], pad - top), text, font=f, fill=255, stroke_width=sw, stroke_fill=255)
    ImageDraw.Draw(fill).text((pad - tb[0], pad - top), text, font=f, fill=255)
    out = []
    for im in (fill, stroke):
        a = np.array(im, np.float32) / 255
        a = cv2.resize(a, (int(round(wpx * xscale / S)), int(round(hpx / S))), interpolation=cv2.INTER_AREA)
        out.append(a)
    fa, sa = out
    canvas_f = np.zeros((H, W), np.float32)
    canvas_s = np.zeros((H, W), np.float32)
    oy = int(round(top_ref - (pad - top + rb[1]) / S))
    ox = int(round(cx - fa.shape[1] / 2))
    canvas_f[oy:oy + fa.shape[0], ox:ox + fa.shape[1]] = fa
    canvas_s[oy:oy + sa.shape[0], ox:ox + sa.shape[1]] = sa
    return canvas_f, canvas_s


def fix_caption(frames):
    """Burned-in caption 'FisioRest' (f806-834, no box, white + thin dark outline) -> 'Nookwave'.
    The old word (fill + outline) is inpainted per frame, then the new word is drawn in the matched
    face (SF Pro, width/weight fitted to the old glyph box), same centre and baseline."""
    white = caption_mask(frames)
    ys, xs = np.where(white > 0)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    # fit size + horizontal scale on the old word's white-fill box
    S = 4
    f = ImageFont.truetype(SFNS, 200)
    f.set_variation_by_axes([CAP_AXES[0], 28, 400, CAP_AXES[1]])
    b = f.getbbox('FisioRest')
    size = 200 * (by1 - by0 + 1) / (b[3] - b[1])
    xscale = (bx1 - bx0 + 1) / ((b[2] - b[0]) * size / 200)
    cx = (bx0 + bx1 + 1) / 2
    fa, sa = render_caption('Nookwave', size, xscale, cx, by0, 'FisioRest')
    hole = cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=3) * 255
    out = {}
    for n, fr in frames.items():
        base = cv2.inpaint(fr, hole, 4, cv2.INPAINT_TELEA).astype(np.float32)
        base = base * (1 - sa[..., None]) + 8.0 * sa[..., None]          # outline (near-black)
        base = base * (1 - fa[..., None]) + 252.0 * fa[..., None]        # white fill
        out[n] = np.clip(base, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------- V-A/B/C dust-bag print
# (first, last, anchor frame, text quad at anchor (x0,y0,x1,y1), print-through ghost?, protect caption box)
BAGS = {
    'VA': (0, 63, 0, (240, 715, 425, 825), True, None),
    'VB': (634, 660, 634, (190, 525, 600, 635), False, None),
    'VC': (776, 804, 776, (335, 965, 625, 1055), False, (0, 905, 720, 985)),
}


def track_quads(frames, n0, quad0, pad=25):
    """Chained frame-to-frame ECC affine on the patch around the print; returns a quad per frame."""
    rng = (min(frames), max(frames))
    quads = {n0: np.float32(quad0)}
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    gray = {n: cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
            for n, f in frames.items()}
    for step in (1, -1):
        q = np.float32(quad0)
        n = n0
        while rng[0] <= n + step <= rng[1]:
            x, y, w, h = cv2.boundingRect(q.astype(np.int32))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            Wm = np.float32([[1, 0, x0], [0, 1, y0]])
            try:
                _, Wm = cv2.findTransformECC(gray[n][y0:y1, x0:x1], gray[n + step], Wm, cv2.MOTION_AFFINE,
                                             crit, None, 5)
            except cv2.error:
                pass
            q = ((q - np.float32([x0, y0])) @ Wm[:, :2].T + Wm[:, 2]).astype(np.float32)
            n += step
            quads[n] = q
    return quads


def text_line_filter(keep, stt, cen):
    """Keep only components on the printed word's line: weighted line fit through the component
    centroids, drop those far off the line or separated from the main cluster by a big gap."""
    if len(keep) < 3:
        return keep
    idx = np.array(keep)
    P = cen[idx].astype(np.float64)
    wts = stt[idx, 4].astype(np.float64)
    mu = (P * wts[:, None]).sum(0) / wts.sum()
    C = ((P - mu) * wts[:, None]).T @ (P - mu) / wts.sum()
    ev, evec = np.linalg.eigh(C)
    u = evec[:, 1]
    nrm = np.array([-u[1], u[0]])
    big = wts >= np.percentile(wts, 50)
    hm = np.median(np.maximum(stt[idx, 2], stt[idx, 3])[big])
    dist = np.abs((P - mu) @ nrm)
    tpos = (P - mu) @ u
    ok = dist < 0.8 * hm
    order = np.argsort(tpos)
    clusters, cur = [], []
    last = None
    for j in order:
        if not ok[j]:
            continue
        if last is not None and tpos[j] - last > 2.2 * hm:
            clusters.append(cur)
            cur = []
        cur.append(j)
        last = tpos[j]
    if cur:
        clusters.append(cur)
    best = max(clusters, key=lambda c: wts[c].sum())
    return [int(idx[j]) for j in best]


def letter_mask(f, quad, grow_q=1.15, core_d=60, core_g=140, weak_d=28, weak_iter=3):
    """Dark, low-chroma strokes inside the tracked quad whose surroundings are white fabric."""
    c = quad.mean(0)
    q = (quad - c) * grow_q + c
    x, y, w, h = cv2.boundingRect(q.astype(np.int32))
    pad = 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    sub = f[y0:y1, x0:x1]
    g = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    sb = sub.astype(np.int16)
    chroma = sb.max(2) - sb.min(2)
    qh = np.linalg.norm(quad[3] - quad[0])
    k = int(max(15, min(0.55 * qh, 121))) | 1
    clo = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    d = clo - g
    poly = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(poly, [(q - np.float32([x0, y0])).astype(np.int32)], 1)
    core = ((d > core_d) & (g < core_g) & (chroma < 45) & (clo > 150) & (poly > 0)).astype(np.uint8)
    weak = ((d > weak_d) & (chroma < 50) & (poly > 0)).astype(np.uint8)
    whiteish = ((g > 150) & (chroma < 60)).astype(np.uint8)
    n, lab, stt, cen = cv2.connectedComponentsWithStats(core)
    keep = []
    for i in range(1, n):
        bx, by, bw, bh = stt[i, :4]
        ax0, ay0 = max(0, bx - 8), max(0, by - 8)
        cc = (lab[ay0:by + bh + 8, ax0:bx + bw + 8] == i).astype(np.uint8)
        ring = cv2.dilate(cc, np.ones((13, 13), np.uint8)) - cv2.dilate(cc, np.ones((5, 5), np.uint8))
        fr = whiteish[ay0:by + bh + 8, ax0:bx + bw + 8][ring > 0].mean() if ring.any() else 0
        inside = cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(cen[i, 0] + x0), float(cen[i, 1] + y0)), False) >= 0
        if fr > 0.25 and inside:
            keep.append(i)
    keep = text_line_filter(keep, stt, cen)
    m = np.isin(lab, keep).astype(np.uint8)
    for _ in range(weak_iter):
        m = (cv2.dilate(m, np.ones((3, 3), np.uint8)) & weak) | m
    dk = ((g < 120) & (chroma < 45) & (poly > 0)).astype(np.uint8)
    for _ in range(6):
        m = (cv2.dilate(m, np.ones((3, 3), np.uint8)) & (dk | weak)) | m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    inv = (m == 0).astype(np.uint8)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(inv)
    for i in range(1, n2):           # fill enclosed holes (letter counters, unmasked specks)
        bx, by, bw, bh = s2[i, :4]
        if bx > 0 and by > 0 and bx + bw < m.shape[1] and by + bh < m.shape[0] and s2[i, 4] < 700:
            m[l2 == i] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3 + int(qh / 35))   # incl. sharpening halo
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    return full


def deghost(f, quad, grow=1.1, gain=1.3):
    """Suppress the faint print-through 'ate' (brighter strokes on the light-blue part of the bag):
    pull mid-frequency highlights back to the local median, keeping the fine mesh texture."""
    v = quad[3] - quad[0]
    c = quad.mean(0)
    q = (quad - c) * grow + c
    x, y, w, h = cv2.boundingRect(q.astype(np.int32))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return f
    sub = f[y0:y1, x0:x1]
    L = cv2.GaussianBlur(sub, (0, 0), 2.0).astype(np.float32)
    k = int(max(21, 0.12 * np.linalg.norm(v))) | 1
    B = cv2.medianBlur(sub, k).astype(np.float32)
    D = L - B
    s = sub.astype(np.int16)
    skin = (s[..., 2] - s[..., 0]) > 5
    poly = np.zeros(sub.shape[:2], np.uint8)
    cv2.fillPoly(poly, [(q - np.float32([x0, y0])).astype(np.int32)], 1)
    bag = ((L.mean(2) > 150) & (~skin) & (poly > 0) & ((B[..., 0] - B[..., 2]) > 6)).astype(np.uint8)
    bag = cv2.erode(bag, np.ones((5, 5), np.uint8))
    a = cv2.GaussianBlur(bag.astype(np.float32), (0, 0), 3)[..., None]
    out = f.copy()
    out[y0:y1, x0:x1] = np.clip(sub - gain * np.clip(D, 0, 45) * a, 0, 255).astype(np.uint8)
    return out


def normconv_fill(f, mask, quad):
    """Fill masked pixels with the Gaussian-weighted mean of the unmasked surroundings (multi-scale
    normalized convolution) - smooth like the fabric's shading, no Telea streaks."""
    qh = np.linalg.norm(quad[3] - quad[0])
    ff = f.astype(np.float32)
    sb = f.astype(np.int16)
    fabric = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 150) & ((sb.max(2) - sb.min(2)) < 45)
    keep = ((mask == 0) & fabric).astype(np.float32)
    out = ff.copy()
    done = keep.copy()
    for sig in (max(3.0, 0.05 * qh), max(6.0, 0.1 * qh), max(12.0, 0.2 * qh), 40.0):
        wsum = cv2.GaussianBlur(keep, (0, 0), sig)
        vals = cv2.GaussianBlur(ff * keep[..., None], (0, 0), sig) / np.maximum(wsum, 1e-4)[..., None]
        take = (done == 0) & (wsum > 0.08)
        out[take] = vals[take]
        done[take] = 1
    rest = (done == 0)
    if rest.any():                      # far from any fabric sample: widest support available
        wsum = cv2.GaussianBlur(keep, (0, 0), 90)
        vals = cv2.GaussianBlur(ff * keep[..., None], (0, 0), 90) / np.maximum(wsum, 1e-6)[..., None]
        out[rest] = vals[rest]
    return out


def fabric_fill(f, mask, quad, seed=0, extra=0):
    """Fill the print (plus a margin that holds the sharpening halo) from the surrounding fabric
    (normalized convolution), then add fine grain matched to the fabric's own high-frequency level."""
    mask = cv2.dilate(mask, np.ones((5 + 2 * extra, 5 + 2 * extra), np.uint8))
    sb = f.astype(np.int16)
    fabric = ((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 150) & ((sb.max(2) - sb.min(2)) < 45)).astype(np.uint8)
    allowed = cv2.dilate(fabric | mask, np.ones((5, 5), np.uint8))
    kq = int(0.45 * np.linalg.norm(quad[3] - quad[0])) | 1      # band-shaped fill: no letter-shaped seams
    band = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kq, kq)))
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    band = cv2.drawContours(np.zeros_like(band), cnts, -1, 1, -1)        # fill letter counters
    hull = np.zeros_like(mask)
    pts = cv2.findNonZero(mask)
    if pts is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    mask = mask | (band & allowed & hull)
    bagarea = cv2.morphologyEx(fabric, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kq, kq)))
    mask = mask & cv2.dilate(bagarea, np.ones((15, 15), np.uint8))    # never paint past the bag's edge
    filled = normconv_fill(f, mask, quad)
    ff = f.astype(np.float32)
    hp = ff - cv2.GaussianBlur(ff, (0, 0), 1.2)
    ring = (cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0) & (mask == 0)
    sd = float(np.median(np.abs(hp[ring]))) * 1.4 if ring.any() else 3.0
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, f.shape[:2]).astype(np.float32), (0, 0), 0.7)
    noise *= sd / max(noise.std(), 1e-6)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
    out = ff * (1 - a) + (filled + noise[..., None]) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def fix_bag_factory(key):
    first, last, anchor, box, ghost, protect = BAGS[key]

    def fix_bag(frames):
        """Dust-bag print: track the print with chained ECC affine quads, mask dark low-chroma strokes
        that sit on white fabric inside the quad, fill them (Telea + copied mesh texture). A burned-in
        caption touching the print is protected and pasted back unchanged."""
        x0, y0, x1, y1 = box
        quads = track_quads(frames, anchor, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        cap = None
        if protect is not None:
            px0, py0, px1, py1 = protect
            # burned-in caption = pixels that stay pure white (fill) or stay dark right next to it
            # (outline) in every frame, while the bag behind it moves
            stack = np.stack([cv2.cvtColor(f[py0:py1, px0:px1], cv2.COLOR_BGR2GRAY)
                              for f in frames.values()]).astype(np.float32)
            fill = (np.percentile(stack, 10, axis=0) > 243).astype(np.uint8)
            outline = (cv2.dilate(fill, np.ones((5, 5), np.uint8)) > 0) & (np.percentile(stack, 90, axis=0) < 110)
            cap = np.zeros((H, W), np.uint8)
            cap[py0:py1, px0:px1] = (fill | outline).astype(np.uint8)
        out = {}
        for n, f in frames.items():
            blurry = n in BAG_BLUR
            m = letter_mask(f, quads[n], core_d=40 if blurry else 60, core_g=165 if blurry else 140,
                            **BAG_WEAK.get(key, {}))
            if cap is not None:
                m = np.maximum(m, cap)         # fill under the caption too, then paste the caption back
            r = fabric_fill(f, m, quads[n], seed=n, extra=BAG_EXTRA.get(key, 0))
            if cap is not None:
                ca = cv2.GaussianBlur(cap.astype(np.float32), (0, 0), 0.8)[..., None]
                r = np.clip(f * ca + r * (1 - ca), 0, 255).astype(np.uint8)
            if ghost:
                r = deghost(r, quads[n])
            out[n] = r
        return out
    return fix_bag


BAG_BLUR = {801, 802, 803, 804}
BAG_EXTRA = {'VB': 3}
BAG_WEAK = {'VB': dict(weak_d=16, weak_iter=7), 'VC': dict(weak_d=20, weak_iter=5)}


# ------------------------------------------------------------------------- COSSEAL (3D animation)
# Third-party brand embossed on the device in the stock 3D split-screen (f858-918), client-approved
# removal. T1 = top panel plate, T2 = middle panel left pillow tag, T3 = bottom panel end tag.
TAGS = {
    'T1': dict(anchor=900, box=(316, 288, 477, 344), first=862, erode=5, thr=14, seg='bright', track='quad'),
    'T2': dict(anchor=858, box=(96, 668, 140, 792), first=858, erode=0, thr=20, seg='dark', track='highpass',
               field=[(108, 668), (121, 664), (124, 678), (121, 730), (118, 756), (110, 761), (104, 754), (101.5, 718), (103, 686)]),
}


def tag_region(f, quad, seg):
    """Tag surface inside the tracked quad: warm-coloured (leather) pixels, bright (T1 beige plate) or
    dark (T2 brown tag), closed over the lettering and hole-filled."""
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(W, x + w + 4), min(H, y + h + 4)
    sub = cv2.GaussianBlur(f[y0:y1, x0:x1], (0, 0), 1.0).astype(np.float32)
    g = sub.mean(2)
    warm = sub[..., 2] - sub[..., 0]
    if seg == 'bright':
        m = (g > 135) & (warm > 8)
    else:
        m = (g < 125) & (warm > 12)
    poly = np.zeros(sub.shape[:2], np.uint8)
    cv2.fillPoly(poly, [(quad - np.float32([x0, y0])).astype(np.int32)], 1)
    m = (m & (poly > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, (x0, y0, x1, y1)
    m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = cv2.drawContours(np.zeros_like(m), cnts, -1, 1, -1)
    return m, (x0, y0, x1, y1)


def fill_inner(f, region, margin, seed, thr=14, noise_gain=0.6):
    """Remove engraved (dark) lettering from a leather tag: dark-relative-to-closing strokes at least
    `margin` px inside the tag outline (the stitching sits closer to the edge), filled from the clean
    leather around them (normalized convolution) with faint grain."""
    ys, xs = np.where(region > 0)
    if len(xs) == 0:
        return f
    x0, y0 = max(0, xs.min() - 30), max(0, ys.min() - 30)
    x1, y1 = min(W, xs.max() + 31), min(H, ys.max() + 31)
    sub = f[y0:y1, x0:x1].astype(np.float32)
    reg = region[y0:y1, x0:x1]
    g = cv2.GaussianBlur(sub.mean(2), (0, 0), 0.8)
    clo = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    dist = cv2.distanceTransform(reg, cv2.DIST_L2, 3)
    dark = (clo - g) > thr
    marks = (dark & (dist > margin)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(marks)
    marks = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 4]).astype(np.uint8)
    fillm = cv2.dilate(marks, np.ones((3, 3), np.uint8), iterations=2)
    fillm = cv2.morphologyEx(fillm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    fillm &= (dist > margin - 2).astype(np.uint8)
    keep = ((reg > 0) & (fillm == 0) & ~cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ).astype(np.float32)
    out = sub.copy()
    done = keep.copy()
    for sig in (2.0, 4.0, 8.0, 16.0, 40.0):
        wsum = cv2.GaussianBlur(keep, (0, 0), sig)
        vals = cv2.GaussianBlur(sub * keep[..., None], (0, 0), sig) / np.maximum(wsum, 1e-6)[..., None]
        take = (done == 0) & (wsum > 0.05)
        out[take] = vals[take]
        done[take] = 1
    hp = sub - cv2.GaussianBlur(sub, (0, 0), 1.0)
    sd = float(np.median(np.abs(hp[keep > 0]))) * noise_gain if keep.any() else 1.5
    noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, keep.shape).astype(np.float32), (0, 0), 0.8)
    noise *= sd / max(noise.std(), 1e-6)
    a = cv2.GaussianBlur(fillm.astype(np.float32), (0, 0), 1.0)[..., None]
    res = f.copy()
    res[y0:y1, x0:x1] = np.clip(sub * (1 - a) + (out + noise[..., None]) * a, 0, 255).astype(np.uint8)
    return res


def track_highpass(frames, anchor, box, pad=25):
    """Affine of the patch around `box` from the anchor frame to every frame, matched on a high-pass
    image (insensitive to the animation's glow/lighting sweeps); each frame is matched to the anchor
    template directly, initialised from the neighbouring frame's result."""
    def hp(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(g - cv2.GaussianBlur(g, (0, 0), 4), (0, 0), 1)
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    tpl = hp(frames[anchor])[y0:y1, x0:x1]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    res = {anchor: np.eye(3)[:2]}
    for seq in (sorted(n for n in frames if n > anchor), sorted((n for n in frames if n < anchor), reverse=True)):
        Wm = np.float32([[1, 0, x0], [0, 1, y0]])
        for n in seq:
            try:
                _, Wm = cv2.findTransformECC(tpl, hp(frames[n]), Wm.copy(), cv2.MOTION_AFFINE, crit, None, 1)
            except cv2.error:
                pass
            res[n] = (np.vstack([Wm, [0, 0, 1]]).astype(np.float64) @ T)[:2]
    return res


def fix_tags(frames):
    """Segment each tag once at a clean anchor frame, carry its outline to every frame with a high-pass
    ECC affine track, and refill the engraved lettering from the surrounding leather."""
    out = {}
    for key, p in TAGS.items():
        sub = {n: out.get(n, f) for n, f in frames.items() if n >= p['first']}
        x0, y0, x1, y1 = p['box']
        qa = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        c = qa.mean(0)
        R = np.zeros((H, W), np.uint8)
        if 'field' in p:            # hand-drawn field inside the stitching (small dark tag, glow-lit)
            cv2.fillPoly(R, [np.int32(np.round(np.float32(p['field']) * 4))], 1, shift=2)
        else:
            region, box = tag_region(sub[p['anchor']], (qa - c) * 1.12 + c, p['seg'])
            R[box[1]:box[3], box[0]:box[2]] = region
        if p['track'] == 'highpass':
            A = track_highpass(sub, p['anchor'], p['box'])
        else:
            quads = track_quads(sub, p['anchor'], qa, pad=12)
            A = {n: cv2.getAffineTransform(np.float32(qa[:3]), np.float32(q[:3])) for n, q in quads.items()}
        for n, f in sub.items():
            Rn = cv2.warpAffine(R, A[n].astype(np.float32), (W, H), flags=cv2.INTER_NEAREST)
            out[n] = fill_inner(f, Rn, p['erode'], seed=n, thr=p['thr'])
    return out


# T3: embossed pill on the end of the bottom-panel device. It slides left, is crossed by a translucent
# dark panel (f878-895) and stays visible through it to f918; ECC tracking is unreliable here, so the
# lettering field (inside the pill rim) is keyframed by hand and interpolated.
T3_KEYS = {  # frame: (x0, x1, y0, y1) of the lettering field
    862: (597, 627, 1034, 1128), 866: (623, 652, 1034, 1128), 870: (629, 657, 1034, 1128),
    874: (633, 660, 1034, 1128), 878: (637, 668, 1034, 1128), 882: (638, 676, 1034, 1128),
    886: (638, 688, 1034, 1128), 890: (638, 692, 1034, 1128), 895: (638, 690, 1034, 1128),
    900: (639, 676, 1034, 1128), 906: (637, 684, 1034, 1128), 912: (632, 682, 1034, 1128),
    918: (622, 675, 1034, 1128),
}


def t3_field(n):
    ks = sorted(T3_KEYS)
    if n < ks[0] or n > ks[-1]:
        return None
    a = max(k for k in ks if k <= n)
    b = min(k for k in ks if k >= n)
    t = 0 if a == b else (n - a) / (b - a)
    return tuple((1 - t) * np.float32(T3_KEYS[a]) + t * np.float32(T3_KEYS[b]))


def column_fit_fill(f, rect, seed, radius=12, iters=4):
    """Per-column robust quadratic fit of the shading inside a pill-shaped field: removes horizontal
    lettering strokes (text runs vertically) while keeping vertical edges such as the panel border."""
    x0, x1, y0, y1 = [int(round(v)) for v in rect]
    sub = f[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape[:2]
    m = np.zeros((h, w), np.uint8)
    r = int(min(radius, w // 2))
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 1, -1)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 1, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 1, -1)
    fit = sub.copy()
    yy = np.arange(h, dtype=np.float64)
    for x in range(w):
        rows = np.where(m[:, x] > 0)[0]
        if len(rows) < 8:
            continue
        for c in range(3):
            v = sub[rows, x, c].astype(np.float64)
            use = np.ones(len(rows), bool)
            for _ in range(iters):
                coef = np.polyfit(rows[use], v[use], 2)
                res = v - np.polyval(coef, rows)
                s = max(2.0, 1.4826 * np.median(np.abs(res[use])))
                use = np.abs(res) < 2.0 * s
                if use.sum() < 6:
                    use = np.ones(len(rows), bool)
                    break
            fit[rows, x, c] = np.polyval(coef, rows)
    fit = cv2.GaussianBlur(fit, (3, 1), 0)          # tiny horizontal smoothing between columns
    noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    noise *= 0.6 / max(noise.std(), 1e-6)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[..., None]
    out = f.copy()
    out[y0:y1, x0:x1] = np.clip(sub * (1 - a) + (fit + noise[..., None]) * a, 0, 255).astype(np.uint8)
    return out


def fix_tag3(frames):
    out = {}
    for n, f in frames.items():
        rect = t3_field(n)
        if rect is not None:
            out[n] = column_fit_fill(f, rect, seed=n)
    return out


# ------------------------------------------------------------------------------------------- items
ITEMS = {
    'VA': ((0, 63), fix_bag_factory('VA')),
    'VB': ((634, 660), fix_bag_factory('VB')),
    'VC': ((776, 804), fix_bag_factory('VC')),
    'VE': ((1500, 1521), fix_website),
    'VF': ((1569, 1592), fix_review),
    'VD': ((806, 834), fix_card),
    'CAP': ((806, 834), fix_caption),
    'VG': ((1614, 1632), fix_blurred_card),
    'VH': ((1849, 1922), fix_endcard),
    'TAG': ((858, 918), fix_tags),
    'TAG3': ((858, 918), fix_tag3),
}
WORDMARK = None
WORDMARK_REF_GRAY = None
CARD_REF_FRAME = None


def run(only=None, qc=False, write=True):
    global WORDMARK, WORDMARK_REF_GRAY, CARD_REF_FRAME
    items = {k: v for k, v in ITEMS.items() if only is None or k in only}
    frames = load_ranges([r for r, _ in items.values()] + [(ENDCARD_REF, ENDCARD_REF), (CARD_REF, CARD_REF)])
    CARD_REF_FRAME = frames[CARD_REF]
    WORDMARK_REF_GRAY = cv2.cvtColor(frames[ENDCARD_REF], cv2.COLOR_BGR2GRAY)
    WORDMARK = build_wordmark(WORDMARK_REF_GRAY)
    edited = {}
    for key, ((a, b), fn) in items.items():
        sub = {n: (edited.get(n, frames[n])) for n in range(a, b + 1)}
        res = fn(sub)
        edited.update(res)
        print(key, 'edited', len(res), 'frames', flush=True)
    return frames, edited


# ---------------------------------------------------------------------------------------- render / QC
# item: list of (crop x0,y0,x1,y1, scale, frames)
QC = {
    'VA_bag_f0-63': [((0, 500, 720, 1000), 0.5, list(range(0, 64, 4)) + [63])],
    'VB_bag_f634-660': [((0, 420, 720, 760), 0.5, list(range(634, 661, 3)) + [660])],
    'VC_bag_f776-804': [((280, 880, 720, 1100), 0.7, list(range(776, 805, 3)) + [802, 803, 804])],
    'VD_card_f806-834': [((250, 300, 720, 650), 0.6, [806, 807, 808, 809, 810] + list(range(812, 835, 4)) + [834])],
    'CAP_caption_f806-834': [((200, 890, 520, 990), 1.5, [806, 807, 812, 820, 828, 834])],
    'COSSEAL_top_f862-918': [((300, 270, 520, 360), 1.4, list(range(862, 919, 4)) + [918])],
    'COSSEAL_left_f858-918': [((60, 640, 170, 810), 1.2, list(range(858, 919, 4)) + [918])],
    'COSSEAL_bottom_f862-918': [((570, 990, 720, 1150), 1.2, list(range(862, 919, 4)) + [918])],
    'VE_web_header_f1500-1513': [((200, 0, 520, 70), 1.2, [1500, 1504, 1507, 1510, 1513])],
    'VE_web_title_f1500-1521': [((0, 1030, 300, 1100), 1.2, [1500, 1505, 1510, 1515, 1521])],
    'VE_web_labels_f1500-1521': [((470, 870, 620, 930), 2.5, [1500, 1510, 1521])],
    'VF_review_f1569-1592': [((0, 0, 720, 640), 0.5, list(range(1569, 1593, 3)) + [1592])],
    'VG_blurred_card_f1614-1632': [((220, 360, 700, 700), 0.6, [1614, 1623, 1632])],
    'VH_endcard_f1856-1922': [((100, 240, 620, 380), 0.8, list(range(1856, 1872, 2)) + [1880, 1900, 1922])],
    'WATERMARK_untouched_f240-263': [((0, 0, 420, 220), 0.8, [240, 251, 263])],
}


def write_qc(orig, edited, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, specs in QC.items():
        for (x0, y0, x1, y1), sc, fl in specs:
            rows = []
            for n in fl:
                a = orig[n][y0:y1, x0:x1]
                b = edited.get(n, orig[n])[y0:y1, x0:x1]
                im = np.hstack([a, np.full((a.shape[0], 4, 3), (0, 0, 255), np.uint8), b])
                im = cv2.resize(im, None, fx=sc, fy=sc,
                                interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
                cv2.putText(im, f'f{n}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                rows.append(im)
            # wrap into a grid of at most ~1800 px wide
            per = max(1, int(1800 // rows[0].shape[1]))
            grid = []
            for i in range(0, len(rows), per):
                r = rows[i:i + per]
                r += [np.zeros_like(rows[0])] * (per - len(r))
                grid.append(np.hstack(r))
            cv2.imwrite(str(outdir / f'{name}.jpg'), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 90])


def render(edited):
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen([FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                            '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                            '-pix_fmt', 'yuv420p', '-r', '30', '-movflags', '+faststart', str(OUT_VIDEO)],
                           stdin=subprocess.PIPE)
    count = 0
    for n, f in decode():
        enc.stdin.write(edited.get(n, f).tobytes())
        count += 1
    enc.stdin.close()
    enc.wait()
    assert count == NFRAMES, count
    OUT_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(OUT_VIDEO), '-i', str(SRC), '-map', '0:v:0',
                    '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy', str(OUT_PREVIEW)], check=True)
    print('wrote', OUT_VIDEO, 'and', OUT_PREVIEW, 'frames', count)


if __name__ == '__main__':
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1].split(',')
    frames, edited = run(only)
    if '--qc' in sys.argv or only is None:
        need = sorted({n for specs in QC.values() for _, _, fl in specs for n in fl} - set(frames))
        extra = dict(decode(set(need))) if need else {}
        write_qc({**frames, **extra}, edited, QC_DIR)
        print('qc ->', QC_DIR)
    if only is None:
        render(edited)
