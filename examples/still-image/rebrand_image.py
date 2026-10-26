#!/usr/bin/env python3
"""Rebrand box artwork in images/source/3.jpg: FisioRest -> Nookwave, artuvate -> nookwave.

Re-runnable:  ~/.cache/video-studio-venv/bin/python -I rebrand_image3.py
Per text line: detect the old ink inside a hand-placed ROI polygon, measure its
orientation / extent in the text frame (PCA), erase it with a quadratic fit of
the surrounding box background (+ matched grain), render the new word flat with
Pillow at 8x, map it with an affine (rotation + per-axis scale fitted from the
old word rendered in the same font), soften to match the photo, composite.
Only pixels inside each ROI are touched; source is never modified.
"""
import os, sys, io
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "source", "3.jpg")
OUT = os.path.join(ROOT, "Nookwave_image3.jpg")
QC = os.path.join(ROOT, "qc")
HF = os.path.expanduser("~/Library/Fonts/")

TITLE_FONT = (HF + "Avenir Next.ttc", 5)                       # Avenir Next Medium
WORD_FONT = ("/System/Library/Fonts/Supplemental/Futura.ttc", 0)  # Futura Medium

def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

# name, old, new, font, roi polygon, box crop for QC
ITEMS = [
    ("B1", "T", rect(330, 12, 452, 40), rect(250, 10, 300, 30)),
    ("B2", "T", rect(614, 12, 706, 38), rect(538, 7, 592, 27)),
    ("B3", "T", rect(338, 294, 426, 321), rect(263, 291, 311, 310)),
    ("B4", "T", rect(617, 294, 706, 321), rect(540, 289, 591, 309)),
    ("B5", "T", rect(351, 573, 436, 598), None),   # wordmark hidden by held box
    ("B6", "T", rect(619, 573, 706, 598), rect(543, 569, 593, 588)),
    ("B7", "T", rect(357, 805, 441, 828), rect(283, 797, 329, 816)),
    ("B8", "T", rect(619, 808, 706, 835), rect(544, 802, 593, 822)),
    ("H1", "T", [(145, 452), (230, 440), (232, 462), (147, 473)], rect(59, 452, 109, 475)),
    ("H2", "T", [(854, 409), (949, 427), (947, 450), (857, 434)], rect(769, 387, 821, 414)),
]
QC_BOX = {"B1": (245, 0, 520, 240), "B2": (530, 0, 805, 240), "B3": (255, 280, 525, 515),
          "B4": (530, 280, 805, 515), "B5": (270, 555, 530, 785), "B6": (535, 555, 800, 790),
          "B7": (272, 785, 535, 1010), "B8": (535, 785, 800, 1015), "H1": (25, 405, 360, 715),
          "H2": (710, 375, 1070, 700)}

TITLE_ASPECT = 0.82   # median measured sx/sy across boxes
WORD_ASPECT = 0.85
TITLE_SIGMA = 0.35
WORD_SIGMA = 0.3
WORD_DARKEN = 4.0
WORD_STROKE = 3   # px at 320px render size: Futura Medium is a touch lighter than the original wordmark
SS = 8  # supersampling for rendering


def poly_mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 255)
    return m > 0


def ink_bbox(text, font, stroke=0):
    im = Image.new("L", (font.size * len(text), font.size * 2), 0)
    ImageDraw.Draw(im).text((font.size // 2, font.size // 2), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    a = np.array(im)
    ys, xs = np.nonzero(a > 127)
    return im, a, xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def process(img, poly, old, new, fontspec, log, aspect, col_override=None, sigma=0.35, align='center', darken=0.0, stroke=0):
    H, W = img.shape[:2]
    roi = poly_mask(img.shape, poly)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = np.median(gray[roi])
    ink = roi & (gray < med - 45)
    # drop specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 2:
            keep |= lab == i
    ink = keep
    ys, xs = np.nonzero(ink)
    w = (med - gray[ys, xs])
    cx, cy = np.average(xs, weights=w), np.average(ys, weights=w)
    cov = np.cov(np.vstack([xs - cx, ys - cy]), aweights=w)
    ev, evec = np.linalg.eigh(cov)
    u = evec[:, 1]
    if u[0] < 0:
        u = -u
    # PCA is biased by glyph distribution; refine angle from the baseline
    # (neither word has descenders): bottom-most ink per column, robust line fit
    bx, by = [], []
    for x in np.unique(xs):
        bx.append(x); by.append(ys[xs == x].max())
    bx, by = np.array(bx, float), np.array(by, float)
    keepb = np.ones(len(bx), bool)
    # lower envelope: column bottoms that are not on the baseline lie ABOVE it
    # (F arms, i-dots, s/e terminals), so iteratively drop points above the fit
    for _ in range(8):
        a_, b_ = np.polyfit(bx[keepb], by[keepb], 1)
        r = by - (a_ * bx + b_)
        keepb = r >= -0.8
    u = np.array([1.0, a_]) / np.hypot(1.0, a_)
    ang = np.arctan2(u[1], u[0])
    vv =np.array([-u[1], u[0]])  # points "down" in text frame
    # colour of ink cores
    core = gray[ys, xs]
    sel = core <= np.percentile(core, 25)
    col = np.median(img[ys[sel], xs[sel]].astype(np.float32), axis=0)
    # extents measured on the half-contrast mask (blur-independent glyph edge)
    half = (med + float(np.median(core[sel]))) / 2
    tight = ink & (gray < half)
    ty, tx = np.nonzero(tight)
    pu = (tx - cx) * u[0] + (ty - cy) * u[1]
    pv = (tx - cx) * vv[0] + (ty - cy) * vv[1]
    u0, u1 = pu.min() - 0.5, pu.max() + 0.5
    v0, v1 = pv.min() - 0.5, pv.max() + 0.5

    # ---- erase: quadratic background fit on non-ink pixels in dilated roi
    erase = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    ring = cv2.dilate(roi.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    fitm = roi & ~erase & ~(gray < med - 20)
    fy, fx = np.nonzero(fitm)
    def basis(x, y):
        x = (x - cx) / 50.0; y = (y - cy) / 50.0
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    A = basis(fx.astype(np.float64), fy.astype(np.float64))
    ey, ex = np.nonzero(erase)
    out = img.astype(np.float32).copy()
    Be = basis(ex.astype(np.float64), ey.astype(np.float64))
    rng = np.random.default_rng(int(cx * 1000 + cy))
    for c in range(3):
        vals = img[fy, fx, c].astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
        resid = vals - A @ coef
        sd = min(np.std(resid), 4.0)
        out[ey, ex, c] = Be @ coef
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) * sd * 1.3
    out[erase] += noise[erase][:, None]
    # feather erase edge
    fe = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 0.8)
    fe = np.maximum(fe, erase.astype(np.float32)) * roi
    out = img.astype(np.float32) * (1 - fe[..., None]) + out * fe[..., None]

    # ---- render new word
    font = ImageFont.truetype(fontspec[0], 40 * SS, index=fontspec[1])
    _, _, rx0, rx1, ry0, ry1 = ink_bbox(old, font, stroke)
    sx = (u1 - u0) / ((rx1 - rx0) / SS)
    sy_meas = (v1 - v0) / ((ry1 - ry0) / SS)
    # height is only ~15px (quantised); width is reliable -> use global aspect
    sy = sx / aspect
    nim, na, nx0, nx1, ny0, ny1 = ink_bbox(new, font, stroke)
    # anchor: new ink horizontal centre -> old ink centre; new ink bottom -> old ink bottom
    ucen = (u0 + u1) / 2
    # map canvas (px,py) -> text frame (u,v) -> image
    # u = (px - ncx)/SS*sx + ucen ; v = (py - ny1)/SS*sy + v1
    ncx = (nx0 + nx1) / 2
    R = np.array([[u[0], vv[0]], [u[1], vv[1]]])
    S = np.diag([sx / SS, sy / SS])
    M = R @ S
    if align == 'left':  # wordmark is left-aligned to the box margin
        ucen, ncx = u0, nx0
    t = R @ np.array([ucen - ncx * sx / SS, v1 - ny1 * sy / SS]) + np.array([cx, cy])
    # warp to 4x canvas then area-downsample
    K = 4
    Mk = np.hstack([M * K, (t * K + (K - 1) / 2.0)[:, None]])  # pixel-centre aware
    Mk[:, 2] = t * K + (K - 1) / 2.0
    big = cv2.warpAffine(na.astype(np.float32) / 255.0, Mk, (W * K, H * K), flags=cv2.INTER_LINEAR)
    alpha = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    alpha[alpha < 0.004] = 0  # new word may extend past the old word's ROI (it is longer)
    if col_override is not None:
        col = col_override
    col = col - darken  # small text: measured core is lightened by blur
    out = out * (1 - alpha[..., None]) + col[None, None, :] * alpha[..., None]
    log.append(f"angle={np.degrees(ang):+.2f}deg extent={u1-u0:.1f}x{v1-v0:.1f}px sx={sx:.4f} sy={sy:.4f} "
               f"meas_sx/sy={sx/sy_meas:.3f} colour(BGR)={col.round(0).tolist()} grain_sd={sd:.2f}")
    touched = (fe > 0.01) | (alpha > 0.01)
    return np.clip(out, 0, 255), touched, ink, col


def main():
    img = cv2.imread(SRC)
    assert img.shape[:2] == (1080, 1080)
    cur = img.astype(np.float32)
    touched_all = np.zeros(img.shape[:2], bool)
    debug = img.copy()
    lines = []
    for name, _, tpoly, wpoly in ITEMS:
        for kind, poly in (("title", tpoly), ("wordmark", wpoly)):
            if poly is None:
                lines.append(f"{name} {kind}: not visible (occluded) - untouched")
                continue
            old, new, fs = ("FisioRest", "Nookwave", TITLE_FONT) if kind == "title" else ("artuvate", "nookwave", WORD_FONT)
            log = []
            src8 = np.clip(cur, 0, 255).astype(np.uint8)
            if kind == "title":
                res, touched, ink, tcol = process(src8, poly, old, new, fs, log, TITLE_ASPECT, sigma=TITLE_SIGMA)
            else:
                res, touched, ink, _ = process(src8, poly, old, new, fs, log, WORD_ASPECT, sigma=WORD_SIGMA, align='left', darken=WORD_DARKEN, stroke=WORD_STROKE)
            cur = np.where(touched[..., None], res, cur)
            touched_all |= touched
            debug[ink] = (0, 0, 255)
            cv2.polylines(debug, [np.int32(poly)], True, (0, 200, 0), 1)
            lines.append(f"{name} {kind}: " + log[0])
    res = np.clip(np.round(cur), 0, 255).astype(np.uint8)
    # light JPEG roundtrip on touched pixels only, so new text shares the photo's compression softness
    ok, enc = cv2.imencode(".jpg", res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rt = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    tm = cv2.dilate(touched_all.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    res[tm] = rt[tm]
    # guard: nothing outside the erased/painted text areas changed
    assert not np.any(res[~tm] != img[~tm]), "pixels outside text areas changed"
    cv2.imwrite(OUT, res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    os.makedirs(QC, exist_ok=True)
    cv2.imwrite(os.path.join(QC, "debug_detect.png"), debug)
    for name, (x0, y0, x1, y1) in QC_BOX.items():
        a = img[y0:y1, x0:x1]; b = res[y0:y1, x0:x1]
        sc = 3
        a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
        b = cv2.resize(b, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
        sep = np.full((a.shape[0], 12, 3), 255, np.uint8)
        cv2.imwrite(os.path.join(QC, f"box_{name}_before_after.jpg"), np.hstack([a, sep, b]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
