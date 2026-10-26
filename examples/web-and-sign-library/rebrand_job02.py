"""Visual rebrand FisioRest / Artuvate -> Nookwave for job-02 and job-03 (same edit).

Usage:
    python rebrand_job.py JOB_DIR [--preview N ...] [--no-render]

Writes JOB_DIR/project/intermediate/rebrand_video.mp4 (video only, same size / fps / frame count as
assets/source.mp4) and JOB_DIR/renders/preview_visual_rebrand.mp4 (with the original source audio).
All frame numbers below are in job-02 numbering; job-03 runs one frame earlier (its hook is one
frame shorter), which is detected automatically from a caption change.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import captions as C  # noqa: E402
import rbk  # noqa: E402

FFMPEG = '/opt/homebrew/bin/ffmpeg'
W, H = 720, 1280

# ----------------------------------------------------------------------------------------------
# Captions: (first, last) job-02 frames (detection decides per frame), ref frame, fill span of the
# brand word at the ref frame.
CAPTIONS = {
    'C1': dict(rng=(3106, 3130), ref=3120, span=(295, 430)),   # Artuvate
    'C2': dict(rng=(3261, 3317), ref=3290, span=(383, 517)),   # The result is FisioRest.
    'C3': dict(rng=(3417, 3464), ref=3440, span=(214, 347)),   # FisioRest combines
    'C4': dict(rng=(4320, 4381), ref=4350, span=(121, 255)),   # FisioRest is designed to be safe,
    'C5': dict(rng=(5394, 5494), ref=5450, span=(111, 245), plate=(5376, 5529)),  # ... backed by a 365-day
    'C6': dict(rng=(5580, 5654), ref=5600, span=(293, 427)),   # Right now FisioRest is in stock
    'C7': dict(rng=(5822, 5871), ref=5845, span=(196, 329)),   # FisioRest has become
}
C3_ONSET = 3420  # first frame of "FisioRest combines" in job-02 (used to detect the job offset)

# Windows of frames kept in memory during analysis (job-02 numbering).
WINDOWS = [(1640, 1650), (3106, 3130), (3263, 3317), (3417, 3464), (3498, 3585), (4320, 4381),
           (4583, 4622), (4718, 4781), (5358, 5535), (5580, 5654), (5596, 5604), (5822, 5871)]


class Store:
    def __init__(self, src, off):
        self.f = {}
        self.off = off
        cap = cv2.VideoCapture(str(src))
        n = 0
        want = set()
        for a, b in WINDOWS:
            want.update(range(a + off, b + off + 1))
        last = max(want)
        while n <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if n in want:
                self.f[n - off] = fr
            n += 1
        cap.release()

    def __call__(self, n):
        return self.f[n]

    def gray(self, n):
        return rbk.gray(self.f[n])


def detect_offset(src):
    cap = cv2.VideoCapture(str(src))
    masks = {}
    n = 0
    while n <= C3_ONSET + 8:
        ok, fr = cap.read()
        if not ok:
            break
        if n >= C3_ONSET - 8:
            masks[n] = C.fill_components(fr)[0].astype(bool)
        n += 1
    cap.release()
    ch = {k: (masks[k] ^ masks[k - 1]).sum() for k in list(masks)[1:]}
    k = max(ch, key=ch.get)
    return k - C3_ONSET


# ----------------------------------------------------------------------------------------------
# Generic helpers for re-lettering a ref plate.

def big_components(mask, min_area):
    k, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    keep = [i for i in range(1, k) if st[i, 4] >= min_area]
    return np.isin(lab, keep).astype(np.uint8) if keep else mask


def letter_geometry(mask, direction=None):
    """PCA of letter pixels: centre, unit text axis u, perpendicular v, extents."""
    ys, xs = np.where(mask > 0)
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - c).T))
    u = evec[:, 1]
    if direction is not None:
        if np.dot(u, direction) < 0:
            u = -u
    elif u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])  # points "down" relative to the text
    pu, pv = (P - c) @ u, (P - c) @ v
    return dict(c=c, u=u, v=v, umin=pu.min(), umax=pu.max(), xh=np.percentile(pv, 92) - np.percentile(pv, 8),
                vmid=(np.percentile(pv, 92) + np.percentile(pv, 8)) / 2, vbase=np.percentile(pv, 92))


def text_layer(word, geo, size_scale=1.0, max_len=None, anchor='center', tracking=0.0, stroke=0.0,
               index=0, xh_ratio=0.475, base_adj=0.0):
    """Alpha (H x W) of `word` laid along geo's axis with x-height matched to the old letters."""
    size = geo['xh'] / xh_ratio * size_scale
    a, info = rbk.text_alpha(word, size, tracking=tracking, stroke=stroke, index=index)
    L = info['w']
    if max_len and L > max_len:
        size *= max_len / L
        a, info = rbk.text_alpha(word, size, tracking=tracking, stroke=stroke, index=index)
        L = info['w']
    u, v, c = geo['u'], geo['v'], geo['c']
    vb = geo['vmid'] + info['xh'] / 2 + base_adj  # baseline so the new x-height band shares the old centre
    if anchor == 'center':
        start = c + u * (-L / 2 + (geo['umin'] + geo['umax']) / 2) + v * vb
    else:  # 'start': first glyph begins where the old first glyph began
        start = c + u * geo['umin'] + v * vb
    M = np.float32([[u[0], v[0], start[0]], [u[1], v[1], start[1]]])
    return rbk.place(a, info, H, W, M)


def inpaint_letters(img, region, rel=0.8, grow=1, radius=4, abs_delta=None, close=9, min_area=0):
    """Erase letters darker than the surrounding surface inside `region` (uint8 mask)."""
    g = rbk.gray(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    med = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, k), (0, 0), 1.0)
    if abs_delta is None:
        lt = (g < med * rel)
    else:
        lt = (g < med - abs_delta)
    lt = (lt & (region > 0)).astype(np.uint8)
    if min_area:
        k_, lab_, st_, _ = cv2.connectedComponentsWithStats(lt)
        lt = np.isin(lab_, [i for i in range(1, k_) if st_[i, 4] >= min_area]).astype(np.uint8)
    lt = cv2.dilate(lt, np.ones((3, 3), np.uint8), iterations=grow)
    out = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), lt * 255, radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32), lt


def relit_replace(F, Nw, Ow, Mw, Sw, sigma=6.0):
    """Within mask Mw replace F by the warped new plate, relit by the local F/Ow ratio over Sw."""
    Ff = F.astype(np.float32)
    num = cv2.GaussianBlur(Ff * Sw[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(Ow * Sw[..., None], (0, 0), sigma)
    cov = cv2.GaussianBlur(Sw, (0, 0), sigma)[..., None]
    L = np.where(cov > 0.05, num / np.maximum(den, 1e-3), 1.0)
    L = np.clip(L, 0.3, 3.0)
    rep = np.clip(Nw * L, 0, 255)
    m = Mw[..., None]
    return Ff * (1 - m) + rep * m


def local_scale(w, c, d=10.0):
    p = np.float64([[c[0], c[1]], [c[0] + d, c[1]], [c[0], c[1] + d]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(p, np.asarray(w, np.float64)).reshape(-1, 2)
    a, b = q[1] - q[0], q[2] - q[0]
    return float(np.sqrt(abs(a[0] * b[1] - a[1] * b[0])) / d)


class PlateItem:
    """A logo re-lettered once on a ref frame and carried to other frames by a tracked warp."""

    def __init__(self, name, ref, new, old, letter_mask, surface_mask, warps):
        self.name, self.ref = name, ref
        self.new, self.old = new.astype(np.float32), old.astype(np.float32)
        self.lm = rbk.feather(cv2.dilate(letter_mask, np.ones((5, 5), np.uint8)), 1.2)
        self.sm = surface_mask.astype(np.float32)
        self.warps = warps  # {n: 3x3}
        ys, xs = np.where(letter_mask > 0)
        self.center = (float(xs.mean()), float(ys.mean()))

    def apply(self, n, F):
        w = self.warps.get(n)
        if w is None:
            return F, False
        wp = lambda im, interp=cv2.INTER_LINEAR: cv2.warpPerspective(im, w, (W, H), flags=interp)
        s = local_scale(w, self.center)
        pre = (lambda im: cv2.GaussianBlur(im, (0, 0), 0.5 / s - 0.5)) if s < 0.95 else (lambda im: im)
        Nw, Ow = wp(pre(self.new)), wp(pre(self.old))
        Mw, Sw = wp(self.lm), wp(self.sm)
        out = relit_replace(F, Nw, Ow, Mw, Sw)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), True


# ----------------------------------------------------------------------------------------------
# Item builders

def tag_seg(f, center, hue=(3, 24), smin=55, vmin=50, maxd=60):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (hue[0], smin, vmin), (hue[1], 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best, bd = None, 1e9
    for i in range(1, k):
        if st[i, 4] < 60:
            continue
        d = np.hypot(*(cen[i] - center))
        if d < bd:
            bd, best = d, i
    if best is None or bd > maxd:
        return None, None
    blob = (lab == best).astype(np.uint8) * 255
    hull = cv2.convexHull(cv2.findNonZero(blob))
    blob = np.zeros_like(blob)
    cv2.fillConvexPoly(blob, hull, 255)
    return blob, cen[best]


def chained_tag_warps(S, R, first, last, center, **segkw):
    centers = {R: np.array(center, float)}

    def mask_fn(f, n):
        near = min(centers, key=lambda k: abs(k - n))
        blob, c = tag_seg(f, centers[near], **segkw)
        if blob is None:
            return None
        centers[n] = c
        return cv2.dilate(blob, np.ones((5, 5), np.uint8))

    tr = rbk.track_chained(S, mask_fn, R, first, last, motion=cv2.MOTION_HOMOGRAPHY, margin=40, min_cc=0.6)
    return {n: w for n, (c, w) in tr.items()}


def build_tag(S, name, R, first, last, center, direction=None, ink_k=0.5, deboss=True, rel=0.88,
              erode=3, min_area=20, geo_area=60, max_len_frac=0.86, size_scale=1.0, segkw=None, warps=None, tracking=0.0,
              base_adj=0.0, anchor='center', center_on_interior=False):
    segkw = segkw or {}
    ref = S(R).astype(np.float32)
    blob, _ = tag_seg(S(R), np.array(center, float), **segkw)
    interior = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    clean, lt = inpaint_letters(ref, interior, rel=rel, min_area=min_area)
    geo = letter_geometry(big_components(lt, geo_area), direction)
    # interior length along the text axis limits the new word length
    ys, xs = np.where(interior > 0)
    pu = (np.stack([xs, ys], 1) - geo['c']) @ geo['u']
    if center_on_interior:
        geo['umin'], geo['umax'] = float(pu.min()), float(pu.max())
    A = text_layer('nookwave', geo, size_scale=size_scale,
                   max_len=None if max_len_frac is None else (pu.max() - pu.min()) * max_len_frac,
                   tracking=tracking, base_adj=base_adj, anchor=anchor)
    A = cv2.GaussianBlur(A, (0, 0), 0.45) * (interior > 0)
    g, gc = rbk.gray(ref), rbk.gray(clean)
    if ink_k is None:
        ink_k = float(np.median(g[lt > 0]) / max(1, np.median(gc[lt > 0])))
        ink_k = float(np.clip(ink_k, 0.35, 0.85))
    new = clean * (1 - A * (1 - ink_k))[..., None]
    if deboss:  # faint highlight on the lower edge of the debossed strokes
        hl = np.clip(cv2.warpAffine(A, np.float32([[1, 0, 0], [0, 1, 1]]), (W, H)) - A, 0, 1)
        new = np.clip(new * (1 + 0.18 * hl)[..., None], 0, 255)
    if warps is None:
        warps = chained_tag_warps(S, R, first, last, center, **segkw)
    lm = ((lt > 0) | (A > 0.05)).astype(np.uint8)
    item = PlateItem(name, R, new, ref, lm, interior > 0, warps)
    item.debug = dict(ink_k=ink_k, xh=geo['xh'], frames=sorted(warps))
    return item


def build_v3(S):
    R = 3540
    ref = S(R).astype(np.float32)
    g = rbk.gray(ref)
    # light plate (beige) on dark device
    roi = np.zeros(g.shape, np.uint8)
    roi[656:700, 315:445] = 1
    sat = cv2.cvtColor(S(R), cv2.COLOR_BGR2HSV)[..., 1]
    plate = ((g > 170) & (sat < 75) & (roi > 0)).astype(np.uint8)
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(plate)
    i = 1 + int(np.argmax(st[1:, 4]))
    plate = (lab == i).astype(np.uint8) * 255
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    plate = cv2.morphologyEx(plate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    interior = cv2.erode(plate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9)))
    clean, lt = inpaint_letters(ref, interior, rel=0.86, grow=2, radius=3, min_area=6)
    geo = letter_geometry(big_components(lt, 40))
    A = text_layer('nookwave', geo, size_scale=0.92, max_len=(geo['umax'] - geo['umin']) * 0.98, tracking=0.3)
    A = cv2.GaussianBlur(A, (0, 0), 0.55)
    ink = 0.42
    new = clean * (1 - A * (1 - ink))[..., None]
    hl = np.clip(cv2.warpAffine(A, np.float32([[1, 0, 0.6], [0, 1, 1]]), (W, H)) - A, 0, 1)
    new = np.clip(new * (1 + 0.12 * hl)[..., None], 0, 255)
    tr = rbk.track_sequence(S.gray, R, 3499, 3584, (300, 640, 470, 715), cv2.MOTION_AFFINE, min_cc=0.75, margin=50)
    warps = {n: rbk._to3(w) for n, (c, w) in tr.items()}
    item = PlateItem('V3', R, new, ref, ((lt > 0) | (A > 0.05)).astype(np.uint8), plate > 0, warps)
    item.debug = dict(ink=ink, xh=geo['xh'], frames=sorted(warps))
    return item


def build_v5(S):
    R = 4587
    ref = S(R).astype(np.float32)
    g = rbk.gray(ref)
    region = np.zeros(g.shape, np.uint8)
    region[790:960, 600:720] = 1
    wl = cv2.GaussianBlur(cv2.dilate(g, np.ones((31, 31), np.uint8)), (0, 0), 6)
    lt = (((wl - g) > 45) & (region > 0)).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(lt)
    keep = [i for i in range(1, k) if st[i, 4] > 80]
    lt = np.isin(lab, keep).astype(np.uint8)
    hole = cv2.dilate(lt, np.ones((3, 3), np.uint8), iterations=2)
    clean = cv2.inpaint(S(R), hole * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
    # geometry from the 'a' alone (the only complete glyph); slope measured from the a->r baseline
    k2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(lt)
    ia = min(range(1, k2), key=lambda i: st2[i, 0] if st2[i, 4] > 300 else 1e9)
    ang = np.radians(18.0)
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-u[1], u[0]])
    ys, xs = np.where(lab2 == ia)
    P = np.stack([xs, ys], 1) - cen2[ia]
    pu, pv = P @ u, P @ v
    geo = dict(c=cen2[ia], u=u, v=v, umin=pu.min(), umax=pu.max(), xh=pv.max() - pv.min(),
               vmid=(pv.max() + pv.min()) / 2, vbase=pv.max())
    A = text_layer('nookwave', geo, anchor='start', tracking=-0.5, xh_ratio=0.47, size_scale=0.82, stroke=-0.4, base_adj=3)
    A = cv2.GaussianBlur(A, (0, 0), 0.7)
    ink = np.percentile(ref[lt > 0], 10, axis=0)
    new = clean * (1 - A[..., None]) + ink * A[..., None]
    tr = rbk.track_sequence(S.gray, R, R, 4620, (600, 800, 720, 960), cv2.MOTION_AFFINE, min_cc=0.75, margin=50)
    warps = {n: rbk._to3(w) for n, (c, w) in tr.items()}
    surf = (g > 160) & (region > 0)
    item = PlateItem('V5', R, new, ref, ((hole > 0) | (A > 0.05)).astype(np.uint8), surf, warps)
    item.debug = dict(xh=geo['xh'], angle=float(np.degrees(np.arctan2(geo['u'][1], geo['u'][0]))),
                      frames=sorted(warps))
    return item


def build_v6(S):
    R = 4760
    tr = rbk.track_sequence(S.gray, R, 4724, 4779, (25, 625, 115, 790), cv2.MOTION_AFFINE, min_cc=0.75, margin=50)
    warps = {n: rbk._to3(w) for n, (c, w) in tr.items()}
    blob, _ = tag_seg(S(R), np.array([68., 705.]), hue=(2, 22), smin=45, vmin=120)
    return build_tag(S, 'V6', R, 0, 0, (68, 705), direction=np.array([0., 1.]), rel=0.93, erode=6,
                     center_on_interior=True,
                     segkw=dict(hue=(2, 22), smin=45, vmin=120), warps=warps, max_len_frac=0.74,
                     ink_k=0.72, size_scale=0.9)


class SignItem:
    """V1: 'artuvate' sign on the building photo inset (pop-in scale + fade, then slow zoom)."""
    R = 3122

    def __init__(self, S):
        ref = S(self.R).astype(np.float32)
        clean = ref.copy()
        ya, yb, xa, xb = 512, 563, 200, 400
        t = (np.arange(ya, yb + 1) - ya) / (yb - ya)
        top = np.median(ref[ya - 1:ya + 3, xa:xb], axis=0)
        bot = np.median(ref[yb - 3:yb + 1, xa:xb], axis=0)
        clean[ya:yb + 1, xa:xb] = top[None] * (1 - t)[:, None, None] + bot[None] * t[:, None, None]
        a, info = rbk.text_alpha('nookwave', 40, tracking=-1.2)
        A = rbk.place(a, info, H, W, rbk.similarity(1, 0, (216 + 368) / 2 - info['w'] / 2, 549.5))
        A = cv2.GaussianBlur(A, (0, 0), 0.9)
        sh = cv2.GaussianBlur(cv2.warpAffine(A, np.float32([[1, 0, 2], [0, 1, 3]]), (W, H)), (0, 0), 2.0) * 0.45
        new = clean * (1 - sh)[..., None]
        new = new * (1 - A)[..., None] + np.float32([18, 20, 18]) * A[..., None]
        self.ref, self.rg, self.D = ref, rbk.gray(ref), new - ref
        box = np.zeros((H, W), np.float32)
        box[506:570, 194:396] = 1
        self.box = rbk.feather(box, 2)
        self.warps = {}

    def track(self, S, frames):
        self.warps = rbk.track_planar_seq(self.rg, S.gray, frames, (199, 509, 386, 566),
                                          np.arange(0.60, 1.08, 0.005), (40, 380, 680, 760), 0.55)

    def apply(self, n, F):
        if n not in self.warps:
            return F, False
        v, w = self.warps[n]
        Ow = rbk.warp_img(self.rg, w)
        Dw = rbk.warp_img(self.D, w)
        Bw = cv2.warpAffine(self.box, w, (W, H))
        mm = Bw > 0.5
        a = rbk.hp(Ow)[mm]
        b = rbk.hp(rbk.gray(F))[mm]
        gain = float(np.clip((a * b).sum() / max(1e-6, (a * a).sum()), 0, 1.0))
        out = F.astype(np.float32) + gain * Dw * Bw[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), True


class WebItem:
    """V7: 'Fisiorest' in the guarantee-page body copy (card scales in, then holds)."""
    R = 5450

    def __init__(self, S):
        ref = S(self.R).astype(np.float32)
        g = rbk.gray(ref)
        ya, yb, wa, wb = 700, 728, 352, 413
        ink = np.percentile(ref[ya:yb, 176:544][g[ya:yb, 176:544] < 200], 3, axis=0)
        a, info = rbk.text_alpha('Nookwave', 17.2)
        d = info['w'] - (wb - wa + 1)
        ls, rs = -(d // 2), d - d // 2
        new = ref.copy()
        new[ya:yb, 150:570] = np.float32([254, 254, 254])
        new[ya:yb, 170 + ls:wa - 3 + ls] = ref[ya:yb, 170:wa - 3]
        new[ya:yb, wb + 3 + rs:560 + rs] = ref[ya:yb, wb + 3:560]
        A = rbk.place(a, info, H, W, rbk.similarity(1, 0, wa + ls, 721))
        A = cv2.GaussianBlur(A, (0, 0), 0.45)
        new = new * (1 - A[..., None]) + ink * A[..., None]
        self.ref, self.rg, self.new = ref, g, new
        box = np.zeros((H, W), np.float32)
        box[699:730, 152:568] = 1
        self.box = rbk.feather(box, 1.0)
        self.warps = {}

    def track(self, S, frames):
        self.warps = rbk.track_planar_seq(self.rg, S.gray, frames, (150, 600, 575, 760),
                                          np.arange(0.40, 1.06, 0.01), (0, 300, 720, 1000), 0.6)

    def apply(self, n, F):
        if n not in self.warps:
            return F, False
        v, w = self.warps[n]
        Ow = rbk.warp_img(self.ref, w)
        Nw = rbk.warp_img(self.new, w)
        Bw = cv2.warpAffine(self.box, w, (W, H))
        mm = cv2.warpAffine(self.box, w, (W, H)) > 0.5
        Ff = F.astype(np.float32)
        out = Ff.copy()
        for ch in range(3):
            x, y = Ow[..., ch][mm], Ff[..., ch][mm]
            gch, bch = np.polyfit(x, y, 1)
            out[..., ch] = Ff[..., ch] * (1 - Bw) + (gch * Nw[..., ch] + bch) * Bw
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), True


# ---- V8: big print on the falling dust bag ----------------------------------------------------
V8_REF = 5595
V8_BOX = (180, 655, 480, 758)
V8_C = np.array([330., 706.])
# Hand-placed keyframes (centre x, y, angle deg, scale vs ref) for the flip/fall, refined per frame.
V8_KEYS = {5601: (335, 726, 0, 1.03), 5602: (345, 750, 3, 1.03), 5603: (360, 765, 4, 1.0),
           5604: (330, 810, 4, 1.0), 5605: (370, 840, 5, 1.0), 5606: (390, 860, 4, 1.0),
           5607: (410, 900, -8, 0.98), 5608: (440, 920, -10, 0.98), 5609: (465, 960, -12, 0.97),
           5610: (495, 974, -15, 0.92), 5611: (525, 986, -22, 0.88), 5612: (545, 994, -28, 0.85),
           5613: (565, 1010, -38, 0.82), 5614: (585, 1020, -48, 0.8), 5615: (605, 1030, -58, 0.78),
           5616: (615, 1040, -65, 0.76), 5617: (644, 1060, -75, 0.75), 5618: (650, 1080, -80, 0.75),
           5619: (670, 1095, -85, 0.74), 5620: (690, 1110, -88, 0.74), 5621: (700, 1120, -90, 0.74),
           5622: (710, 1130, -90, 0.74)}


def bag_dark(f):
    g = rbk.gray(f)
    wl = cv2.GaussianBlur(cv2.dilate(g, np.ones((31, 31), np.uint8)), (0, 0), 6)
    d = np.clip(wl - g, 0, 255)
    d[wl < 170] = 0
    return d, wl


def sim3(cx, cy, th, s):
    c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
    A = s * np.array([[c, -sn], [sn, c]])
    t = np.array([cx, cy]) - A @ V8_C
    return np.vstack([np.hstack([A, t[:, None]]), [0, 0, 1]])


class BagItem:
    def __init__(self, S):
        ref = S(V8_REF)
        Dr, wl = bag_dark(ref)
        x0, y0, x1, y1 = V8_BOX
        T = np.zeros_like(Dr)
        T[y0:y1, x0:x1] = Dr[y0:y1, x0:x1]
        T[T < 20] = 0
        self.T = T
        lt = (T > 30).astype(np.uint8)
        geo = letter_geometry(lt)
        self.A = cv2.GaussianBlur(text_layer('nookwave', geo, tracking=-0.8, stroke=-0.3,
                                             max_len=(geo['umax'] - geo['umin']) * 1.0), (0, 0), 0.8)
        self.ink_dark = float(np.percentile(T[lt > 0], 90))  # darkness of the ink below the fabric white
        self.ref_white = float(np.median(wl[y0:y1, x0:x1]))
        self.poses = {}  # n: (3x3 warp, blur vector or None)
        # 1) homography track while the bag is held up
        tr = rbk.track_sequence(S.gray, V8_REF, 5585, 5600, (175, 650, 485, 760), cv2.MOTION_HOMOGRAPHY,
                                min_cc=0.85, margin=80)
        for n, (c, w) in tr.items():
            self.poses[n] = (rbk._to3(w), None)
        # 2) flip / fall: refine the hand-placed keyframes against the print's darkness
        prev = None
        for n in sorted(V8_KEYS):
            self.poses[n] = self._refine(S, n, V8_KEYS[n], prev)
            prev = self.poses[n][2]

    def _refine(self, S, n, key, prev):
        Dn, _ = bag_dark(S(n))
        cx, cy, th, s = key
        x0, y0, x1, y1 = V8_BOX
        valid = np.ones(Dn.shape, bool)
        valid[758:822] = False
        box = np.zeros(Dn.shape, np.uint8)
        quad = np.float32([[[x0 - 40, y0 - 40], [x1 + 40, y0 - 40], [x1 + 40, y1 + 40], [x0 - 40, y1 + 40]]])
        cv2.fillConvexPoly(box, cv2.perspectiveTransform(quad, sim3(cx, cy, th, s))[0].astype(np.int32), 1)
        valid &= box > 0
        vel = None if prev is None else ((cx - prev[0]) * 0.5, (cy - prev[1]) * 0.5)

        def score(m, bl):
            t = cv2.warpPerspective(self.T, m, (W, H))
            if bl is not None:
                t = rbk.motion_blur(t, *bl)
            a, b = t[valid], Dn[valid]
            if a.sum() < 1:
                return -1
            a, b = a - a.mean(), b - b.mean()
            return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-6))

        best = (-2, cx, cy, th, s, None)
        for it in range(3):
            st, sa, ss = (12, 5, 2)[it], (5, 2, 1)[it], (0.06, 0.03, 0.015)[it]
            for dx in (-st, 0, st):
                for dy in (-st, 0, st):
                    for da in (-sa, 0, sa):
                        for ds in (-ss, 0, ss):
                            for bl in ([None, vel] if vel is not None else [None]):
                                v = score(sim3(cx + dx, cy + dy, th + da, s + ds), bl)
                                if v > best[0]:
                                    best = (v, cx + dx, cy + dy, th + da, s + ds, bl)
            _, cx, cy, th, s, _ = best
        v, cx, cy, th, s, bl = best
        return sim3(cx, cy, th, s), bl, (cx, cy)

    def apply(self, n, F, cap_region=None):
        if n not in self.poses:
            return F, False
        w, bl = self.poses[n][:2]
        Tw = cv2.warpPerspective(self.T, w, (W, H))
        Aw = cv2.warpPerspective(self.A, w, (W, H))
        if bl is not None:
            Tw = rbk.motion_blur(Tw, *bl)
            Aw = rbk.motion_blur(Aw, *bl)
        Dn, wl = bag_dark(F)
        L = 10 + (0 if bl is None else int(1.5 * np.hypot(*bl)) + 8)
        zone = cv2.dilate(((Tw > 6) | (Aw > 0.02)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L)))
        hole = ((Dn > 10) & (cv2.dilate((Tw > 6).astype(np.uint8), np.ones((L, L), np.uint8)) > 0)).astype(np.uint8)
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8), iterations=2)
        capm = None
        if cap_region is not None:
            capm = np.zeros((H, W), np.float32)
            capm[C.BAND[0]:C.BAND[1]] = cap_region
            hole = np.maximum(hole, (capm > 0.02).astype(np.uint8) * (zone > 0))
        if not hole.any():
            return F, False
        clean = cv2.inpaint(F, hole * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
        # fabric white level near the print scales the ink
        white = float(np.median(wl[zone > 0])) if (zone > 0).any() else self.ref_white
        ink = np.clip(white - self.ink_dark * white / self.ref_white, 0, 255)
        out = clean * (1 - Aw[..., None]) + ink * Aw[..., None]
        out = F.astype(np.float32) * (1 - zone[..., None]) + out * zone[..., None]
        if capm is not None:  # keep the original caption pixels on top
            out = out * (1 - capm[..., None]) + F.astype(np.float32) * capm[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), True


# ----------------------------------------------------------------------------------------------

def build_all(S):
    items = {}
    print('captions...', flush=True)
    frames = {n: S(n) for n in S.f}
    lib = C.harvest_glyphs(frames)
    caps = {}
    for k, cfg in CAPTIONS.items():
        cs = C.CaptionSwap(cfg['span'], S(cfg['ref']), lib)
        if 'plate' in cfg:  # static card: clean plate of the caption band from uncovered pixels
            a, b = cfg['plate']
            acc, cnt = 0, 0
            for n in range(a, b + 1):
                band = S(n)[C.BAND[0]:C.BAND[1]].astype(np.float32)
                mask, _ = C.fill_components(S(n))
                free = (C.word_region(mask) < 0.01).astype(np.float32)
                acc = acc + band * free[..., None]
                cnt = cnt + free
            valid = cnt > 3
            plate = acc / np.maximum(cnt, 1)[..., None]
            cs.plate = (plate, valid)
        caps[k] = cs
    items['captions'] = caps
    print('V1 sign...', flush=True)
    v1 = SignItem(S)
    v1.track(S, range(3108, 3129))
    items['V1'] = v1
    print('V2 tag...', flush=True)
    items['V2'] = build_tag(S, 'V2', 3305, 3266, 3315, (535, 650))
    print('V3 plate...', flush=True)
    items['V3'] = build_v3(S)
    print('V4 tag...', flush=True)
    items['V4'] = build_tag(S, 'V4', 4377, 4360, 4379, (620, 608), max_len_frac=None, anchor='start', size_scale=0.52)
    print('V5 bag edge...', flush=True)
    items['V5'] = build_v5(S)
    print('V6 tag...', flush=True)
    items['V6'] = build_v6(S)
    print('V7 web copy...', flush=True)
    v7 = WebItem(S)
    v7.track(S, range(5358, 5535))
    items['V7'] = v7
    print('V8 bag print...', flush=True)
    items['V8'] = BagItem(S)
    return items


ORDER = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']


def process(n, F, items, log=None):
    """n in job-02 numbering."""
    caps = items['captions']
    for k in ORDER:
        F, ok = items[k].apply(n, F)
        if ok and log is not None:
            log.setdefault(k, []).append(n)
    if 'V8' in items:
        reg = caps['C6'].region(F) if CAPTIONS['C6']['rng'][0] <= n <= CAPTIONS['C6']['rng'][1] else None
        F, ok = items['V8'].apply(n, F, reg)
        if ok and log is not None:
            log.setdefault('V8', []).append(n)
    for k, cfg in CAPTIONS.items():
        a, b = cfg['rng']
        if a <= n <= b:
            F, ok = caps[k].apply(F)
            if ok and log is not None:
                log.setdefault(k, []).append(n)
    return F


def main():
    job = Path(sys.argv[1]).resolve()
    src = job / 'assets' / 'source.mp4'
    off = detect_offset(src)
    print('frame offset vs job-02 numbering:', off, flush=True)
    S = Store(src, off)
    items = build_all(S)
    if '--preview' in sys.argv:
        out = job / 'analysis' / 'visual_qc' / 'preview'
        out.mkdir(parents=True, exist_ok=True)
        for n in map(int, sys.argv[sys.argv.index('--preview') + 1:]):
            f = S(n)
            cv2.imwrite(str(out / f'p{n}.png'), np.hstack([f, process(n, f.copy(), items)]))
        return
    inter = job / 'project' / 'intermediate'
    inter.mkdir(parents=True, exist_ok=True)
    vid = inter / 'rebrand_video.mp4'
    cap = cv2.VideoCapture(str(src))
    ff = subprocess.Popen([FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                           '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                           '-pix_fmt', 'yuv420p', '-r', '30', str(vid)], stdin=subprocess.PIPE)
    log = {}
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m = n - off  # job-02 numbering
        g = process(m, f.copy(), items, log) if 3100 <= m <= 5880 else f
        ff.stdin.write(g.tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    print('frames written', n)
    (job / 'analysis' / 'visual_qc').mkdir(parents=True, exist_ok=True)
    (job / 'analysis' / 'visual_qc' / 'applied_frames.json').write_text(
        json.dumps({'offset_vs_job02': off, 'frames_job02_numbering': log, 'total_frames': n}, indent=1))
    rend = job / 'renders'
    rend.mkdir(exist_ok=True)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(vid), '-i', str(src), '-map', '0:v', '-map', '1:a?',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
                    str(rend / 'preview_visual_rebrand.mp4')], check=True)


if __name__ == '__main__':
    main()
