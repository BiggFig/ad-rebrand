"""job-05 visual rebrand: FisioRest / Artuvate -> Nookwave.

Usage:
  rebrand_job.py                 full run -> project/intermediate/rebrand_video.mp4 (+ preview mux)
  rebrand_job.py --qc            also write analysis/visual_qc before/after crops
  rebrand_job.py --frames 120 250 ...   only write before/after previews of those frames (fast check)

Items
  C1  caption f95-138 "who's ever bought FisioRest" -> "who's ever bought Nookwave"
      rebuilt from the video's own caption glyphs (caption_lib), re-centred.
  V1  metal badge on CGI pillow f202-323: ECC-tracked (affine, smoothed), "artuvate" erased and
      "nookwave" re-engraved in fitted Futura on the same arched baseline, ink relit per frame.
  V2  vertical leather tag f429-491: same approach (text runs top->bottom).
  V3  print on dust bag f923-986: word pose fitted per frame (similarity + x-stretch + bend),
      motion-blur kernel fitted per frame, caption "Right now you can get 55% off," kept on top.
Source is decoded sequentially only (no seeking).
"""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from caption_lib import Caption, segment_minmax, glyph_alpha, outline_alpha, BAND  # noqa: E402
from text_lib import canonical, render_layer, fit_text  # noqa: E402
from track_lib import ecc_track  # noqa: E402

JOB = HERE.parent
SRC = JOB / 'assets' / 'source.mp4'
OUT = HERE / 'intermediate' / 'rebrand_video.mp4'
PREVIEW = JOB / 'renders' / 'preview_visual_rebrand.mp4'
QC = JOB / 'analysis' / 'visual_qc'
CACHE = HERE / 'intermediate' / 'analysis_cache.pkl'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FUT = ('/System/Library/Fonts/Supplemental/Futura.ttc', 0)
OLD, NEW = 'artuvate', 'nookwave'

CAP = (95, 138)
V1 = (202, 323)
V2 = (429, 491)
V3 = (923, 986)
CAP3 = (923, 986)          # caption segment over the bag shot ("Right now you can get 55% off,")


# ----------------------------------------------------------------------------- utilities
def decode(ranges):
    """Sequential decode; returns {n: frame} for n inside any range, and total frame count."""
    cap = cv2.VideoCapture(str(SRC))
    keep, n = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if any(a <= n <= b for a, b in ranges):
            keep[n] = f
        n += 1
    return keep, n


def darkness(gray, k):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return np.clip(bg - gray, 0, None)


def kernel_bank():
    """Blur kernels: none, isotropic, and motion lines (length x angle), all lightly softened."""
    bank = [None]
    for s in (0.6, 1.0, 1.6, 2.4):
        bank.append(('g', s))
    for L in (3, 5, 7, 9, 12, 15, 19, 24, 30):
        for ang in range(0, 180, 15):
            bank.append(('m', L, ang))
    return bank


def apply_kernel(a, k):
    if k is None:
        return a
    if k[0] == 'g':
        return cv2.GaussianBlur(a, (0, 0), k[1])
    _, L, ang = k
    sz = L + 4
    ker = np.zeros((sz, sz), np.float32)
    c = sz / 2.0
    dx, dy = np.cos(np.radians(ang)) * L / 2, np.sin(np.radians(ang)) * L / 2
    cv2.line(ker, (int(round((c - dx) * 8)), int(round((c - dy) * 8))),
             (int(round((c + dx) * 8)), int(round((c + dy) * 8))), 1.0, 1, cv2.LINE_AA, shift=3)
    ker = cv2.GaussianBlur(ker, (0, 0), 0.6)
    ker /= ker.sum()
    return cv2.filter2D(a, -1, ker)


BANK = kernel_bank()


def const_ink(rec):
    """One ink strength per item (median over frames whose lettering is sharp): the print doesn't change,
    lighting is already carried by the multiplicative model and blur by the kernel - avoids flicker."""
    ns = sorted(rec)
    sharp = [rec[n]['c'] for n in ns if rec[n]['k'] is None or rec[n]['k'][0] == 'g']
    c = np.median(np.array(sharp if sharp else [rec[n]['c'] for n in ns]), axis=0)
    return [c for _ in ns]


def smooth_series(vals, w=2):
    vals = np.asarray(vals, np.float64)
    out = vals.copy()
    for i in range(len(vals)):
        out[i] = np.median(vals[max(0, i - w):i + w + 1], axis=0)
    return out


# ----------------------------------------------------------------------------- relabel core
def orange_tag_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] >= 4) & (hsv[..., 0] <= 24) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 95)).astype(np.uint8)


def clean_plate(crop, A_old, protect, grow, surface=None):
    """Remove the old word.  Default: Telea inpaint (plus any caption pixels touching it, so they aren't
    used as source).  With `surface` (a mask of the material the word is printed on, e.g. the narrow
    leather tag), fill by normalised convolution from that material only, so the dark leather around a
    narrow tag never bleeds in."""
    m = (A_old > 0.06).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    if protect is not None:
        near = cv2.dilate(m, np.ones((15, 15), np.uint8))
        m = m | ((protect > 0.02).astype(np.uint8) & near)
    tel = cv2.inpaint(crop, m * 255, 4, cv2.INPAINT_TELEA)
    if surface is None:
        return tel, m
    surf = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    w = (surf & (1 - m)).astype(np.float32)
    out = tel.astype(np.float32)
    acc = np.zeros(crop.shape[:2], np.float32)
    fill = np.zeros_like(out)
    for sg in (2.0, 4.0, 8.0):      # small kernel first, wider where the tag gives no support nearby
        wb = cv2.GaussianBlur(w, (0, 0), sg)
        vb = cv2.GaussianBlur(crop.astype(np.float32) * w[..., None], (0, 0), sg)
        ok = (wb > 0.08) & (acc == 0)
        fill[ok] = vb[ok] / wb[ok][:, None]
        acc[ok] = 1
    tgt = (m > 0) & (acc > 0) & (cv2.dilate(surf, np.ones((3, 3), np.uint8)) > 0)
    out[tgt] = fill[tgt]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), m


def _sharp_sig(D, region):
    gx = np.abs(cv2.Sobel(D, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(D, cv2.CV_32F, 0, 1, ksize=3))
    m = max(float(D[region].mean()), 1e-3)
    return np.log(np.array([gx[region].mean(), gy[region].mean()]) / m + 1e-4)


def fit_blur_ink(crop, clean, A_old, valid, kernels=None):
    """Blur kernel chosen so the model's edge sharpness (x and y separately) matches the observed
    lettering (robust to small glyph-shape mismatch, which a plain least-squares fit hides with blur);
    then per-channel ink strength c by least squares:  crop ≈ clean * (1 - c * K*A_old)."""
    cf, of = clean.astype(np.float32), crop.astype(np.float32)
    Dg = (cf - of).mean(2)
    cg = cf.mean(2)
    region = (cv2.dilate((A_old > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & valid
    target = _sharp_sig(np.clip(Dg, 0, None), region)
    best = None
    for k in (kernels or BANK):
        X = cg * apply_kernel(A_old, k)
        if X[region].sum() < 1e-3:
            continue
        r = float(((_sharp_sig(X, region) - target) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, k)
    k = best[1]
    KA = apply_kernel(A_old, k)
    core = (KA > 0.6 * KA[region].max()) & valid
    if core.sum() < 10:
        core = (KA > 0.3 * KA[region].max()) & valid
    cs = []
    for ch in range(3):   # ink strength measured on the letter cores (a global LS fit is diluted by misfit)
        r = (cf[..., ch] - of[..., ch])[core] / np.maximum(cf[..., ch][core] * KA[core], 1.0)
        cs.append(float(np.median(r)))
    return k, np.clip(np.array(cs), 0, 0.92)


def composite_new(clean, A_new, k, c):
    KA = apply_kernel(A_new, k)[..., None]
    out = clean.astype(np.float32) * (1 - c[None, None, :] * KA)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- per-item models
def fit_word(gray_crop, weight, init, starts, free=('size', 'cx', 'cy', 'angle', 'bend', 'track'), k=7):
    D = darkness(gray_crop, k)
    best = None
    for st in starts:
        p0 = dict(init, **st)
        b = fit_text(D, weight, OLD, [FUT], p0, free=free)
        if best is None or b[1] < best[1]:
            best = b
    return best[0], -best[1]


def match_new_width(p, allow=1.04):
    """Same pose for 'nookwave'.  It sets wider than 'artuvate' in the same face, so let it run up to
    `allow` x wider, then condense horizontally (sx) rather than crushing the letter-spacing."""
    _, (_, _, hw_old) = canonical(OLD, p['font'], p['size'], p.get('track', 0), p.get('stroke', 0))
    _, (_, _, hw_new) = canonical(NEW, p['font'], p['size'], p.get('track', 0), p.get('stroke', 0))
    q = dict(p)
    if hw_new > hw_old * allow:
        q['sx'] = p.get('sx', 1.0) * hw_old * allow / hw_new
    p['halfw'] = hw_old
    q['halfw'] = hw_old * p.get('sx', 1.0) / q.get('sx', 1.0)   # same arch over the same span
    return q


def affine_smooth(tr, frames, deg=3, fixed=()):
    """Polynomial-in-time smoothing of the 2x3 affine part (CGI camera moves are smooth)."""
    ns = np.array([n for n in frames if n not in fixed])
    A = np.array([tr[n][0][:2].ravel() for n in ns])
    w = np.array([max(tr[n][1], 0.05) for n in ns])
    t = (ns - ns.mean()) / max(1, ns.std())
    V = np.vander(t, deg + 1)
    coef = np.linalg.lstsq(V * w[:, None], A * w[:, None], rcond=None)[0]
    out = {}
    for n in frames:
        if n in fixed:
            out[n] = tr[n][0]
            continue
        tt = (n - ns.mean()) / max(1, ns.std())
        a = (np.vander([tt], deg + 1) @ coef).reshape(2, 3)
        out[n] = np.vstack([a, [0, 0, 1]])
    return out


class Badge:
    """CGI badge/tag: ECC-tracked plane; word fitted once on the reference frame."""

    def __init__(self, name, frames, rng, ref, roi, crop, weight_box, init, starts, vt_init=None,
                 fixed=(), grow=2, dark_k=7, similarity=False, alt=None, skip=(), surface=None):
        self.name, self.rng, self.grow = name, rng, grow
        self.whip = set(fixed)
        self.ref = ref
        self.skip = set(skip)
        self.surface = surface
        x0, y0, x1, y1 = crop
        self.crop = crop
        fr = {n: frames[n] for n in range(rng[0], rng[1] + 1)}
        tr = ecc_track(fr, ref, roi, init=vt_init, alt=alt)
        if similarity:   # narrow template: shear is unconstrained, keep scale+rotation+translation only
            c = np.array([(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2, 1.0])
            for n in list(tr):
                H, cc = tr[n]
                A = H[:2, :2]
                sc = np.sqrt(abs(np.linalg.det(A)))
                th = np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])
                Rm = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                pt = (H @ c)[:2]
                if vt_init is not None and n in vt_init and n in fixed:
                    p0 = (vt_init[n] @ c)[:2]
                    if np.linalg.norm(pt - p0) > 12 or abs(sc - vt_init[n][0, 0]) > 0.15:
                        Rm, pt = vt_init[n][:2, :2], p0      # ECC lost on the whip-blurred frames
                Hs = np.eye(3)
                Hs[:2, :2] = Rm
                Hs[:2, 2] = pt - Rm @ c[:2]
                tr[n] = (Hs, cc)
        self.cc = {n: tr[n][1] for n in fr}
        self.H = affine_smooth(tr, list(fr), fixed=fixed)
        g = cv2.cvtColor(frames[ref][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        w = np.zeros_like(g)
        bx0, by0, bx1, by1 = weight_box
        w[by0:by1, bx0:bx1] = 1
        p, score = fit_word(g, w, init, starts, k=dark_k)
        print(f'{name}: fit score {score:.3f}', {k: (round(float(v), 2) if not isinstance(v, tuple) else v)
                                                    for k, v in p.items()})
        self.p_old = p
        self.p_new = match_new_width(p, allow=1.0)   # stay inside the stitched border
        # layers in reference-frame pixel coords (whole frame size for simple warping)
        H, W = frames[ref].shape[:2]
        pad = 30
        self.box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        bx0, by0, bx1, by1 = self.box
        po = dict(p, cx=p['cx'] + x0 - bx0, cy=p['cy'] + y0 - by0)
        pn = dict(self.p_new, cx=po['cx'], cy=po['cy'])
        self.L_old = render_layer(OLD, po, (by1 - by0, bx1 - bx0))
        self.L_new = render_layer(NEW, pn, (by1 - by0, bx1 - bx0))
        self.score = score

    def layers(self, n, shape):
        bx0, by0, _, _ = self.box
        T = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
        M = self.H[n] @ T
        Wd = (shape[1], shape[0])
        A_old = cv2.warpAffine(self.L_old, M[:2], Wd, flags=cv2.INTER_LINEAR)
        A_new = cv2.warpAffine(self.L_new, M[:2], Wd, flags=cv2.INTER_LINEAR)
        return A_old, A_new


def tag_blob(frame, search=(30, 560, 330, 900)):
    """Segment the orange leather tag (V2): returns (mask full-frame uint8, centre, long-axis length, angle)."""
    x0, y0, x1, y1 = search
    m = orange_tag_mask(frame[y0:y1, x0:x1])
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        if st[i, 4] < 150:
            continue
        pts = cv2.findNonZero((lab == i).astype(np.uint8))
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        fill = st[i, 4] / max(w * h, 1)
        if fill < 0.55 or max(w, h) < 2.2 * min(w, h):
            continue
        if best is None or st[i, 4] > best[0]:
            if w < h:
                w, h, ang = h, w, ang + 90
            best = (st[i, 4], i, (cx + x0, cy + y0), w, ang)
    if best is None:
        return None
    _, i, c, L, ang = best
    full = np.zeros(frame.shape[:2], np.uint8)
    blob = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blob, cnts, -1, 1, -1)      # fill the debossed letters inside
    full[y0:y1, x0:x1] = blob
    return full, np.array(c), L, ang % 180


class LeatherTag:
    """V2: the tag is segmented every frame (its outline gives centre / length / angle); the old word
    is replaced by re-filling the tag face from its own leather colour, then 'nookwave' is debossed
    with the ink ratio measured on the reference frame."""

    def __init__(self, frames, rng, ref, crop, weight_box, init, starts, skip=()):
        self.rng, self.skip, self.name = rng, set(skip), 'V2'
        x0, y0, x1, y1 = crop
        g = cv2.cvtColor(frames[ref][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        w = np.zeros_like(g)
        bx0, by0, bx1, by1 = weight_box
        w[by0:by1, bx0:bx1] = 1
        p, score = fit_word(g, w, init, starts)
        print(f'V2: fit score {score:.3f}', {k: round(float(v), 2) for k, v in p.items() if k != 'font'})
        self.p_old = dict(p, cx=p['cx'] + x0, cy=p['cy'] + y0)
        self.p_new = match_new_width(dict(self.p_old))
        self.p_new.update(cx=self.p_old['cx'], cy=self.p_old['cy'])
        self.blobs = {}
        for n in range(rng[0], rng[1] + 1):
            if n in skip:
                continue
            b = tag_blob(frames[n])
            if b is not None and b[2] > 25:
                self.blobs[n] = b
        _, c0, L0, a0 = self.blobs[ref]
        ns = sorted(self.blobs)
        arr = np.array([[self.blobs[n][1][0], self.blobs[n][1][1], self.blobs[n][2],
                         ((self.blobs[n][3] - a0 + 90) % 180) - 90] for n in ns])
        sm = arr.copy()
        for j in range(4):
            col = np.pad(arr[:, j], 1, mode='edge')
            sm[:, j] = np.convolve(col, [0.25, 0.5, 0.25], mode='valid')
        sm[:2] = arr[:2]      # whip-in frames move too fast to average
        self.H = {}
        for i, n in enumerate(ns):
            cx, cy, L, da = sm[i]
            sc = L / L0
            th = np.radians(da)
            R = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            H = np.eye(3)
            H[:2, :2] = R
            H[:2, 2] = np.array([cx, cy]) - R @ c0
            self.H[n] = H
        # ink ratio on the reference frame: darkest 15 % of the tag face vs the rest
        f = frames[ref].astype(np.float32)
        face = cv2.erode(self.blobs[ref][0], np.ones((5, 5), np.uint8)) > 0
        gg = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY)
        t = np.percentile(gg[face], 15)
        ink = np.median(f[face & (gg <= t)], axis=0)
        base = np.median(f[face & (gg > np.percentile(gg[face], 40))], axis=0)
        self.c = np.clip(1 - ink / base, 0, 0.9)
        print('V2 ink c', self.c.round(2))
        H, W = frames[ref].shape[:2]
        self.L_new = render_layer(NEW, self.p_new, (H, W))

    def relabel(self, frame, n):
        blob, c, L, ang = self.blobs[n]
        ys, xs = np.where(blob > 0)
        x0, x1 = max(0, xs.min() - 12), min(frame.shape[1], xs.max() + 12)
        y0, y1 = max(0, ys.min() - 12), min(frame.shape[0], ys.max() + 12)
        sl = (slice(y0, y1), slice(x0, x1))
        crop = frame[sl].astype(np.float32)
        b = blob[sl]
        face = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        g = cv2.cvtColor(frame[sl], cv2.COLOR_BGR2GRAY).astype(np.float32)
        med = cv2.medianBlur(frame[sl], 9)
        letters = (g < cv2.cvtColor(med, cv2.COLOR_BGR2GRAY).astype(np.float32) - 5) & (b > 0)
        w = ((b > 0) & ~cv2.dilate(letters.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(
            np.float32)
        fill = crop.copy()
        done = np.zeros(g.shape, bool)
        for sg in (3.0, 6.0, 12.0, 24.0):
            wb = cv2.GaussianBlur(w, (0, 0), sg)
            vb = cv2.GaussianBlur(crop * w[..., None], (0, 0), sg)
            ok = (wb > 0.03) & ~done
            fill[ok] = vb[ok] / wb[ok][:, None]
            done |= ok
        a = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 0.8)[..., None]
        clean = crop * (1 - a) + fill * a
        H = self.H[n]
        A = cv2.warpAffine(self.L_new, H[:2], (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)[sl]
        A = cv2.GaussianBlur(A, (0, 0), 1.3 if n <= 433 else 0.5)   # whip-in frames are motion-soft
        A = A * cv2.GaussianBlur(face.astype(np.float32), (0, 0), 0.7)
        out = clean * (1 - self.c[None, None, :] * A[..., None])
        return sl, np.clip(out + 0.5, 0, 255).astype(np.uint8)


class BagPrint:
    """Handheld bag: per-frame word fit (chained from a sharp reference frame), caption protected."""
    FREE = ('size', 'cx', 'cy', 'angle', 'sx', 'bend')

    def __init__(self, frames, rng, ref, init, starts, cap_alpha):
        self.rng = rng
        self.cap_alpha = cap_alpha      # full-frame float alpha of the caption layer
        self.params = {}
        g = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY).astype(np.float32)
        best = None
        for st in starts:
            b = self.fit_at(g, dict(init, **st), ('size', 'cx', 'cy', 'angle', 'sx', 'track', 'bend', 'stroke'), 0.8,
                            900)
            if best is None or b[1] > best[1]:
                best = b
        p, s = best
        print(f'V3 ref {ref}: {s:.3f}', {k: round(float(v), 2) for k, v in p.items() if k != 'font'})
        self.params[ref] = (p, s)
        for direction in (1, -1):
            prev = p
            n = ref + direction
            while rng[0] <= n <= rng[1]:
                g = cv2.cvtColor(frames[n], cv2.COLOR_BGR2GRAY).astype(np.float32)
                best = None
                for dx, dy in ((0, 0), (0, -25), (0, 25), (-25, 0), (25, 0), (0, -50), (0, 50), (-50, 0), (50, 0)):
                    b = self.fit_at(g, dict(prev, cx=prev['cx'] + dx, cy=prev['cy'] + dy), self.FREE, 1.5, 250,
                                    self.prior(prev, p))
                    if best is None or b[1] > best[1]:
                        best = b
                    if b[1] > 0.6:
                        break
                b = self.fit_at(g, best[0], self.FREE, 1.2, 300, self.prior(prev, p))   # re-centred window
                if b[1] >= best[1]:
                    best = b
                q, s = best
                self.params[n] = (q, s)
                print(f'V3 {n}: {s:.3f} cx={q["cx"]:.1f} cy={q["cy"]:.1f} size={q["size"]:.1f} '
                      f'ang={q["angle"]:.1f} sx={q["sx"]:.2f} bend={q["bend"]:.1f}')
                prev = q
                n += direction

    @staticmethod
    def prior(prev, ref):
        # the print is a flat logo on a bag: size/angle change smoothly, shape stays close to the reference
        return {'size': (prev['size'], 4.0), 'angle': (prev['angle'], 4.0), 'sx': (ref['sx'], 0.05),
                'bend': (ref['bend'], 4.0)}

    def fit_at(self, g, p, free, blur, iters, prior=None):
        H, W = g.shape
        x0, x1 = int(max(0, p['cx'] - 230)), int(min(W, p['cx'] + 230))
        y0, y1 = int(max(0, p['cy'] - 140)), int(min(H, p['cy'] + 140))
        gc = g[y0:y1, x0:x1]
        pc = dict(p, cx=p['cx'] - x0, cy=p['cy'] - y0)
        A = render_layer(OLD, pc, gc.shape)
        w = cv2.dilate((A > 0.05).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32)
        w[self.cap_alpha[y0:y1, x0:x1] > 0.02] = 0
        q, cost = fit_text(darkness(gc, 13), w, OLD, [p['font']], pc, free=free, blur=blur, iters=iters,
                           prior=prior)
        q = dict(q, cx=q['cx'] + x0, cy=q['cy'] + y0)
        return q, -cost

    def smoothed(self):
        ns = sorted(self.params)
        keys = self.FREE
        arr = np.array([[self.params[n][0][k] for k in keys] for n in ns])
        sm = np.empty_like(arr)
        score = np.array([self.params[n][1] for n in ns])
        good = score >= 0.62          # poorly fitted (blurred) frames: interpolate from good neighbours
        for j in range(arr.shape[1]):
            col = arr[:, j].copy()
            idx = np.arange(len(col))
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            med = np.array([np.median(col[max(0, i - 1):i + 2]) for i in idx])
            sm[:, j] = np.convolve(np.pad(med, 1, mode='edge'), [0.25, 0.5, 0.25], mode='valid')
        out = {}
        for i, n in enumerate(ns):
            p = dict(self.params[n][0])
            p.update({k: sm[i, j] for j, k in enumerate(keys)})
            out[n] = p
        return out


# ----------------------------------------------------------------------------- analysis
def analyse():
    ranges = [(0, 61), CAP, V1, V2, V3]
    frames, total = decode(ranges)
    print('decoded', len(frames), 'of', total)

    cap = Caption([frames[n] for n in range(CAP[0], CAP[1] + 1)], [frames[n] for n in range(0, 62)])

    # caption layer over the bag shot (kept on top of the new print)
    mn, mx = segment_minmax([frames[n] for n in range(CAP3[0], CAP3[1] + 1)])
    fill_a, keep = glyph_alpha(mn, mx)
    fill = fill_a * cv2.dilate(keep, np.ones((3, 3), np.uint8))
    capA_band = np.maximum(fill, outline_alpha(fill))
    capA = np.zeros(frames[V3[0]].shape[:2], np.float32)
    capA[BAND[0]:BAND[1]] = capA_band

    # V1 badge: reference f260
    v1 = Badge('V1', frames, V1, ref=260, roi=(334, 656, 442, 694), crop=(300, 640, 460, 712),
               weight_box=(40, 28, 136, 45), init=dict(size=20, cx=86, cy=39.5, angle=0, bend=5, track=1.4),
               starts=[{}], grow=1)

    # V2 tag: reference f470; f429-430 are the whip-in smear (lettering illegible, left untouched)
    v2 = LeatherTag(frames, V2, ref=470, crop=(90, 670, 150, 810), weight_box=(22, 36, 50, 124),
                    init=dict(size=20.6, cx=35.8, cy=76, angle=93, bend=-7, track=0.3),
                    starts=[{}, dict(bend=-4, size=18)], skip=(429, 430))

    # V3 bag print
    v3 = BagPrint(frames, V3, ref=950, init=dict(font=FUT, size=72, cx=380, cy=1022, angle=-2, bend=-3, track=0,
                                                 sx=0.8, stroke=0.0),
                  starts=[{}, dict(size=68, sx=0.85)], cap_alpha=capA)

    # ---- per-frame relabel of logo frames (two steps so ink strength can be smoothed in time)
    results = {}
    for n in sorted(v2.H):
        results[n] = v2.relabel(frames[n], n)
    print('V2 frames relabelled:', len(v2.H), 'of', V2[1] - V2[0] + 1)
    for item in (v1,):
        rec = {}
        for n in range(item.rng[0], item.rng[1] + 1):
            if n in item.skip:
                continue
            f = frames[n]
            A_old, A_new = item.layers(n, f.shape[:2])
            ys, xs = np.where(A_old > 0.02)
            x0, x1 = max(0, xs.min() - 30), min(f.shape[1], xs.max() + 30)
            y0, y1 = max(0, ys.min() - 30), min(f.shape[0], ys.max() + 30)
            sl = (slice(y0, y1), slice(x0, x1))
            crop = f[sl]
            clean, m = clean_plate(crop, A_old[sl], None, item.grow,
                                   None if item.surface is None else item.surface(crop))
            kb = BANK if n in item.whip else [None, ('g', 0.6), ('g', 1.0)]
            k, c = fit_blur_ink(crop, clean, A_old[sl], np.ones(crop.shape[:2], bool), kernels=kb)
            rec[n] = dict(sl=sl, clean=clean, A_new=A_new[sl], k=k, c=c)
        cs = const_ink(rec)
        # engraved hairline strokes: the core-median under-reads the ink (1-2 px strokes never fully
        # covered by the model), so measure on the reference frame: darkest engraved pixels vs plate
        f = frames[item.ref].astype(np.float32)
        A_old, _ = item.layers(item.ref, f.shape[:2])
        core = A_old > 0.4
        ring = (cv2.dilate((A_old > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (A_old < 0.02)
        ink = np.percentile(f[core], 12, axis=0)
        base = np.median(f[ring], axis=0)
        c_ref = np.clip(1 - ink / base, 0, 0.9)
        print(item.name, 'ink core-median', cs[0].round(2), '-> engraved', c_ref.round(2))
        cs = [c_ref for _ in cs]
        for i, n in enumerate(sorted(rec)):
            r = rec[n]
            results[n] = (r['sl'], composite_new(r['clean'], r['A_new'], r['k'], cs[i]))
        print(item.name, 'ink c (first/mid/last):', cs[0].round(2), cs[len(cs) // 2].round(2), cs[-1].round(2))

    P = v3.smoothed()
    rec = {}
    for n in range(V3[0], V3[1] + 1):
        f = frames[n]
        p = P[n]
        q = match_new_width(dict(p))
        A_old = render_layer(OLD, p, f.shape[:2])
        A_new = render_layer(NEW, dict(q, cx=p['cx'], cy=p['cy']), f.shape[:2])
        ys, xs = np.where(A_old > 0.02)
        x0, x1 = max(0, xs.min() - 45), min(f.shape[1], xs.max() + 45)
        y0, y1 = max(0, ys.min() - 45), min(f.shape[0], ys.max() + 45)
        sl = (slice(y0, y1), slice(x0, x1))
        crop = f[sl]
        prot = capA[sl]
        # blur first (needs a plate): inpaint with a generous mask, then choose kernel + ink
        clean, m = clean_plate(crop, cv2.dilate(A_old[sl], np.ones((7, 7), np.uint8)), prot, 2)
        k, c = fit_blur_ink(crop, clean, A_old[sl], prot < 0.02)
        KA_old = apply_kernel(A_old[sl], k)
        grow = 3 + (int(k[1] // 2) if k is not None and k[0] == 'm' else (int(k[1] * 2) if k else 0))
        clean, m = clean_plate(crop, KA_old, prot, grow)
        rec[n] = dict(sl=sl, clean=clean, A_new=A_new[sl], k=k, c=c, prot=prot, orig=crop)
    cs = const_ink(rec)
    for i, n in enumerate(sorted(rec)):
        r = rec[n]
        out = composite_new(r['clean'], r['A_new'], r['k'], cs[i]).astype(np.float32)
        a = r['prot'][..., None]
        out = out * (1 - a) + r['orig'].astype(np.float32) * a
        results[n] = (r['sl'], np.clip(out + 0.5, 0, 255).astype(np.uint8))
        print('V3', n, 'kernel', r['k'], 'c', cs[i].round(2))
    return cap, results, total


def process(frame, n, cap, results):
    if CAP[0] <= n <= CAP[1]:
        frame = cap.apply(frame)
    if n in results:
        sl, patch = results[n]
        frame[sl] = patch
    return frame


def write_qc(cap, results):
    QC.mkdir(parents=True, exist_ok=True)
    plan = {
        'C1_caption': (list(range(CAP[0], CAP[1] + 1, 6)) + [CAP[1]], (60, 790, 680, 890), 1.5),
        'V1_badge': (list(range(V1[0], V1[1] + 1, 10)) + [V1[1]], (290, 625, 470, 725), 3),
        'V2_tag': (list(range(V2[0], V2[1] + 1, 4)) + [V2[1]], (60, 640, 200, 860), 2),
        'V3_bag': (list(range(V3[0], V3[1] + 1, 3)) + [V3[1]], (60, 780, 660, 1220), 0.8),
    }
    want = {n for v in plan.values() for n in v[0]}
    before, after = {}, {}
    vc = cv2.VideoCapture(str(SRC))
    n = 0
    while True:
        ok, f = vc.read()
        if not ok:
            break
        if n in want:
            before[n] = f.copy()
            after[n] = process(f.copy(), n, cap, results)
        n += 1
    for name, (ns, (x0, y0, x1, y1), sc) in plan.items():
        rows = []
        for n in ns:
            b = cv2.resize(before[n][y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            a = cv2.resize(after[n][y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            sep = np.full((b.shape[0], 6, 3), 255, np.uint8)
            row = np.hstack([b, sep, a])
            cv2.putText(row, f'f{n} before | after', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(QC / f'{name}_f{n:04d}.png'), row)
            rows.append(row)
        # contact sheets of up to 8 rows
        for i in range(0, len(rows), 8):
            cv2.imwrite(str(QC / f'sheet_{name}_{i // 8 + 1}.jpg'), np.vstack(rows[i:i + 8]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])


def render(cap, results, total):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(str(SRC))
    W, H = int(vc.get(3)), int(vc.get(4))
    ff = subprocess.Popen([FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                           '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow',
                           '-pix_fmt', 'yuv420p', '-r', '30', '-movflags', '+faststart', str(OUT)],
                          stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, f = vc.read()
        if not ok:
            break
        ff.stdin.write(process(f, n, cap, results).tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    assert n == total, (n, total)
    print('wrote', OUT, 'frames', n)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(OUT), '-i', str(SRC), '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'copy', '-c:a', 'copy', '-movflags', '+faststart', str(PREVIEW)], check=True)
    print('wrote', PREVIEW)


def main():
    args = sys.argv[1:]
    if '--reuse' in args and CACHE.exists():
        cap, results, total = pickle.load(open(CACHE, 'rb'))
    else:
        cap, results, total = analyse()
        pickle.dump((cap, results, total), open(CACHE, 'wb'))
    if '--frames' in args:
        ns = [int(a) for a in args[args.index('--frames') + 1:] if a.isdigit()]
        QC.mkdir(parents=True, exist_ok=True)
        return write_qc_frames(cap, results, ns)
    if '--qc-only' in args:
        return write_qc(cap, results)
    render(cap, results, total)
    write_qc(cap, results)


def write_qc_frames(cap, results, ns):
    out = HERE / 'intermediate' / 'check'
    out.mkdir(parents=True, exist_ok=True)
    vc = cv2.VideoCapture(str(SRC))
    n = 0
    while n <= max(ns):
        ok, f = vc.read()
        if not ok:
            break
        if n in ns:
            cv2.imwrite(str(out / f'check_{n:04d}.png'), np.hstack([f, process(f.copy(), n, cap, results)]))
        n += 1


if __name__ == '__main__':
    main()
