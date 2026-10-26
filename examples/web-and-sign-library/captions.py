"""Burned-in caption swap: FisioRest / Artuvate -> Nookwave.

The captions are white, bold, rounded sans with a ~3 px black outline and a soft shadow, no box,
on one line with x-height rows 782-798. Rather than guess the font, glyph fills for N o k w a v e
are harvested from other captions in the same video. Every word of the affected line is then
re-composited: the old brand word is redrawn from harvested glyphs, the other words are moved
(original pixels) so that the line stays centred, and the background under the caption is inpainted.
"""
import cv2
import numpy as np

BAND = (764, 816)          # rows that contain caption glyphs incl. outline/shadow/descenders
XTOP, BASE = 782, 798      # x-height top row / baseline row of the fills
OUTLINE_R = 3              # outline radius in px (calibrated, see calibrate())
SHADOW = dict(dx=0.7, dy=1.0, sigma=1.2, strength=0.2)
FILL_LO, FILL_HI = 40.0, 250.0

# (frame in job-02 numbering, approx fill x0) of each harvested glyph; several frames are averaged.
GLYPH_SRC = {
    'N': (1645, 158),   # "Now here's why stretching"
    'o': (3440, 264),   # "FisioRest combines"
    'e': (3440, 303),
    'w': (5600, 261),   # "Right now ..."
    'k': (5600, 569),   # "... in stock"
    'v': (3120, 363),   # "Artuvate"
    'a': (3120, 382),
}
WORD = 'Nookwave'
# pairs whose facing sides are diagonal sit a pixel tighter (matches measured Poppins-like spacing)
TIGHT = {('k', 'w'), ('w', 'a'), ('a', 'v'), ('v', 'e')}


def fill_components(frame):
    """Binary fill mask (band coords) of caption glyph fills: bright blobs ringed by dark outline."""
    y0, y1 = BAND
    g = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
    core = (g > 205).astype(np.uint8)
    dark = (g < 110).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    keep = np.zeros(n, bool)
    k3, k7 = np.ones((3, 3), np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 12 or h > 30 or w > 40 or y == 0 or y + h >= g.shape[0]:
            continue
        sub = (lab[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] == i).astype(np.uint8)
        ring = cv2.dilate(sub, k7) - cv2.dilate(sub, k3)
        d = dark[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3]
        if ring.sum() and (d[ring > 0].mean() > 0.75):
            keep[i] = True
    return keep[lab].astype(np.uint8), g


def fill_alpha(g, mask):
    a = np.clip((g.astype(np.float32) - FILL_LO) / (FILL_HI - FILL_LO), 0, 1)
    return a * cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)


def words_from_mask(mask, min_gap=7):
    """Split the fill mask into words by column gaps; returns list of (x0, x1) inclusive."""
    cols = np.where(cv2.dilate(mask, np.ones((1, 3), np.uint8)).any(axis=0))[0]
    if len(cols) == 0:
        return []
    words, s, p = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - p > min_gap:
            words.append((s, p))
            s = c
        p = c
    words.append((s, p))
    return words


def harvest_glyphs(frames):
    """frames: {job-02 frame number: BGR frame}. Returns {ch: (alpha float32 [h,w], y0 abs)}."""
    lib = {}
    for ch, (n0, gx) in GLYPH_SRC.items():
        acc = []
        for n in range(n0 - 4, n0 + 5):
            f = frames.get(n)
            if f is None:
                continue
            mask, g = fill_components(f)
            num, lab = cv2.connectedComponents(mask, connectivity=4)
            # component nearest the expected x
            best, bd = None, 1e9
            for i in range(1, num):
                ys, xs = np.where(lab == i)
                if ys.max() - ys.min() < 10:
                    continue
                d = abs(xs.min() - gx)
                if d < bd:
                    best, bd = i, d
            if best is None or bd > 4:
                continue
            m = (lab == best).astype(np.uint8)
            ys, xs = np.where(m)
            a = fill_alpha(g, m)[ys.min() - 1:ys.max() + 2, xs.min() - 1:xs.max() + 2]
            acc.append((a, ys.min() - 1 + BAND[0]))
        if not acc:
            raise RuntimeError(f'could not harvest glyph {ch}')
        shp = acc[0][0].shape
        acc = [a for a in acc if a[0].shape == shp]
        lib[ch] = (np.median(np.stack([a for a, _ in acc]), axis=0).astype(np.float32), acc[0][1])
    return lib


def layout_word(lib, word=WORD, gap=3):
    """Alpha strip of the word (band coords) and its width."""
    parts, x = [], 0
    for i, ch in enumerate(word):
        a, y = lib[ch]
        if i:
            x += gap - (1 if (word[i - 1], ch) in TIGHT else 0)
        parts.append((x, y, a))
        x += a.shape[1] - 2  # alpha crops carry a 1 px margin each side
    W = x + 2
    strip = np.zeros((BAND[1] - BAND[0], W + 2), np.float32)
    for px, py, a in parts:
        yy = py - BAND[0]
        h, w = a.shape
        strip[yy:yy + h, px:px + w] = np.maximum(strip[yy:yy + h, px:px + w], a)
    return strip


def render_layers(F):
    """Outline and shadow alphas from a fill alpha (all band-sized float32)."""
    k = 2 * OUTLINE_R + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    O = cv2.dilate(F, disk)
    O = np.maximum(O, cv2.GaussianBlur(O, (0, 0), 0.6) * 1.0)
    M = np.float32([[1, 0, SHADOW['dx']], [0, 1, SHADOW['dy']]])
    S = cv2.warpAffine(O, M, (F.shape[1], F.shape[0]))
    S = cv2.GaussianBlur(S, (0, 0), SHADOW['sigma']) * SHADOW['strength']
    return O, S


def composite_caption(bg, F):
    O, S = render_layers(F)
    out = bg.astype(np.float32)
    out *= (1 - S)[..., None]
    out *= (1 - O)[..., None]          # outline is black
    out = out * (1 - F)[..., None] + 255.0 * F[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def word_region(mask_word):
    """Everything a word paints (fill + outline + shadow), feathered alpha."""
    k = 2 * OUTLINE_R + 5
    r = cv2.dilate(mask_word, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    r = cv2.dilate(r, np.ones((3, 3), np.uint8), anchor=(0, 0))  # shadow sits down-right
    return cv2.GaussianBlur(r.astype(np.float32), (0, 0), 0.8)


class CaptionSwap:
    """One caption phrase. `span` = (x0, x1) fill extent of the brand word in the reference frame."""

    def __init__(self, span, ref_frame, lib):
        self.span = span
        mask, _ = fill_components(ref_frame)
        self.ref_mask = mask
        self.new_strip = layout_word(lib)
        self.plate = None  # optional (band image, valid mask) clean background for static shots

    def present(self, mask):
        a, b = self.ref_mask.astype(bool), mask.astype(bool)
        inter = (a & b).sum()
        return inter / max(1, a.sum()) > 0.8 and inter / max(1, b.sum()) > 0.8

    def region(self, frame):
        """Caption paint region (band coords) if this caption is on screen, else None."""
        mask, _ = fill_components(frame)
        return word_region(mask) if self.present(mask) else None

    def apply(self, frame):
        mask, g = fill_components(frame)
        if not self.present(mask):
            return frame, False
        y0, y1 = BAND
        band = frame[y0:y1].copy()
        H, Wf = band.shape[:2]
        cols = np.where(mask.any(axis=0))[0]
        # 1) background under the caption
        hole = (word_region(mask) > 0.02).astype(np.uint8)
        x_lo, x_hi = max(0, cols[0] - 14), min(Wf, cols[-1] + 14)
        bg = band.copy()
        bg[:, x_lo:x_hi] = cv2.inpaint(band[:, x_lo:x_hi], hole[:, x_lo:x_hi], 4, cv2.INPAINT_TELEA)
        if self.plate is not None:
            pimg, pval = self.plate
            v = (pval.astype(np.float32) * hole)[..., None]
            bg = (bg * (1 - v) + pimg * v).astype(np.uint8)
        # 2) new geometry: grow the line symmetrically so it stays centred
        ox0, ox1 = self.span
        new_w = self.new_strip.shape[1] - 2
        d = new_w - (ox1 - ox0 + 1)
        ls, rs = -int(np.floor(d / 2)), int(np.ceil(d / 2))
        out = bg.astype(np.float32)
        # 3) move the untouched words (their own pixels, incl. outline/shadow)
        for s, sel in ((ls, slice(0, ox0 - 2)), (rs, slice(ox1 + 3, Wf))):
            wm = np.zeros_like(mask)
            wm[:, sel] = mask[:, sel]
            if not wm.any():
                continue
            reg = word_region(wm)
            Mx = np.float32([[1, 0, s], [0, 1, 0]])
            src = cv2.warpAffine(band, Mx, (Wf, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            al = cv2.warpAffine(reg, Mx, (Wf, H))[..., None]
            out = out * (1 - al) + src * al
        # 4) the new word
        F = np.zeros((H, Wf), np.float32)
        nx = ox0 + ls - 1
        sw = self.new_strip.shape[1]
        F[:, max(0, nx):nx + sw] = self.new_strip[:, max(0, -nx):Wf - nx]
        frame = frame.copy()
        frame[y0:y1] = composite_caption(out, F)
        return frame, True
