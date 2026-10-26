"""Caption re-set for job-05: rebuild "who's ever bought Nookwave" from the video's own caption glyphs.

The burned-in captions are static per segment, so a temporal min over a segment isolates the
white fill (the outline around it is always black).  Glyph fills are lifted from
  seg f95-138  "who's ever bought FisioRest"   (who's ever bought, o, w, v, e)
  seg f0-61    "with that knot" (line 2)       (k, a)
and 'N' (absent from every caption) is drawn as a geometric polygon matched to the F/R stems.
The black outline is re-synthesised from the measured radial profile (solid to ~3.2 px, gone by ~4.6 px).
"""
import cv2
import numpy as np

BAND = (780, 900)          # rows analysed
OUT_R0, OUT_R1 = 3.2, 4.6  # outline alpha 1 until R0, linear to 0 at R1 (px from fill edge)
SS = 4                     # supersampling for outline distance


def segment_minmax(frames):
    mn = mx = None
    for f in frames:
        g = cv2.cvtColor(f[BAND[0]:BAND[1]], cv2.COLOR_BGR2GRAY)
        mn = g if mn is None else np.minimum(mn, g)
        mx = g if mx is None else np.maximum(mx, g)
    return mn, mx


def glyph_alpha(mn, mx):
    """Fill alpha of caption glyphs (components whose surrounding ring is always black)."""
    fill = (mn >= 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fill)
    keep = np.zeros_like(fill)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 20 or h > 45:
            continue
        m = (lab == i).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.dilate(m, np.ones((3, 3), np.uint8))
        if ring.sum() and np.mean(mx[ring > 0] < 90) > 0.85:
            keep |= m
    return np.clip(mn.astype(np.float32) / 255.0, 0, 1), keep


def pick(alpha, keep, x0, x1, r0=0, r1=999):
    """Fill alpha of the kept components whose centre lies in columns [x0,x1) and rows [r0,r1)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(keep)
    m = np.zeros_like(keep)
    for i in range(1, n):
        if x0 <= cen[i][0] < x1 and r0 <= cen[i][1] < r1:
            m |= (lab == i).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    a = alpha * m
    cols = np.where(a.max(0) > 0.05)[0]
    return a[:, cols.min():cols.max() + 1]


def poly_N(cap_h=25, width=21, stem=6, diag=8.5):
    """Geometric N (Poppins-like) as fill alpha, cap_h tall, baseline at bottom row."""
    S = 8
    H, W = cap_h * S, width * S
    img = np.zeros((H, W), np.uint8)
    st = stem * S
    cv2.rectangle(img, (0, 0), (st - 1, H - 1), 255, -1)
    cv2.rectangle(img, (W - st, 0), (W - 1, H - 1), 255, -1)
    d = diag * S
    pts = np.array([[0, 0], [d, 0], [W, H], [W - d, H]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return cv2.resize(img, (width, cap_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def build_line(g95, g0):
    """Return fill alpha (BAND rows x 720 cols) for the new centred line and its x extent."""
    parts = []  # (alpha_strip in seg95 row coords, gap_before)
    parts.append((pick(*g95, 100, 450), 0))          # "who's ever bought" exactly as set
    rows = g95[0].shape[0]

    def from0(x0, x1):  # seg 0 line 2 sits 26 px lower than seg95's line
        return np.roll(pick(*g0, x0, x1, 70), -26, axis=0)
    o = pick(*g95, 163, 184)
    w = pick(*g95, 106, 137)
    e = pick(*g95, 222, 243)
    v = pick(*g95, 244, 266)
    k = from0(407, 427)
    a = from0(359, 381)
    N = np.zeros((rows, 21), np.float32)
    base = 855 - BAND[0]   # baseline row (bottom of F)
    N[base - 25:base] = poly_N()
    seq = [(N, 11), (o, 3), (o, 3), (k, 3), (w, 2), (a, 2), (v, 1), (e, 1)]
    parts += seq
    total = sum(p.shape[1] + g for p, g in parts)
    x = int(round(363 - total / 2))     # original line centre x≈363
    line = np.zeros((rows, 720), np.float32)
    x0 = x
    for p, g in parts:
        x += g
        line[:, x:x + p.shape[1]] = np.maximum(line[:, x:x + p.shape[1]], p)
        x += p.shape[1]
    return line, (x0, x)


def outline_alpha(fill):
    up = cv2.resize(fill, None, fx=SS, fy=SS, interpolation=cv2.INTER_CUBIC)
    d = cv2.distanceTransform((up < 0.5).astype(np.uint8), cv2.DIST_L2, 5) / SS
    o = np.clip((OUT_R1 - d) / (OUT_R1 - OUT_R0), 0, 1)
    return cv2.resize(o, (fill.shape[1], fill.shape[0]), interpolation=cv2.INTER_AREA)


class Caption:
    def __init__(self, seg95_frames, seg0_frames):
        mn95, mx95 = segment_minmax(seg95_frames)
        mn0, mx0 = segment_minmax(seg0_frames)
        g95 = glyph_alpha(mn95, mx95)
        g0 = glyph_alpha(mn0, mx0)
        old = g95[0] * cv2.dilate(g95[1], np.ones((3, 3), np.uint8))
        old[:, :95] = 0
        d = cv2.distanceTransform((old < 0.3).astype(np.uint8), cv2.DIST_L2, 5)
        self.erase = (d <= 6.5).astype(np.uint8) * 255      # old caption footprint (fill+outline)
        self.fill, self.extent = build_line(g95, g0)
        self.outl = outline_alpha(self.fill)

    def apply(self, frame):
        y0, y1 = BAND
        band = frame[y0:y1]
        clean = cv2.inpaint(band, self.erase, 4, cv2.INPAINT_TELEA).astype(np.float32)
        ao = self.outl[..., None]
        af = self.fill[..., None]
        out = clean * (1 - ao)                       # black outline
        out = out * (1 - af) + 254.0 * af            # white fill
        frame[y0:y1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return frame
