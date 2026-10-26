"""Render / fit a single word laid on a (possibly arched) baseline.

params = dict(font=(path, index), size=em px, cx, cy (centre of the word's x-height box),
              angle (deg, text direction, 0 = left->right), bend (px, + = middle raised),
              track (extra px between letters), sx (horizontal stretch), stroke (px dilation, <0 thins))
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

SS = 4
VERBOSE = False
_fcache = {}


def _font(path, idx, size):
    k = (path, idx, size)
    if k not in _fcache:
        _fcache[k] = ImageFont.truetype(path, size, index=idx)
    return _fcache[k]


def canonical(word, font, size, track=0.0, stroke=0.0):
    """Word alpha at SS x scale.  Returns (alpha float32, (ox, oy)) where (ox, oy) is the centre of the
    x-height box (horizontally: centre of the inked extent) in canonical px."""
    path, idx = font
    f = _font(path, idx, max(4, int(round(size * SS))))
    asc, desc = f.getmetrics()
    xh = f.getbbox('x')[3] - f.getbbox('x')[1]
    W = int(sum(f.getlength(c) for c in word) + track * SS * len(word) + 8 * SS * 4)
    H = asc + desc + 8 * SS
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    x = 4 * SS * 2
    base = asc + 4 * SS
    for c in word:
        d.text((x, base), c, font=f, fill=255, anchor='ls')
        x += f.getlength(c) + track * SS
    a = np.asarray(img, np.float32) / 255.0
    if abs(stroke) > 0.05:
        r = int(round(abs(stroke) * SS))
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            a = cv2.dilate(a, k) if stroke > 0 else cv2.erode(a, k)
    cols = np.where(a.max(0) > 0.3)[0]
    ox = (cols.min() + cols.max()) / 2.0
    oy = base - xh / 2.0
    return a, (ox, oy, (cols.max() - cols.min()) / 2.0 / SS)


def render_layer(word, p, shape, canon=None):
    """Alpha (float32, shape HxW) of `word` placed with params p in output pixel coords."""
    a, (ox, oy, ihw) = canon if canon is not None else canonical(word, p['font'], p['size'], p.get('track', 0),
                                                            p.get('stroke', 0))
    H, W = shape
    th = np.radians(p.get('angle', 0.0))
    c, s = np.cos(th), np.sin(th)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xs - p['cx'], ys - p['cy']
    u = (c * dx + s * dy) / p.get('sx', 1.0)     # along text
    v = -s * dx + c * dy                         # across text (down = +)
    hw = max(1.0, p.get('halfw', ihw))
    v = v + p.get('bend', 0.0) * (1 - np.clip(u / hw, -1.6, 1.6) ** 2)   # arched baseline
    mx = (u * SS + ox).astype(np.float32)
    my = (v * SS + oy).astype(np.float32)
    small = cv2.GaussianBlur(a, (0, 0), SS * 0.42)   # pre-filter before point sampling
    return cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderValue=0)


def fit_text(D, weight, word, fonts, init, free=('size', 'cx', 'cy', 'angle', 'bend', 'track'),
             blur=0.7, iters=600, prior=None):
    """Maximise correlation between rendered `word` and darkness map D (within `weight`)."""
    best = None
    for font in fonts:
        keys = list(free)

        def unpack(x):
            p = dict(init)
            p['font'] = font
            for k, v in zip(keys, x):
                p[k] = v
            return p

        def cost(x):
            p = unpack(x)
            if p['size'] < 4:
                return 1.0
            r = render_layer(word, p, D.shape)
            if blur:
                r = cv2.GaussianBlur(r, (0, 0), blur)
            w = weight
            rm, dm = (r * w).sum() / w.sum(), (D * w).sum() / w.sum()
            num = ((r - rm) * (D - dm) * w).sum()
            den = np.sqrt(((r - rm) ** 2 * w).sum() * ((D - dm) ** 2 * w).sum()) + 1e-9
            pen = 0.0
            if prior:
                for k, (c0, sg) in prior.items():
                    pen += 0.02 * ((p[k] - c0) / sg) ** 2
            return -num / den + pen
        x0 = np.array([init[k] for k in keys], np.float64)
        steps = {'size': 1.5, 'cx': 2, 'cy': 2, 'angle': 2, 'bend': 1.5, 'track': 0.5, 'sx': 0.05,
                 'stroke': 0.3, 'halfw': 5}
        simplex = [x0] + [x0 + np.eye(len(keys))[i] * steps[k] for i, k in enumerate(keys)]
        res = minimize(cost, x0, method='Nelder-Mead',
                       options=dict(maxiter=iters, initial_simplex=np.array(simplex), xatol=1e-2, fatol=1e-4))
        p = unpack(res.x)
        if best is None or res.fun < best[1]:
            best = (p, res.fun)
        if VERBOSE: print('  fit', font, round(-res.fun, 4), {k: round(float(p[k]), 2) for k in keys})
    return best
