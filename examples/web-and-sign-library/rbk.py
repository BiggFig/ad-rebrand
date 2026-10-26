"""Small rebrand toolkit shared by the job-02 / job-03 rebrand passes (OpenCV + Pillow)."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FUTURA = '/System/Library/Fonts/Supplemental/Futura.ttc'  # index 0 = Medium


def text_alpha(text, size, font_path=FUTURA, index=0, ss=4, stroke=0.0, tracking=0.0):
    """Antialiased alpha (float32) of `text`, tight-cropped horizontally.

    Returns (alpha, info) where info has 'baseline' (row of the baseline inside alpha),
    'xh' (x-height px), 'w' (ink width px). `stroke` > 0 emboldens, < 0 thins (in output px).
    """
    f = ImageFont.truetype(font_path, int(round(size * ss)), index=index)
    asc, desc = f.getmetrics()
    W = int(f.getlength(text) + max(0, tracking) * ss * len(text)) + 8 * ss
    img = Image.new('L', (W, asc + desc + 8 * ss), 0)
    d = ImageDraw.Draw(img)
    if tracking == 0:
        d.text((4 * ss, 4 * ss + asc), text, font=f, fill=255, anchor='ls')
    else:  # per-glyph placement keeps the font's kerning-free advances plus `tracking` px
        x = 4 * ss
        for i, ch in enumerate(text):
            layer = Image.new('L', img.size, 0)
            ImageDraw.Draw(layer).text((x, 4 * ss + asc), ch, font=f, fill=255, anchor='ls')
            img = Image.fromarray(np.maximum(np.asarray(img), np.asarray(layer)))
            x += f.getlength(ch) + tracking * ss
    a = np.asarray(img).astype(np.float32) / 255
    if stroke:
        r = max(1, int(round(abs(stroke) * ss)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        a = cv2.dilate(a, k) if stroke > 0 else cv2.erode(a, k)
    cols = np.where(a.max(axis=0) > 0.02)[0]
    a = a[:, cols[0]:cols[-1] + 1]
    base = 4 * ss + asc
    xb = f.getbbox('x', anchor='ls')
    a = np.pad(a, ((0, (-a.shape[0]) % ss), (0, (-a.shape[1]) % ss)))
    small = cv2.resize(a, (a.shape[1] // ss, a.shape[0] // ss), interpolation=cv2.INTER_AREA)
    return small, dict(baseline=base / ss, xh=-xb[1] / ss, w=small.shape[1])


def place(alpha, info, H, W, M):
    """Warp a text alpha into an HxW canvas with 2x3 affine M that maps text coords
    (origin = left end of baseline) to canvas coords."""
    T = np.float32([[1, 0, 0], [0, 1, -info['baseline']], [0, 0, 1]])
    A = np.vstack([M, [0, 0, 1]]).astype(np.float32) @ T
    return cv2.warpAffine(alpha, A[:2], (W, H), flags=cv2.INTER_LINEAR)


def similarity(scale, angle_deg, tx, ty, sx=None):
    """2x3 matrix: scale (x-scale may differ), rotate (deg, image coords), then translate."""
    sx = scale if sx is None else sx
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.float32([[c * sx, -s * scale, tx], [s * sx, c * scale, ty]])


def feather(mask, r):
    m = mask.astype(np.float32)
    if m.max() > 1:
        m /= 255
    return cv2.GaussianBlur(m, (0, 0), r) if r > 0 else m


def ecc(template_gray, frame_gray, warp, mask=None, motion=cv2.MOTION_AFFINE, iters=200, gauss=5, input_mask=None):
    """ECC; `mask` is a TEMPLATE-domain mask (uint8), `input_mask` optional input-domain mask."""
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    t = template_gray.astype(np.float32)
    i = frame_gray.astype(np.float32)
    tm = mask if mask is not None else np.full(t.shape, 255, np.uint8)
    im = input_mask if input_mask is not None else np.full(i.shape, 255, np.uint8)
    try:
        cc, w = cv2.findTransformECCWithMask(t, i, tm, im, warp.astype(np.float32).copy(), motion, crit, gauss)
        return cc, w
    except cv2.error:
        return -1.0, warp


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def motion_blur(img, vx, vy):
    """Linear motion blur along (vx, vy) px (total smear length)."""
    L = float(np.hypot(vx, vy))
    if L < 1.0:
        return img
    n = int(np.ceil(L)) * 2 + 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    for t in np.linspace(-0.5, 0.5, 4 * n):
        k[int(round(c + t * vy)), int(round(c + t * vx))] += 1
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def hp(g, sigma=4):
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def track_planar_seq(ref_gray, get_gray, frames, tpl_box, scales, search_box, min_score=0.55):
    """track_planar per frame; after a lock, search only near the previous pose."""
    out, prev = {}, None
    x0, y0, x1, y1 = tpl_box
    for n in frames:
        if prev is None:
            v, w = track_planar(ref_gray, get_gray(n), tpl_box, scales, search_box)
        else:
            s0 = warp_scale(prev)
            sc = np.arange(s0 - 0.07, s0 + 0.0701, 0.005)
            px = prev @ np.float64([[x0, x1], [y0, y1], [1, 1]])
            sb = (int(max(0, px[0].min() - 60)), int(max(0, px[1].min() - 60)),
                  int(min(get_gray(n).shape[1], px[0].max() + 60)), int(min(get_gray(n).shape[0], px[1].max() + 60)))
            v, w = track_planar(ref_gray, get_gray(n), tpl_box, sc, sb)
            if w is None or v < min_score:
                v, w = track_planar(ref_gray, get_gray(n), tpl_box, scales, search_box)
        if w is not None and v >= min_score:
            out[n] = (v, w)
            prev = w
        else:
            prev = None
    return out


def track_planar(ref_gray, frame_gray, tpl_box, scales, search_box=None, ecc_motion=cv2.MOTION_AFFINE):
    """Locate the ref region `tpl_box` (x0,y0,x1,y1) in frame. Coarse multi-scale template match on
    high-passed images, then ECC refinement. Returns (score, 2x3 warp ref->frame)."""
    x0, y0, x1, y1 = tpl_box
    H, W = frame_gray.shape
    sx0, sy0, sx1, sy1 = search_box or (0, 0, W, H)
    fh = hp(frame_gray)
    roi = fh[sy0:sy1, sx0:sx1]
    best = (-1.0, None)
    for s in scales:
        t = cv2.resize(ref_gray[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1] or min(t.shape) < 8:
            continue
        r = cv2.matchTemplate(roi, hp(t), cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, np.float32([[s, 0, loc[0] + sx0 - x0 * s], [0, s, loc[1] + sy0 - y0 * s]]))
    v, w0 = best
    if w0 is None:
        return -1.0, None
    cc, w = ecc_roi(hp(ref_gray), fh, w0, tpl_box, margin=30, motion=ecc_motion, gauss=1, pre_blur=0)
    return (cc, w) if cc >= v else (v, w0)


def warp_scale(w):
    return float(np.sqrt(abs(np.linalg.det(w[:2, :2]))))


def warp_img(img, w, W=720, H=1280):
    """Warp ref-space image to frame space, pre-filtering when minifying."""
    s = warp_scale(w)
    if s < 0.97:
        img = cv2.GaussianBlur(img, (0, 0), 0.5 / s - 0.5)
    return cv2.warpAffine(img, w, (W, H), flags=cv2.INTER_LINEAR)


def _to3(w):
    return np.vstack([w, [0, 0, 1]]).astype(np.float64) if w.shape[0] == 2 else w.astype(np.float64)


def ecc_roi(tpl_gray, frame_gray, warp, box, margin=60, motion=cv2.MOTION_AFFINE, gauss=3, mask=None, pre_blur=1.0):
    """ECC on crops for speed. `box` (x0,y0,x1,y1) is the template region in ref coords; `warp` maps
    ref->frame (2x3 or 3x3). The input crop follows the predicted position. Returns (cc, warp)."""
    x0, y0, x1, y1 = box
    H, W = frame_gray.shape
    tx0, ty0 = max(0, x0 - 8), max(0, y0 - 8)
    tx1, ty1 = min(W, x1 + 8), min(H, y1 + 8)
    T = tpl_gray[ty0:ty1, tx0:tx1]
    m = np.zeros(T.shape, np.uint8)
    if mask is None:
        m[y0 - ty0:y1 - ty0, x0 - tx0:x1 - tx0] = 255
    else:
        m[:] = mask[ty0:ty1, tx0:tx1]
    w3 = _to3(warp)
    corners = np.float64([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]).T
    pc = w3 @ corners
    pc = pc[:2] / pc[2]
    ix0, iy0 = int(max(0, pc[0].min() - margin)), int(max(0, pc[1].min() - margin))
    ix1, iy1 = int(min(W, pc[0].max() + margin)), int(min(H, pc[1].max() + margin))
    if ix1 - ix0 < 16 or iy1 - iy0 < 16:
        return -1.0, warp
    I = frame_gray[iy0:iy1, ix0:ix1]
    Ct = np.array([[1, 0, tx0], [0, 1, ty0], [0, 0, 1]], np.float64)
    Ci = np.array([[1, 0, -ix0], [0, 1, -iy0], [0, 0, 1]], np.float64)
    wc = Ci @ w3 @ Ct
    if motion != cv2.MOTION_HOMOGRAPHY:
        wc = wc[:2] / wc[2, 2]
    if pre_blur:
        T = cv2.GaussianBlur(T, (0, 0), pre_blur)
        I = cv2.GaussianBlur(I, (0, 0), pre_blur)
    cc, wc2 = ecc(T, I, wc.astype(np.float32), m, motion=motion, gauss=gauss)
    if cc < 0:
        return cc, warp
    wf = np.linalg.inv(Ci) @ _to3(wc2) @ np.linalg.inv(Ct)
    return cc, (wf[:2] / wf[2, 2]).astype(np.float32) if warp.shape[0] == 2 else (wf / wf[2, 2]).astype(np.float32)


def track_sequence(get_gray, R, first, last, box, motion=cv2.MOTION_AFFINE, min_cc=0.6, margin=60,
                   velocity=True, mask=None, pre_blur=1.0):
    """Track ref region from frame R outward to [first, last] with ECC, initialising each frame
    from the previous (plus constant velocity). Returns {n: (cc, warp)}."""
    rg = get_gray(R)
    eye = np.eye(3, dtype=np.float32) if motion == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    out = {R: (1.0, eye)}
    for rng in (range(R - 1, first - 1, -1), range(R + 1, last + 1)):
        w, hist = eye, [eye]
        for n in rng:
            init = w
            if velocity and len(hist) >= 2:
                init = (w + (hist[-1] - hist[-2])).astype(np.float32)
            cc, w2 = ecc_roi(rg, get_gray(n), init, box, margin, motion, mask=mask, pre_blur=pre_blur)
            if cc < min_cc and velocity:
                cc_b, w2_b = ecc_roi(rg, get_gray(n), w, box, margin, motion, mask=mask, pre_blur=pre_blur)
                if cc_b > cc:
                    cc, w2 = cc_b, w2_b
            if cc < min_cc:
                break
            w = w2
            hist.append(w)
            out[n] = (cc, w)
    return out


def track_chained(get_frame, mask_fn, R, first, last, motion=cv2.MOTION_HOMOGRAPHY, margin=40, min_cc=0.5,
                  pre_blur=1.0):
    """Frame-to-frame ECC (template = previous frame, masked by mask_fn(frame) = uint8 mask of the
    object in that frame), composed back to the ref frame R. Returns {n: (cc, 3x3 warp ref->n)}."""
    out = {R: (1.0, np.eye(3))}
    for step, stop in ((-1, first - 1), (1, last + 1)):
        prev_n, Wp, inc_prev = R, np.eye(3), np.eye(3)
        for n in range(R + step, stop, step):
            fp, fn = get_frame(prev_n), get_frame(n)
            m = mask_fn(fp, prev_n)
            if m is None or m.sum() < 255 * 30:
                break
            ys, xs = np.where(m > 0)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            best = (-1, None)
            for init in (inc_prev, np.eye(3)):
                ini = init if motion == cv2.MOTION_HOMOGRAPHY else init[:2]
                c, inc = ecc_roi(gray(fp), gray(fn), ini.astype(np.float32), box, margin, motion, mask=m,
                                 pre_blur=pre_blur)
                if c > best[0]:
                    best = (c, inc)
            c, inc = best
            if c < min_cc:
                break
            inc3 = _to3(inc)
            Wp = inc3 @ Wp
            inc_prev = inc3
            out[n] = (c, Wp.copy())
            prev_n = n
    return out
