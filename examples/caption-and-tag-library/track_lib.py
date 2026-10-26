"""ECC-based planar tracking: per-frame 3x3 warps mapping reference-frame coords -> frame coords."""
import cv2
import numpy as np


def _prep(img, blur):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    if blur:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    return g


def ecc_track(frames, ref, roi, motion=cv2.MOTION_AFFINE, blur=1.0, mask_fn=None, init=None,
              iters=200, eps=1e-5, min_cc=0.5, alt=None):
    """frames: dict n->BGR.  roi: x0,y0,x1,y1 template box in the reference frame.
    Template = reference crop (fixed, no drift).  Each frame is initialised from its neighbour's
    result (walking outwards from ref).  Returns {n: (H 3x3, cc)}."""
    x0, y0, x1, y1 = roi
    tpl = _prep(frames[ref], blur)[y0:y1, x0:x1]
    T0 = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)   # template px -> ref px
    out = {ref: (np.eye(3), 1.0)}
    keys = sorted(frames)
    for direction in (1, -1):
        H = np.eye(3) if init is None else init.get(ref, np.eye(3))
        n = ref + direction
        while n in frames:
            if init is not None and n in init:
                H = init[n]
            g = _prep(frames[n], blur)
            mask = None if mask_fn is None else mask_fn(n)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
            cands = [H] + ([alt[n]] if alt is not None and n in alt else [])
            best = (-1.0, H)
            for H0 in cands:
                W = (H0 @ T0)
                if motion == cv2.MOTION_HOMOGRAPHY:
                    warp = W.astype(np.float32)
                else:
                    warp = (W / W[2, 2])[:2].astype(np.float32)
                try:
                    cc, warp = cv2.findTransformECC(tpl, g, warp, motion, crit, mask, 5)
                except cv2.error:
                    cc = 0.0
                if cc >= min_cc and cc > best[0]:
                    Wn = np.vstack([warp, [0, 0, 1]]) if warp.shape[0] == 2 else warp.astype(np.float64)
                    best = (cc, Wn.astype(np.float64) @ np.linalg.inv(T0))
            cc, H = max(best[0], 0.0), best[1]
            out[n] = (H.copy(), float(cc))
            n += direction
    return out


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]
