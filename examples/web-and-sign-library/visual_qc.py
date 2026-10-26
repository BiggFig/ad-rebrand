"""Before/after QC sheets + flicker check for the rebrand pass.

Usage: python visual_qc.py JOB_DIR
Reads assets/source.mp4 and project/intermediate/rebrand_video.mp4, writes analysis/visual_qc/*.jpg
and analysis/visual_qc/qc_metrics.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# job-02 numbering crops (x0, y0, x1, y1), scale, sample step for moving shots (0 = start/mid/end only)
ITEMS = {
    'C1_caption_Artuvate': ((200, 755, 520, 820), 2.0, 0),
    'C2_caption_result': ((150, 755, 570, 820), 1.6, 0),
    'C3_caption_combines': ((170, 755, 550, 820), 1.6, 0),
    'C4_caption_safe': ((90, 755, 630, 820), 1.3, 0),
    'C5_caption_backed': ((80, 755, 640, 820), 1.3, 0),
    'C6_caption_in_stock': ((110, 755, 610, 820), 1.3, 0),
    'C7_caption_has_become': ((170, 755, 550, 820), 1.6, 0),
    'V1_building_sign': ((120, 440, 470, 640), 1.0, 2),
    'V2_tag_turntable1': ((440, 560, 680, 720), 1.5, 4),
    'V3_device_plate': ((290, 630, 470, 720), 2.0, 10),
    'V4_tag_turntable2': ((560, 550, 680, 670), 2.0, 2),
    'V5_bag_edge': ((560, 760, 720, 980), 1.2, 4),
    'V6_tag_selfie': ((0, 600, 150, 800), 1.5, 4),
    'V7_web_copy': ((100, 680, 620, 750), 1.3, 0),
    'V8_bag_print': ((0, 560, 720, 1280), 0.4, 2),
}
KEY = {'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'C5': 'C5', 'C6': 'C6', 'C7': 'C7'}


def main():
    job = Path(sys.argv[1]).resolve()
    meta = json.loads((job / 'analysis/visual_qc/applied_frames.json').read_text())
    off = meta['offset_vs_job02']
    applied = meta['frames_job02_numbering']
    want = {}
    ranges = {}
    for name, (box, sc, step) in ITEMS.items():
        fr = sorted(applied.get(name.split('_')[0], []))
        if not fr:
            continue
        ranges[name] = (fr[0], fr[-1], len(fr))
        pick = {fr[0], fr[len(fr) // 2], fr[-1]}
        if step:
            pick |= set(fr[::step])
        want[name] = sorted(pick)
    need = {n for v in want.values() for n in v}
    lo, hi = min(min(v) for v in ranges.values()) - 1, max(max(v) for v in ranges.values()) + 1
    capA = cv2.VideoCapture(str(job / 'assets/source.mp4'))
    capB = cv2.VideoCapture(str(job / 'project/intermediate/rebrand_video.mp4'))
    frames, flick = {}, {k: [] for k in ranges}
    prevA = prevB = None
    n = 0
    while True:
        okA, a = capA.read()
        okB, b = capB.read()
        if not (okA and okB):
            break
        m = n - off
        if lo <= m <= hi:
            if m in need:
                frames[m] = (a, b)
            if prevA is not None:
                for name, (f0, f1, _) in ranges.items():
                    if f0 <= m <= f1 + 1:
                        x0, y0, x1, y1 = ITEMS[name][0]
                        dA = np.abs(a[y0:y1, x0:x1].astype(float) - prevA[y0:y1, x0:x1]).mean()
                        dB = np.abs(b[y0:y1, x0:x1].astype(float) - prevB[y0:y1, x0:x1]).mean()
                        flick[name].append((m, round(dA, 2), round(dB, 2)))
            prevA, prevB = a, b
        n += 1
    out = job / 'analysis/visual_qc'
    metrics = {}
    for name, fr in want.items():
        (x0, y0, x1, y1), sc, _ = ITEMS[name]
        rows = []
        for m in fr:
            a, b = frames[m]
            c = np.hstack([a[y0:y1, x0:x1], np.full((y1 - y0, 4, 3), 255, np.uint8), b[y0:y1, x0:x1]])
            c = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            lab = f'{m + off} (job-02 #{m})' if off else str(m)
            cv2.putText(c, lab, (4, 16), 0, 0.5, (0, 0, 255), 1)
            rows.append(c)
        cols = 1 if rows[0].shape[1] > 500 else 2
        if name.startswith('V8'):
            cols = 3
        while len(rows) % cols:
            rows.append(np.zeros_like(rows[0]))
        sheet = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
        cv2.imwrite(str(out / f'{name}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # flicker: frame-to-frame change inside the crop that the edit added beyond the source's own change
        fl = flick[name]
        extra = [(m, dB - dA) for m, dA, dB in fl]
        worst = sorted(extra, key=lambda t: -t[1])[:3]
        metrics[name] = dict(frames_job02=list(ranges[name][:2]), n_frames=ranges[name][2],
                             sampled=[m + off for m in fr],
                             worst_added_temporal_change=[(m + off, round(v, 2)) for m, v in worst])
    (out / 'qc_metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
