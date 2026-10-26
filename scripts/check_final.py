"""QC a finished rebrand: frame count, full decode, loudness vs source, spoken old-name count, before/after sheet.

Usage: check_final.py SOURCE.mp4 FINAL.mp4 --old "physio|fisio|artuvate" --new "nook" [--frames 120 260 945]
Writes FINAL_check.jpg (source | final side by side at each frame) next to FINAL.
Exit code 1 if frame counts differ, decode fails, or any old name is still transcribed.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

WHISPER_MODEL = Path('~/.cache/whisper/ggml-small.en.bin').expanduser()


def run(*cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def frames(path):
    r = run('ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v',
            '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path)
    return int(r.stdout.strip())


def lufs(path):
    r = run('ffmpeg', '-hide_banner', '-i', path, '-af', 'ebur128', '-f', 'null', '-')
    m = re.findall(r'I:\s+(-?[\d.]+) LUFS', r.stderr)
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('final')
    ap.add_argument('--old', required=True, help='regex of old names as whisper may hear them')
    ap.add_argument('--new', required=True, help='regex of the new name')
    ap.add_argument('--frames', type=int, nargs='*', default=[])
    a = ap.parse_args()
    ok = True

    fs, ff = frames(a.source), frames(a.final)
    print(f'frames source {fs} final {ff}', '' if fs == ff else '  <-- MISMATCH')
    ok &= fs == ff
    dec = run('ffmpeg', '-v', 'error', '-i', a.final, '-f', 'null', '-')
    print('decode', 'ok' if not dec.stderr.strip() else 'ERRORS: ' + dec.stderr[:300])
    ok &= not dec.stderr.strip()
    print(f'loudness source {lufs(a.source)} LUFS, final {lufs(a.final)} LUFS')

    with tempfile.TemporaryDirectory() as t:
        wav = Path(t) / 'a.wav'
        run('ffmpeg', '-v', 'error', '-y', '-i', a.final, '-ar', 16000, '-ac', 1, wav)
        text = run('whisper-cli', '-m', WHISPER_MODEL, '-f', wav, '-nt', '-t', 8).stdout
    old = re.findall(a.old, text, re.I)
    new = re.findall(a.new, text, re.I)
    print(f'spoken old names: {len(old)} {old}  new name: {len(new)}')
    ok &= not old

    if a.frames:
        want = set(a.frames)
        ca, cb = cv2.VideoCapture(a.source), cv2.VideoCapture(a.final)
        tiles, n = [], 0
        while n <= max(want):
            o1, f1 = ca.read()
            o2, f2 = cb.read()
            if not (o1 and o2):
                break
            if n in want:
                tile = cv2.resize(np.hstack([f1, f2]), (480, int(480 * f1.shape[0] / (2 * f1.shape[1]))))
                cv2.putText(tile, str(n), (4, 20), 0, 0.6, (0, 0, 255), 2)
                tiles.append(tile)
            n += 1
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        out = Path(a.final).with_name(Path(a.final).stem + '_check.jpg')
        cv2.imwrite(str(out), sheet)
        print('sheet', out)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
