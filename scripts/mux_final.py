"""Mux a job's rebranded picture with its Nookwave voice stem + original music stem.

Usage: mux_final.py JOB_DIR OUT_NAME
Picture: project/intermediate/rebrand_video.mp4 (or the source when the job needed no visual fix).
Audio: analysis/stems/vocals_nookwave.wav + the job's no_vocals.wav stem.
"""
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

J = Path(sys.argv[1])
out = J.parent / sys.argv[2]

picture = J / 'project/intermediate/rebrand_video.mp4'
if not picture.exists():
    picture = J / 'assets/source.mp4'
voice = J / 'analysis/stems/vocals_nookwave.wav'
music = Path(sorted(glob.glob(str(J / 'analysis/stems/htdemucs/*/no_vocals.wav')))[0])

v, sr = sf.read(voice, always_2d=True)
m, sr2 = sf.read(music, always_2d=True)
assert sr == sr2
n = min(len(v), len(m))
mix = J / 'renders/_final_mix.wav'
sf.write(mix, (v[:n] + m[:n]).astype(np.float32), sr)

subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(picture), '-i', str(mix), '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart',
                '-shortest', str(out)], check=True)
print('wrote', out)
