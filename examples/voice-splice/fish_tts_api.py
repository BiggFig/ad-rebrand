"""Zero-shot Fish Audio TTS using a reference clip of the narrator (no voice model is stored)."""
import os
import sys
from pathlib import Path

import ormsgpack
import requests


def load_key():
    for line in Path('~/.config/video-studio/.env').expanduser().read_text().splitlines():
        if line.startswith('FISH_AUDIO_API_KEY='):
            return line.split('=', 1)[1].strip()
    sys.exit('FISH_AUDIO_API_KEY missing')


def tts(text, ref_audio, ref_text, out, model='s1'):
    body = {
        'text': text,
        'references': [{'audio': Path(ref_audio).read_bytes(), 'text': ref_text}],
        'format': 'wav',
        'normalize': True,
        'latency': 'normal',
    }
    r = requests.post('https://api.fish.audio/v1/tts', data=ormsgpack.packb(body),
                      headers={'authorization': f'Bearer {load_key()}',
                               'content-type': 'application/msgpack', 'model': model},
                      timeout=180)
    if r.status_code != 200:
        sys.exit(f'Fish Audio error {r.status_code}: {r.text[:300]}')
    Path(out).write_bytes(r.content)


if __name__ == '__main__':
    ref_audio, ref_text_file, out_dir = sys.argv[1:4]
    ref_text = Path(ref_text_file).read_text().strip()
    os.makedirs(out_dir, exist_ok=True)
    for i, line in enumerate(sys.argv[4:], 1):
        for take in (1, 2, 3):
            tts(line, ref_audio, ref_text, f'{out_dir}/line{i}_take{take}.wav')
        print('ok', i, line)
