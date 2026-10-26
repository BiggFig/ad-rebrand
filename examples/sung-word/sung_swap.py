#!/usr/bin/env python
"""Sung word swap: "It's called FisioRest" -> "It's called Nookwave" (job-06).

Re-sings a spoken TTS-clone "Nookwave" with WORLD (pyworld): time-warps the
phones onto the original syllable grid, imposes the original sung F0 contour,
keeps the clone's spectral envelope (natural formants), matches loudness/EQ,
adds a short synthetic reverb, and splices into the demucs vocals stem.

Run:  ~/.cache/video-studio-venv/bin/python -I project/sung_swap.py
Outputs: analysis/stems/vocals_nookwave.wav, renders/preview_audio_nookwave.wav,
         renders/alt_*.wav (62-70 s excerpts), project/sung_swap.json
"""
import json, os, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, pyworld as pw, librosa
from scipy.signal import fftconvolve, butter, sosfiltfilt

JOB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKES = os.path.join(os.path.dirname(JOB), "voice_refs", "takes")
VOC = os.path.join(JOB, "analysis/stems/htdemucs/source/vocals.wav")
BED = os.path.join(JOB, "analysis/stems/htdemucs/source/no_vocals.wav")
WHISPER = "/opt/homebrew/bin/whisper-cli"
WMODEL = os.path.expanduser("~/.cache/whisper/ggml-small.en.bin")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FP = 5.0  # WORLD frame period, ms
SR = 44100

# Source phone anchors (s) in the clone takes, measured from spectrogram/energy:
# [N onset, "oo" onset, k closure, k burst, w onset, v onset, end]
SOURCES = {
    "take1b": ("singerD_take1.mp3", [4.840, 4.970, 5.130, 5.215, 5.270, 5.535, 5.720]),
    "take2b": ("singerD_take2.mp3", [4.832, 4.950, 5.045, 5.125, 5.240, 5.500, 5.690]),
    "take1a": ("singerD_take1.mp3", [0.585, 0.700, 0.866, 0.945, 1.000, 1.315, 1.470]),
}
# Target anchors on the song timeline (original: F 65.60, "Fi-si-o" A3/Bb3 to 65.98,
# "Re" C4 66.00-66.14 -> D4 66.16-66.46, "st" 66.50-66.62, t 66.66)
TIMINGS = {
    # Nook on the Fi-si-o notes, wave on Re(C4)->D4 sustain, v on "st"
    "std":     [65.583, 65.680, 65.925, 65.972, 66.000, 66.500, 66.680],
    # Noo held through C4 (melisma), k at 66.13, "wave" only on the D4 note
    "longnoo": [65.583, 65.670, 66.110, 66.140, 66.165, 66.500, 66.680],
    # wave vowel held to 66.56, short v (closer to sung-vowel style)
    "longwave":[65.583, 65.680, 65.925, 65.972, 66.000, 66.560, 66.685],
}
CUT0, CUT1 = 65.600, 66.685   # replaced region on the stem (orig fades out 65.580-65.600)
VBOOST_DB = 4.0


def load_take(name):
    x, _ = librosa.load(os.path.join(TAKES, name), sr=SR, mono=True)
    return x.astype(np.float64)


def orig_f0(vm):
    a, b = int(65.0 * SR), int(67.4 * SR)
    seg = vm[a:b].astype(np.float64)
    f0, t = pw.harvest(seg, SR, f0_floor=150, f0_ceil=700, frame_period=FP)
    t = t + 65.0
    vi = f0 > 0
    f0i = np.exp(np.interp(t, t[vi], np.log(f0[vi])))  # fill unvoiced gaps (the "s" of Fisio)
    return t, f0, f0i


def frame_db(x, t0, t1, hop=0.01):
    out = []
    for s in np.arange(t0, t1, hop):
        seg = x[int(s * SR):int((s + hop) * SR)]
        out.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
    return np.array(out)


def resing(take, src_anch, tgt_anch, f0_t, f0_c, vibrato_cents=0.0):
    x = load_take(take)
    a0, a1 = src_anch[0] - 0.02, src_anch[-1] + 0.03
    seg = x[int(a0 * SR):int(a1 * SR)]
    f0s, ts = pw.harvest(seg, SR, f0_floor=100, f0_ceil=600, frame_period=FP)
    sp = pw.cheaptrick(seg, f0s, ts, SR)
    ap = pw.d4c(seg, f0s, ts, SR)
    ts = ts + a0
    t0, t1 = tgt_anch[0], tgt_anch[-1] + 0.02
    tt = np.arange(t0, t1, FP / 1000)
    # piecewise-linear target->source time map
    src_t = np.interp(tt, tgt_anch + [tgt_anch[-1] + 0.02], src_anch + [src_anch[-1] + 0.02])
    idx = (src_t - ts[0]) / (FP / 1000)
    i0 = np.clip(np.floor(idx).astype(int), 0, len(ts) - 2); fr = (idx - i0)[:, None]
    lsp = np.log(sp + 1e-16)
    spn = np.exp((1 - fr) * lsp[i0] + fr * lsp[i0 + 1])
    apn = (1 - fr) * ap[i0] + fr * ap[i0 + 1]
    # voicing from source (nearest), pitch from original melody
    vs = f0s[np.clip(np.round(idx).astype(int), 0, len(f0s) - 1)] > 0
    # k closure/burst are unvoiced by design
    vs &= ~((tt >= tgt_anch[2] + 0.005) & (tt < tgt_anch[4] - 0.003))
    f0n = np.interp(tt, f0_t, f0_c)
    if vibrato_cents > 0:
        ramp = np.clip((tt - 66.20) / 0.15, 0, 1)
        f0n *= 2 ** (vibrato_cents / 1200 * ramp * np.sin(2 * np.pi * 5.5 * (tt - 66.2)))
    f0n = np.where(vs, f0n, 0.0)
    y = pw.synthesize(np.ascontiguousarray(f0n), np.ascontiguousarray(spn), np.ascontiguousarray(apn), SR, frame_period=FP)
    return y, t0, dict(src=take, src_anchors=src_anch, tgt_anchors=tgt_anch)


def eq_match(y, ref, n_fft=2048):
    """static 1/3-oct-smoothed EQ so y's long-term spectrum matches ref (clipped +-9 dB)"""
    def ltas(z):
        S = np.abs(librosa.stft(z, n_fft=n_fft, hop_length=512))
        e = (S ** 2).sum(0); keep = e > np.percentile(e, 40)
        return np.sqrt((S[:, keep] ** 2).mean(1)) + 1e-9
    f = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    r = 20 * np.log10(ltas(ref)) - 20 * np.log10(ltas(y))
    sm = np.array([r[(f >= fc / 2 ** (1 / 6)) & (f <= fc * 2 ** (1 / 6))].mean() if fc > 40 else r[1]
                   for fc in np.maximum(f, 1)])
    sm = np.clip(sm - np.median(sm[(f > 300) & (f < 3000)]), -9, 9)
    sm[f < 80] = -12; sm[f > 12000] = np.minimum(sm[f > 12000], 0)
    Y = librosa.stft(y, n_fft=n_fft, hop_length=512)
    return librosa.istft(Y * (10 ** (sm / 20))[:, None], hop_length=512, length=len(y))


def make_ir(rt60=1.0, n=1.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n * SR)) / SR
    irs = []
    for ch in range(2):
        noise = rng.standard_normal(len(t)) * np.exp(-6.91 * t / rt60)
        noise[:int(0.015 * SR)] = 0  # predelay
        sos = butter(2, [250, 6000], btype="band", fs=SR, output="sos")
        ir = sosfiltfilt(sos, noise)
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    return np.stack(irs, 1)


def build(vocals, vm, f0_t, f0_c, take_key="take1b", timing="std", wet_db=-13.0,
          tail_keep_db=0.0, vibrato_cents=0.0):
    take, src = SOURCES[take_key]
    tgt = TIMINGS[timing]
    y, t0, meta = resing(take, src, tgt, f0_t, f0_c, vibrato_cents)
    # EQ toward the stem's sung timbre (lead vocal 60-65.5 s)
    y = eq_match(y, vm[int(60.0 * SR):int(65.5 * SR)])
    # loudness: per-syllable 75th-pct frame RMS matched to the original
    tt = t0 + np.arange(len(y)) / SR
    g = np.ones_like(y)
    for (s0, s1) in [(tgt[0], tgt[2]), (tgt[4], tgt[6])]:
        o = frame_db(vm, s0, s1); n = frame_db(np.concatenate([np.zeros(int(t0 * SR)), y]), s0, s1)
        gdb = np.clip(np.percentile(o, 75) - np.percentile(n, 75), -15, 15)
        g[(tt >= s0 - 0.01) & (tt < s1 + 0.01)] = 10 ** (gdb / 20)
    # second syllable vowel ("wave" on the C4->D4 sustain): follow the original's
    # 40 ms loudness envelope (limited +-6 dB) so the held note doesn't sag
    yy = y * g
    hop = 0.01; fr_t = np.arange(tgt[4], tgt[5], hop)
    o = frame_db(vm, tgt[4], tgt[5], hop)
    n = frame_db(np.concatenate([np.zeros(int(t0 * SR)), yy]), tgt[4], tgt[5], hop)
    d = np.convolve(np.clip(o - n, -6, 6), np.ones(4) / 4, "same")
    d[:3] = d[3]; d[-3:] = d[-4]
    gv = np.interp(tt, fr_t + hop / 2, d, left=0, right=0)
    ramp = np.clip((tt - tgt[4]) / 0.03, 0, 1) * np.clip((tgt[5] + 0.02 - tt) / 0.04, 0, 1)
    g = g * 10 ** (gv * ramp / 20)
    g[tt >= tgt[5]] *= 10 ** (VBOOST_DB / 20)  # v is weaker than the original "st" sibilants
    k = int(0.03 * SR); g = np.convolve(g, np.hanning(k) / np.hanning(k).sum(), "same")
    y = y * g
    # fades on the new word
    fi, fo = int(0.012 * SR), int(0.02 * SR)
    y[:fi] *= np.linspace(0, 1, fi); y[-fo:] *= np.linspace(1, 0, fo)
    # place on a stereo buffer + reverb
    N = len(vocals)
    new = np.zeros((N, 2)); a = int(t0 * SR); new[a:a + len(y)] += y[:, None]
    wet = np.zeros((N, 2)); ir = make_ir()
    seg = new[a - SR // 10: a + len(y) + int(1.3 * SR), 0]
    for c in range(2):
        w = fftconvolve(seg, ir[:, c])[:len(seg)]
        wet[a - SR // 10: a - SR // 10 + len(w), c] = w
    new += wet * 10 ** (wet_db / 20)
    # remove the old word: mask original from CUT0 to CUT1 (15 ms fades);
    # keep original D4 tail after CUT1 at tail_keep_db (it is the same note we end on)
    t = np.arange(N) / SR
    keep = np.ones(N)
    fade = 0.020
    keep = np.where(t < CUT0 - fade, 1.0, keep)
    m = (t >= CUT0 - fade) & (t < CUT0); keep[m] = np.linspace(1, 0, m.sum())
    keep[(t >= CUT0) & (t < CUT1)] = 0.0
    m = (t >= CUT1) & (t < CUT1 + 0.03); keep[m] = np.linspace(0, 10 ** (tail_keep_db / 20), m.sum())
    m = t >= CUT1 + 0.03; keep[m] = 10 ** (tail_keep_db / 20)
    m = t >= 67.4; keep[m] = 1.0
    out = vocals * keep[:, None] + new
    return out, meta


def f0_error(out_m, f0_t, f0_raw, t0=65.60, t1=66.62):
    a, b = int(65.0 * SR), int(67.4 * SR)
    f0n, t = pw.harvest(out_m[a:b].astype(np.float64), SR, f0_floor=150, f0_ceil=700, frame_period=FP)
    t = t + 65.0
    sel = (t >= t0) & (t <= t1)
    fo = np.interp(t[sel], f0_t, f0_raw); fn = f0n[sel]
    ok = (fo > 0) & (fn > 0)
    err = 12 * np.log2(fn[ok] / fo[ok])
    return dict(median_abs_semitone_err=float(np.median(np.abs(err))),
                p90_abs_semitone_err=float(np.percentile(np.abs(err), 90)),
                voiced_frames_compared=int(ok.sum()),
                octave_errors=int((np.abs(err) > 6).sum()))


def whisper(x, t0, t1, tag, prompt=True):
    tmp = os.path.join(JOB, "renders", f"_tmp_{tag}.wav")
    seg = x[int(t0 * SR):int(t1 * SR)]
    seg = seg.mean(1) if seg.ndim > 1 else seg
    y = librosa.resample(seg, orig_sr=SR, target_sr=16000)
    sf.write(tmp, y / (np.abs(y).max() + 1e-9) * 0.9, 16000)
    cmd = [WHISPER, "-m", WMODEL, "-f", tmp, "-nt", "-np"] + (["--prompt", "Nookwave"] if prompt else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp)
    return " ".join(r.stdout.split())


def main():
    vocals, sr = sf.read(VOC); assert sr == SR
    bed, _ = sf.read(BED)
    vm = vocals.mean(1)
    f0_t, f0_raw, f0_c = orig_f0(vm)
    os.makedirs(os.path.join(JOB, "renders"), exist_ok=True)
    report = {"original": {
        "word_region": [65.58, 66.70], "replaced_region": [CUT0, CUT1],
        "notes": "called ~Bb3 (233 Hz) | Fi ~A3 (224 Hz) 65.60-65.72 | si-o ~Bb3 (240 Hz) 65.80-65.98 | "
                 "Re C4 (262 Hz) 66.00-66.14 | glide to D4 (294 Hz) 66.16-66.62, tail rings on D4 to ~67.25; "
                 "no measurable vibrato (D4 sustain within +-10 cents)",
        "f0_10ms": [[round(float(a), 3), round(float(b), 1)] for a, b in zip(f0_t[::2], f0_raw[::2]) if 65.55 <= a <= 66.75],
    }, "variants": {}}
    variants = {
        "best": dict(take_key="take1b", timing="std"),
        "alt_take2": dict(take_key="take2b", timing="std"),
        "alt_longnoo": dict(take_key="take1b", timing="longnoo"),
        "alt_longwave_vib": dict(take_key="take1b", timing="longwave", vibrato_cents=25),
        "alt_newtail": dict(take_key="take1b", timing="std", tail_keep_db=-12.0, wet_db=-9.0),
    }
    for name, kw in variants.items():
        out, meta = build(vocals, vm, f0_t, f0_c, **kw)
        om = out.mean(1)
        mix = out + bed
        pk = np.abs(mix).max()
        rep = dict(params=kw, **meta, f0=f0_error(om, f0_t, f0_raw),
                   whisper_vocals_prompted=whisper(out, 64.9, 67.4, name),
                   whisper_vocals_unprompted=whisper(out, 64.9, 67.4, name + "u", prompt=False),
                   whisper_mix_line_prompted=whisper(mix, 61.4, 71.4, name + "m"),
                   new_word_rms_db=float(20 * np.log10(np.sqrt(np.mean(om[int(65.6 * SR):int(66.62 * SR)] ** 2)))),
                   orig_word_rms_db=float(20 * np.log10(np.sqrt(np.mean(vm[int(65.6 * SR):int(66.62 * SR)] ** 2)))),
                   mix_peak=float(pk))
        if name == "best":
            assert out.shape == vocals.shape
            sf.write(os.path.join(JOB, "analysis/stems/vocals_nookwave.wav"), np.clip(out, -1, 1), SR, subtype="PCM_16")
            sf.write(os.path.join(JOB, "renders/preview_audio_nookwave.wav"), mix / max(1.0, pk / 0.99), SR, subtype="PCM_16")
            sf.write(os.path.join(JOB, "renders/alt_best_excerpt_62-70s.wav"),
                     (mix / max(1.0, pk / 0.99))[62 * SR:70 * SR], SR, subtype="PCM_16")
        else:
            sf.write(os.path.join(JOB, f"renders/{name}_62-70s.wav"), (mix / max(1.0, pk / 0.99))[62 * SR:70 * SR], SR, subtype="PCM_16")
        report["variants"][name] = rep
        print(name, json.dumps({k: rep[k] for k in ("f0", "whisper_vocals_prompted", "whisper_vocals_unprompted", "whisper_mix_line_prompted", "new_word_rms_db", "orig_word_rms_db")}))
    # reference: original transcripts for comparison
    report["original"]["whisper_vocals_prompted"] = whisper(vocals, 64.9, 67.4, "orig")
    report["original"]["whisper_vocals_unprompted"] = whisper(vocals, 64.9, 67.4, "origu", prompt=False)
    json.dump(report, open(os.path.join(JOB, "project/sung_swap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
