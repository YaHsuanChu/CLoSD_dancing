import os
import argparse
import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None

"""Build audio feature .npy files for AIST++ from wavs.

- Reads all .wav files under --wav_root (recursively if --recursive)
- Computes log-mel spectrograms (default: 80 mel bins) with librosa
- Saves per-file numpy arrays to --out_dir with same basename, shape (F, T)
- Optionally computes and saves mean/std across features to --stats_out

If librosa is unavailable, falls back to waveform energy features: a single-dim (1, T) log-energy.
"""

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def compute_logmel(y, sr, n_mels=80, hop_length=512, win_length=1024):
    if librosa is None:
        # Fallback: frame-level log energy as a single feature
        # Frame with hop_length and win_length
        # Pad to center like librosa
        pad = win_length // 2
        ypad = np.pad(y, (pad, pad), mode='reflect')
        n_frames = 1 + (len(ypad) - win_length) // hop_length
        feats = np.zeros((1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            e = s + win_length
            frame = ypad[s:e]
            feats[0, i] = np.log(np.sum(frame ** 2) + 1e-8)
        return feats
    else:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)


def gather_wavs(root, recursive=True):
    wavs = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith('.wav'):
                    wavs.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith('.wav'):
                wavs.append(os.path.join(root, f))
    wavs.sort()
    return wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav_root', type=str, required=True, help='Root folder containing .wav files')
    ap.add_argument('--out_dir', type=str, required=True, help='Output folder for .npy features')
    ap.add_argument('--stats_out', type=str, default=None, help='Optional folder to write audio_mean.npy and audio_std.npy')
    ap.add_argument('--n_mels', type=int, default=80)
    ap.add_argument('--hop_length', type=int, default=512)
    ap.add_argument('--win_length', type=int, default=1024)
    ap.add_argument('--recursive', action='store_true')
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    if args.stats_out:
        ensure_dir(args.stats_out)

    wav_files = gather_wavs(args.wav_root, recursive=args.recursive)
    if not wav_files:
        print('No .wav files found under', args.wav_root)
        return

    sums = None
    sumsqs = None
    counts = 0

    for wp in wav_files:
        try:
            y, sr = sf.read(wp)
        except Exception as e:
            print(f'[WARN] Failed to read {wp}: {e}')
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        feats = compute_logmel(y.astype(np.float32), sr, n_mels=args.n_mels, hop_length=args.hop_length, win_length=args.win_length)
        # Save as (F, T)
        base = os.path.splitext(os.path.basename(wp))[0]
        outp = os.path.join(args.out_dir, base + '.npy')
        np.save(outp, feats)

        # accumulate stats over time
        if args.stats_out:
            if sums is None:
                sums = np.zeros(feats.shape[0], dtype=np.float64)
                sumsqs = np.zeros(feats.shape[0], dtype=np.float64)
            sums += feats.sum(axis=1)
            sumsqs += (feats ** 2).sum(axis=1)
            counts += feats.shape[1]

    if args.stats_out and counts > 0:
        mean = (sums / counts).astype(np.float32)
        var = (sumsqs / counts) - (mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        np.save(os.path.join(args.stats_out, 'aistpp_audio_mean.npy'), mean)
        np.save(os.path.join(args.stats_out, 'aistpp_audio_std.npy'), std)
        print('Saved audio stats to', args.stats_out)

    print('Done. Features saved to', args.out_dir)

if __name__ == '__main__':
    main()
