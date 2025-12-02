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
- Targets a specific feature frame-rate via --target_fps (default 60)
- If --motion_lookup_root is provided, aligns each audio feature's time axis
    to exactly match the corresponding motion length (same basename), ensuring
    audio frames T == motion frames for direct slicing later.
- Saves per-file numpy arrays to --out_dir with same basename, shape (F, T)
- Optionally computes and saves mean/std across features to --stats_out

If librosa is unavailable, falls back to waveform energy features: a single-dim (1, T) log-energy.
"""

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def compute_logmel(y, sr, n_mels=80, hop_length=512, win_length=1024, center=False):
    if librosa is None:
        # Fallback: frame-level log energy as a single feature
        # Frame with hop_length and win_length
        # Pad to center like librosa if center=True
        pad = win_length // 2 if center else 0
        ypad = np.pad(y, (pad, pad), mode='reflect') if center else y
        if len(ypad) < win_length:
            n_frames = 0
        else:
            n_frames = 1 + (len(ypad) - win_length) // hop_length
        feats = np.zeros((1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            e = s + win_length
            frame = ypad[s:e]
            feats[0, i] = np.log(np.sum(frame ** 2) + 1e-8)
        return feats
    else:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length, power=2.0, center=center)
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
    ap.add_argument('--hop_length', type=int, default=None, help='Explicit hop length in samples. If not set, computed from target_fps.')
    ap.add_argument('--win_length', type=int, default=None, help='Explicit window length in samples. If not set, defaults to 2*hop_length.')
    ap.add_argument('--target_fps', type=float, default=60.0, help='Desired feature frame rate (frames per second). Default 60.')
    ap.add_argument('--motion_lookup_root', type=str, default=None, help='Root to search for corresponding motion files (.npy/.pkl) to match time length.')
    ap.add_argument('--center', action='store_true', help='Use librosa center=True framing. Default False for exact control.')
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

    def find_motion_path(basename_no_ext: str):
        if not args.motion_lookup_root:
            return None
        for dirpath, _, filenames in os.walk(args.motion_lookup_root):
            for f in filenames:
                if not (f.endswith('.npy') or f.endswith('.pkl')):
                    continue
                if os.path.splitext(f)[0] == basename_no_ext:
                    return os.path.join(dirpath, f)
        return None

    def load_motion_length(motion_path: str) -> int:
        try:
            if motion_path.endswith('.npy'):
                arr = np.load(motion_path, allow_pickle=True)
                if arr.ndim == 3:
                    return int(arr.shape[0])
                if arr.ndim == 2:
                    return int(arr.shape[0])
                if arr.dtype == object and arr.shape == ():
                    obj = arr.item()
                    for key in ['joints', 'joints3d', 'positions', 'motion', 'smpl_poses']:
                        if key in obj:
                            v = np.array(obj[key])
                            return int(v.shape[0])
            elif motion_path.endswith('.pkl'):
                import pickle
                with open(motion_path, 'rb') as f:
                    obj = pickle.load(f)
                for key in ['joints', 'joints3d', 'positions', 'motion', 'smpl_trans', 'smpl_poses']:
                    if key in obj:
                        v = np.array(obj[key])
                        return int(v.shape[0])
        except Exception:
            return None
        return None

    def align_time_to_T(feats: np.ndarray, T_target: int) -> np.ndarray:
        F, T_src = feats.shape
        if T_target == T_src:
            return feats
        if T_src <= 1:
            return np.repeat(feats, T_target, axis=1)[:, :T_target]
        src_idx = np.linspace(0, T_src - 1, num=T_target, dtype=np.float32)
        x = np.arange(T_src, dtype=np.float32)
        out = np.empty((F, T_target), dtype=feats.dtype)
        for f in range(F):
            out[f] = np.interp(src_idx, x, feats[f])
        return out

    for wp in wav_files:
        try:
            y, sr = sf.read(wp)
        except Exception as e:
            print(f'[WARN] Failed to read {wp}: {e}')
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        hop_length = args.hop_length if args.hop_length is not None else max(1, int(round(sr / float(args.target_fps))))
        win_length = args.win_length if args.win_length is not None else hop_length * 2
        feats = compute_logmel(y.astype(np.float32), sr, n_mels=args.n_mels, hop_length=hop_length, win_length=win_length, center=args.center)
        # Save as (F, T)
        base = os.path.splitext(os.path.basename(wp))[0]
        outp = os.path.join(args.out_dir, base + '.npy')
        if args.motion_lookup_root:
            mp = find_motion_path(base)
            if mp is None:
                print(f'[WARN] Motion not found for {base}, keep audio T={feats.shape[1]} (sr={sr}, hop={hop_length})')
            else:
                T_motion = load_motion_length(mp)
                if T_motion is not None and T_motion > 0:
                    feats = align_time_to_T(feats, T_motion)
                else:
                    print(f'[WARN] Could not determine motion length for {base} from {mp}')
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
