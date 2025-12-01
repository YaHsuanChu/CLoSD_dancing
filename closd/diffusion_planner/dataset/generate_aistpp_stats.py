import os
import argparse
import numpy as np
import torch
import pickle
import soundfile as sf

try:
    # HumanML feature extraction (returns (bs, T-1, 263))
    from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m
except ImportError:
    extract_features_t2m = None  # Handle gracefully; user can still compute raw stats

# Optional SMPL for joint reconstruction from edge_aistpp .pkl
try:
    from closd.utils.smpllib.smpllib.smpl_parser import SMPL_Parser
except Exception:
    SMPL_Parser = None

"""Generate mean/std statistics for AIST++ motion and audio features.

Supports two representations:
1. raw       : flattened joint coordinates (T, J*3) or (T, D)
2. humanml   : converted to HumanML (SMPL-derived) 263-dim per-frame features using extract_features_t2m

Motion source: EDGE-main/data/{train,test}/motions_sliced/*.npy
Audio source : EDGE-main/data/{train,test}/{baseline_feats|jukebox_feats}/*.npy

Outputs (default names):
 - aistpp_motion_mean.npy / aistpp_motion_std.npy (raw or humanml depending on --representation)
 - aistpp_audio_mean.npy  / aistpp_audio_std.npy

Notes:
 - HumanML conversion drops the last frame (features computed for T-1 steps).
 - When using humanml representation, stats must match the training loader that sets remap_joints=true.
"""


def collect_files(dir_path: str, exts=(".npy", ".pkl")):
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if any(f.endswith(e) for e in exts)]
    files.sort()
    return files


def load_motion_raw(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Try common structures
        if isinstance(obj, dict):
            # Prefer explicit joint arrays; else build flat features from SMPL params
            for key in ['joints', 'joints3d', 'positions', 'motion']:
                if key in obj:
                    arr = np.array(obj[key])
                    break
            else:
                poses = np.array(obj.get('smpl_poses')) if 'smpl_poses' in obj else None
                trans = np.array(obj.get('smpl_trans')) if 'smpl_trans' in obj else None
                if poses is not None and trans is not None:
                    T = poses.shape[0]
                    poses_flat = poses.reshape(T, -1)
                    arr = np.concatenate([poses_flat, trans.reshape(T, -1)], axis=1)
                elif poses is not None:
                    T = poses.shape[0]
                    arr = poses.reshape(T, -1)
                else:
                    # Fallback to first value
                    arr = np.array(list(obj.values())[0]) if len(obj) > 0 else np.array([])
        elif isinstance(obj, (list, tuple)):
            arr = np.array(obj)
        else:
            arr = np.array(obj)
    else:
        raise ValueError(f"Unsupported motion file {path}")
    if arr.ndim == 3:  # (T, J, C)
        T, J, C = arr.shape
        arr = arr.reshape(T, J * C)
    elif arr.ndim == 2:  # (T, D)
        pass
    else:
        raise ValueError(f"Unexpected motion shape {arr.shape} in {path}")
    return arr.astype(np.float32)

def convert_motion_humanml(raw_flat: np.ndarray) -> np.ndarray:
    """raw_flat: (T, J*3) -> features (T-1, 263) using extract_features_t2m."""
    if extract_features_t2m is None:
        raise RuntimeError("HumanML conversion requested but extract_features_t2m import failed.")
    T, D = raw_flat.shape
    J = D // 3
    pos = raw_flat.reshape(1, T, J, 3).astype(np.float32)  # (1, T, J, 3)
    feats, _ = extract_features_t2m(torch.from_numpy(pos))  # (1, T-1, 263)
    return feats.numpy()[0]


def load_audio_array(path: str) -> np.ndarray:
    arr = np.load(path)
    # normalize to 2D (F, T_a) or 1D (F,)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected audio shape {arr.shape} in {path}")
    return arr.astype(np.float32)


def compute_motion_stats(motion_dirs, representation="raw"):
    sums = None
    sumsq = None
    count = 0
    for d in motion_dirs:
        for fp in collect_files(d):
            raw = load_motion_raw(fp)  # (T, D_raw)
            if representation == "humanml":
                m = convert_motion_humanml(raw)  # (T-1, 263)
            else:
                m = raw  # (T, D_raw)
            if sums is None:
                sums = np.zeros(m.shape[1], dtype=np.float64)
                sumsq = np.zeros(m.shape[1], dtype=np.float64)
            sums += m.sum(axis=0)
            sumsq += (m ** 2).sum(axis=0)
            count += m.shape[0]
    if count == 0:
        raise RuntimeError("No motion files found for stats")
    mean = (sums / count).astype(np.float32)
    var = (sumsq / count) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


def smpl_pkl_to_joints(obj, smpl_model_dir):
    """Derive joints [T, J, 3] from SMPL params in an EDGE AIST++ .pkl dict.
    Requires SMPL_Parser and torch. Uses neutral model by default.
    """
    if SMPL_Parser is None or torch is None:
        raise RuntimeError("SMPL_Parser/torch not available for SMPL-to-joints conversion.")
    poses = np.array(obj.get('smpl_poses')) if 'smpl_poses' in obj else None
    trans = np.array(obj.get('smpl_trans')) if 'smpl_trans' in obj else None
    if poses is None or trans is None:
        raise RuntimeError("Expected 'smpl_poses' and 'smpl_trans' in .pkl for SMPL conversion.")
    # poses: [T, 24*3] axis-angle; trans: [T, 3]
    T = poses.shape[0]
    smpl = SMPL_Parser(model_path=smpl_model_dir, gender="neutral")
    # SMPL_Parser expects torch tensors
    pose_tensor = torch.from_numpy(poses).float().view(T, 24, 3)
    trans_tensor = torch.from_numpy(trans).float()
    # Forward per frame to get joints. Batch in chunks to reduce memory.
    joints_list = []
    chunk = 256
    for i in range(0, T, chunk):
        p = pose_tensor[i:i+chunk]
        t = trans_tensor[i:i+chunk]
        out = smpl.forward(pose_body=p[:, 1:], pose_root=p[:, 0], trans=t)  # parser API: pose_root (3), pose_body (23x3)
        # out.joints: [B, J, 3]
        joints_list.append(out.joints.detach().cpu().numpy())
    joints = np.concatenate(joints_list, axis=0)
    return joints  # [T, J, 3]


def compute_audio_stats(audio_dirs):
    sums = None
    sumsq = None
    count = 0
    for d in audio_dirs:
        for fp in collect_files(d):
            a = load_audio_array(fp)  # (F,) or (F, T_a)
            if a.ndim == 2:
                # sum over time axis
                # accumulate per-feature
                if sums is None:
                    sums = np.zeros(a.shape[0], dtype=np.float64)
                    sumsq = np.zeros(a.shape[0], dtype=np.float64)
                sums += a.sum(axis=1)
                sumsq += (a ** 2).sum(axis=1)
                count += a.shape[1]
            else:  # 1D
                if sums is None:
                    sums = np.zeros(a.shape[0], dtype=np.float64)
                    sumsq = np.zeros(a.shape[0], dtype=np.float64)
                sums += a
                sumsq += (a ** 2)
                count += 1
    if count == 0:
        raise RuntimeError("No audio feature files found for stats")
    mean = (sums / count).astype(np.float32)
    var = (sumsq / count) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_root", type=str,
                        help="Path to EDGE-main/data directory, e.g. /path/to/EDGE-main/data")
    parser.add_argument("--motion_dir", type=str, default=None,
                        help="Optional: explicit motion dir with .npy/.pkl to compute stats from (bypasses edge_root)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Optional: explicit audio feature dir with .npy to compute stats from (bypasses edge_root)")
    parser.add_argument("--edge_aistpp_root", type=str, default=None,
                        help="Alternative root to raw AIST++ under EDGE (e.g. /.../EDGE-main/data/edge_aistpp). If set, stats will be computed from here instead of train/test.")
    parser.add_argument("--audio_type", type=str, default="baseline_feats",
                        choices=["baseline_feats", "jukebox_feats"],
                        help="Which audio features directory to use")
    parser.add_argument("--out_dir", type=str, default=os.path.dirname(__file__),
                        help="Where to write the stats .npy files")
    parser.add_argument("--representation", type=str, default="humanml", choices=["raw", "humanml"],
                        help="Motion feature representation for stats. Use 'humanml' if remap_joints=true during training.")
    parser.add_argument("--both", action="store_true",
                        help="If set, compute and save stats for BOTH raw and humanml in one run.")
    parser.add_argument("--smpl_model_dir", type=str, default=None,
                        help="Directory containing SMPL model files. Required to compute humanml from edge_aistpp .pkl")
    args = parser.parse_args()

    # Resolve motion/audio directories either from explicit overrides, edge_aistpp root, or split dirs
    if args.motion_dir is not None:
        motion_dirs = [args.motion_dir]
        audio_dirs = []
        has_wavs_only = False
    elif args.edge_aistpp_root:
        # Use raw AIST++ location
        motion_dirs = [
            os.path.join(args.edge_aistpp_root, "motions"),
        ]
        # Prefer precomputed feature npy if available; else, try baseline_feats
        audio_dirs = []
        for name in (args.audio_type, "baseline_feats"):
            cand = os.path.join(args.edge_aistpp_root, name)
            if os.path.isdir(cand):
                audio_dirs.append(cand)
                break
        # If only wavs exist, we cannot compute npy-based stats here; skip with warning later
        wav_dir = os.path.join(args.edge_aistpp_root, "wavs")
        has_wavs_only = os.path.isdir(wav_dir) and not audio_dirs
        wav_dirs = [wav_dir] if os.path.isdir(wav_dir) else []
    else:
        # Use split dirs produced by EDGE preprocessing
        train_motion_dir = os.path.join(args.edge_root, "train", "motions_sliced")
        test_motion_dir = os.path.join(args.edge_root, "test", "motions_sliced")
        motion_dirs = [train_motion_dir, test_motion_dir]

        train_audio_dir = os.path.join(args.edge_root, "train", args.audio_type)
        test_audio_dir = os.path.join(args.edge_root, "test", args.audio_type)
        audio_dirs = [train_audio_dir, test_audio_dir]
        has_wavs_only = False

    if args.audio_dir is not None:
        audio_dirs = [args.audio_dir]

    motion_stats = {}
    if args.both:
        motion_stats['raw'] = compute_motion_stats(motion_dirs, representation='raw')
        try:
            if args.edge_aistpp_root:
                # Compute humanml by reconstructing joints from SMPL if needed
                if args.smpl_model_dir is None:
                    raise RuntimeError("--smpl_model_dir is required to compute HumanML from edge_aistpp .pkl")
                # Build a temporary motion dir list of reconstructed joints in-memory
                sums = None; sumsq = None; count = 0
                sample_dir = motion_dirs[0]
                files = [f for f in os.listdir(sample_dir) if f.endswith('.pkl') or f.endswith('.npy')]
                files.sort()
                if not files:
                    raise RuntimeError("No motion files found under edge_aistpp/motions")
                for name in files:
                    fp = os.path.join(sample_dir, name)
                    if fp.endswith('.pkl'):
                        with open(fp, 'rb') as f:
                            obj = pickle.load(f)
                        joints = smpl_pkl_to_joints(obj, args.smpl_model_dir)  # [T, J, 3]
                        raw_flat = joints.reshape(joints.shape[0], -1).astype(np.float32)
                    else:
                        arr = np.load(fp, allow_pickle=True)
                        raw_flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
                    m = convert_motion_humanml(raw_flat)  # (T-1, 263)
                    if sums is None:
                        sums = np.zeros(m.shape[1], dtype=np.float64)
                        sumsq = np.zeros(m.shape[1], dtype=np.float64)
                    sums += m.sum(axis=0)
                    sumsq += (m ** 2).sum(axis=0)
                    count += m.shape[0]
                if count == 0:
                    raise RuntimeError("No frames aggregated for HumanML stats")
                mean = (sums / count).astype(np.float32)
                var = (sumsq / count) - (mean.astype(np.float64) ** 2)
                std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
                motion_stats['humanml'] = (mean, std)
            else:
                motion_stats['humanml'] = compute_motion_stats(motion_dirs, representation='humanml')
        except Exception as e:
            print(f"[WARN] HumanML stats skipped: {e}")
    else:
        if args.representation == 'humanml' and args.edge_aistpp_root:
            if args.smpl_model_dir is None:
                raise RuntimeError("--smpl_model_dir is required to compute HumanML from edge_aistpp .pkl")
            # Reuse reconstruction loop
            sums = None; sumsq = None; count = 0
            sample_dir = motion_dirs[0]
            files = [f for f in os.listdir(sample_dir) if f.endswith('.pkl') or f.endswith('.npy')]
            files.sort()
            if not files:
                raise RuntimeError("No motion files found under edge_aistpp/motions")
            for name in files:
                fp = os.path.join(sample_dir, name)
                if fp.endswith('.pkl'):
                    with open(fp, 'rb') as f:
                        obj = pickle.load(f)
                    joints = smpl_pkl_to_joints(obj, args.smpl_model_dir)  # [T, J, 3]
                    raw_flat = joints.reshape(joints.shape[0], -1).astype(np.float32)
                else:
                    arr = np.load(fp, allow_pickle=True)
                    raw_flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
                m = convert_motion_humanml(raw_flat)  # (T-1, 263)
                if sums is None:
                    sums = np.zeros(m.shape[1], dtype=np.float64)
                    sumsq = np.zeros(m.shape[1], dtype=np.float64)
                sums += m.sum(axis=0)
                sumsq += (m ** 2).sum(axis=0)
                count += m.shape[0]
            if count == 0:
                raise RuntimeError("No frames aggregated for HumanML stats")
            mean = (sums / count).astype(np.float32)
            var = (sumsq / count) - (mean.astype(np.float64) ** 2)
            std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
            motion_stats['humanml'] = (mean, std)
        else:
            motion_stats[args.representation] = compute_motion_stats(motion_dirs, representation=args.representation)

    if audio_dirs:
        audio_mean, audio_std = compute_audio_stats(audio_dirs)
    else:
        audio_mean, audio_std = None, None

    os.makedirs(args.out_dir, exist_ok=True)
    saved_motion_paths = []
    for rep, (motion_mean, motion_std) in motion_stats.items():
        motion_mean_path = os.path.join(args.out_dir, f"aistpp_motion_mean_{rep}.npy")
        motion_std_path = os.path.join(args.out_dir, f"aistpp_motion_std_{rep}.npy")
        np.save(motion_mean_path, motion_mean)
        np.save(motion_std_path, motion_std)
        saved_motion_paths.extend([motion_mean_path, motion_std_path])
    audio_mean_path = os.path.join(args.out_dir, "aistpp_audio_mean.npy")
    audio_std_path = os.path.join(args.out_dir, "aistpp_audio_std.npy")
    if audio_mean is not None and audio_std is not None:
        np.save(audio_mean_path, audio_mean)
        np.save(audio_std_path, audio_std)
    elif args.edge_aistpp_root and 'wav_dirs' in locals() and wav_dirs:
        # Fallback: compute simple waveform stats if only wavs are present
        waves = []
        for d in wav_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith('.wav'):
                    fp = os.path.join(d, f)
                    try:
                        data, sr = sf.read(fp)
                    except Exception:
                        continue
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    waves.append(data.astype(np.float32))
        if waves:
            cat = np.concatenate(waves)
            audio_mean = np.array([cat.mean()], dtype=np.float32)
            audio_std = np.array([cat.std() + 1e-8], dtype=np.float32)
            np.save(audio_mean_path, audio_mean)
            np.save(audio_std_path, audio_std)

    print("Saved:")
    for p in saved_motion_paths:
        print(p)
    if audio_mean is not None:
        print(audio_mean_path)
        print(audio_std_path)
    elif has_wavs_only:
        print("[INFO] Only wavs found under edge_aistpp; used waveform stats fallback.")


if __name__ == "__main__":
    main()
