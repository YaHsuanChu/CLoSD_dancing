"""AIST++ (audio-conditioned) evaluation using external `results.npy`.

This script is intentionally *not* doing text-motion matching metrics.
It focuses on:
- FID (motion embedding)
- Diversity (motion embedding)
- Foot skating / penetration / floating (joints-based heuristics)
- Beat Align Score (audio beats vs foot-contact onsets)

Assumptions
- Motions are stored in `results.npy` as produced by `sample/generate.py`:
  dict with keys {motion, lengths, text, ...}
- Motions are HumanML-style 263-d features in shape (N, 263, 1, T) or (N, T, 263).
- Dataset is `aistpp` and provides raw audio waveforms via `cond['y']['audio']`.

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric
from closd.diffusion_planner.data_loaders.humanml.utils.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_floating,
    calculate_foot_sliding,
    calculate_penetration,
    calculate_skating_ratio,
)
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed


@dataclass
class AudioMeta:
    waveform: np.ndarray  # (samples,)
    sample_rate: int


def _to_263_tn(motion: torch.Tensor) -> torch.Tensor:
    """Convert motion tensor to (B, T, 263) float32.

    Accepts common shapes:
    - (B, 263, 1, T)
    - (B, 263, T)
    - (B, T, 263)
    """
    if motion.dim() == 4:
        # (B, 263, 1, T)
        if motion.shape[1] == 263:
            return motion[:, :, 0, :].permute(0, 2, 1).contiguous()
        # Other 4D layouts are not expected in this repo.
        raise ValueError(f"Unsupported 4D motion shape: {tuple(motion.shape)}")

    if motion.dim() == 3:
        if motion.shape[1] == 263:
            return motion.permute(0, 2, 1).contiguous()
        if motion.shape[2] == 263:
            return motion.contiguous()
        raise ValueError(f"Unsupported 3D motion shape: {tuple(motion.shape)}")

    raise ValueError(f"Unsupported motion dim: {motion.dim()} shape={tuple(motion.shape)}")


def _is_joints_motion(motion: torch.Tensor) -> bool:
    """Heuristic check for joints motion shaped like (B, 22, 3, T)."""
    return motion.dim() == 4 and motion.shape[1] in [21, 22] and motion.shape[2] == 3


def _recover_joints_from_263(
    motion_tn_263: torch.Tensor,
    n_joints: int = 22,
    hml_type=None,
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
) -> torch.Tensor:
    """Recover joints from HumanML 263 features.

    Input: (B, T, 263) => output: (B, n_joints, 3, T)
    
    If mean/std are provided, denormalize the motion first.
    """
    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        # motion_tn_263: (B, T, 263)
        motion_tn_263 = motion_tn_263 * std + mean
    
    # recover_from_ric expects (B, T, 263) (float)
    joints = recover_from_ric(motion_tn_263, n_joints, hml_type)  # (B, T, n_joints, 3)
    joints = joints.permute(0, 2, 3, 1).contiguous()  # (B, n_joints, 3, T)
    return joints


def _load_external_results(path: str, use_gt: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """Load external results from .npy or .pkl file.
    
    Supports:
    - .npy from generate_from_audio.py: keys 'motion', 'gt_motion', 'lengths'
    - .pkl from run.py (CLoSD): keys 'motion', 'length'
    
    Args:
        use_gt: If True, load gt_motion instead of motion (npy files only)
    """
    if path.endswith('.pkl'):
        if use_gt:
            print("[WARN] --eval_gt not supported for pkl files, using motion")
        return _load_external_results_pkl(path)
    else:
        return _load_external_results_npy(path, use_gt=use_gt)


def _load_external_results_pkl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load from CLoSD pkl format (run.py)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    motions = data["motion"]
    # pkl uses 'length' (singular), not 'lengths'
    lengths = data["length"]
    # Convert to numpy if tensor
    if hasattr(motions, 'numpy'):
        motions = motions.numpy()
    if hasattr(lengths, 'numpy'):
        lengths = lengths.numpy()
    return motions, lengths, None  # pkl files don't have paired audio


def _load_external_results_npy(path: str, use_gt: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """Load from generate_from_audio.py npy format.
    
    Args:
        path: Path to npy file
        use_gt: If True, load gt_motion instead of motion (for baseline comparison)
    
    Returns:
        motions, lengths, audio_waveforms (list of {waveform, sample_rate} dicts or None)
    """
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if hasattr(arr, "item") else arr
    
    if use_gt:
        if "gt_motion" in data:
            motions = data["gt_motion"]
            print("[INFO] Loading ground truth motions (gt_motion)")
        else:
            print("[WARN] gt_motion not found in file, using motion instead")
            motions = data["motion"]
    else:
        motions = data["motion"]
    
    lengths = data["lengths"]
    
    # Load paired audio waveforms if available
    audio_waveforms = data.get("audio_waveforms", None)
    if audio_waveforms is not None:
        print(f"[INFO] Loaded {len(audio_waveforms)} paired audio waveforms from npy file")
    
    return motions, lengths, audio_waveforms


def _get_gt_batch_with_audio(
    *,
    device: torch.device,
    num_samples: int,
    fixed_len: int,
    pred_len: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[Dict]]]:
    """Sample a batch from AIST++ dataset and return GT motions + audio meta.

    Returns:
      motion_263: (B, 263, 1, pred_len) when pred_len>0 (prefix mode), else full fixed_len.
      lengths: (B,)
      audio_meta_list: list of dicts (may include waveform/sample_rate/path/...)
    """
    fixseed(seed)
    print("[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...")

    loader = get_dataset_loader(
        name="aistpp",
        batch_size=num_samples,
        num_frames=None,
        split="test",
        hml_mode="train",
        fixed_len=fixed_len,
        pred_len=pred_len,
        device=device,
        drop_last=True,
    )
    print("[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...")
    motion, cond = next(iter(loader))
    print("[DEBUG] _get_gt_batch_with_audio: got batch!")
    y = cond["y"]
    lengths = y["lengths"].to(device)
    audio_meta = y.get("audio", None)
    if audio_meta is None:
        audio_meta = [None for _ in range(num_samples)]
    return motion.to(device), lengths, audio_meta


def _extract_beats_from_waveform(waveform: np.ndarray, sr: int) -> Tuple[np.ndarray, bool]:
    """Return (beat_times_in_seconds, used_librosa).

    Uses librosa if available; otherwise falls back to a simple onset envelope peak picking.
    Returns a tuple: (beat_times, True if librosa was used else False)
    """
    try:
        import librosa  # type: ignore
        
        # Ensure waveform is 1D and float32
        if waveform.ndim > 1:
            waveform = waveform.flatten()
        waveform = waveform.astype(np.float32)
        
        # librosa.beat.beat_track returns (tempo, beat_frames)
        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times.astype(np.float32), True
    except Exception as e:
        # minimal fallback: detect peaks on energy envelope
        x = waveform.astype(np.float32)
        if x.ndim > 1:
            x = x.flatten()
        x = x / (np.max(np.abs(x)) + 1e-8)
        win = int(0.05 * sr)
        if win <= 1:
            win = 1
        env = np.convolve(np.abs(x), np.ones(win, dtype=np.float32) / win, mode="same")
        # pick peaks with a simple threshold
        thr = np.quantile(env, 0.9)
        idx = np.where((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > thr))[0] + 1
        # enforce a refractory period ~200ms
        refractory = int(0.2 * sr)
        keep = []
        last = -10**9
        for i in idx.tolist():
            if i - last >= refractory:
                keep.append(i)
                last = i
        return np.array(keep, dtype=np.float32) / float(sr), False


def _extract_kinematic_beats(joints: np.ndarray, fps: float = 60.0, smooth: bool = False) -> np.ndarray:
    """Extract kinematic beats as local minima of body velocity (AIST++ standard).

    joints: (T, 22, 3) in meters.
    smooth: If True, apply Savitzky-Golay filter to reduce jitter.
    Returns beat times in seconds.
    
    Principle: When dancers 'hit' a beat, they typically pause/freeze momentarily,
    causing body velocity to drop to a local minimum.
    """
    from scipy.signal import argrelextrema, savgol_filter
    
    # Compute velocity: diff of joint positions
    # velocity shape: (T-1, 22, 3)
    velocity = np.diff(joints, axis=0) * fps  # scale by fps to get m/s
    
    # Compute speed per joint: (T-1, 22)
    speed_per_joint = np.linalg.norm(velocity, axis=2)
    
    # Average across all joints to get kinetic speed curve: (T-1,)
    kinetic_speed = speed_per_joint.mean(axis=1)
    
    # Optional: Apply smoothing to reduce high-frequency jitter
    if smooth and len(kinetic_speed) >= 15:
        window_length = min(15, len(kinetic_speed) // 2 * 2 + 1)
        if window_length >= 5:
            kinetic_speed = savgol_filter(kinetic_speed, window_length, polyorder=2)
    
    # Find local minima (valley points in speed curve)
    # order controls minimum distance between peaks
    order = 6 if smooth else 2
    local_min_idx = argrelextrema(kinetic_speed, np.less, order=order)[0]
    
    # Convert frame indices to time (seconds)
    beat_times = local_min_idx.astype(np.float32) / fps
    
    return beat_times


def _beat_align_score(music_beats: np.ndarray, dance_beats: np.ndarray, sigma: float = 0.08) -> float:
    """Calculate Beat Alignment Score (BAS) - AIST++ standard.
    
    Measures the average distance from each dance beat to the nearest music beat.
    This is Precision: how accurately does each dance movement land on a beat?
    (Dancers can skip beats, but when they move, they should be on beat.)
    
    Args:
        music_beats: Times of music beats (seconds)
        dance_beats: Times of kinematic beats / velocity minima (seconds)
        sigma: Gaussian tolerance parameter (default 0.08s = 80ms)
    
    Returns:
        Score in [0, 1], higher is better alignment
    """
    if dance_beats.size == 0:
        # No dance movements detected
        return 0.0
    if music_beats.size == 0:
        # No music beats detected
        return 0.0
    
    # Matrix: [Num_Dance_Beats, Num_Music_Beats]  
    # For each dance beat, compute distance to all music beats
    d = np.abs(dance_beats[:, None] - music_beats[None, :])
    
    # Find the nearest music beat for each dance beat
    # axis=1 means: along Music Beats dimension, keep Dance Beats dimension
    min_dist = d.min(axis=1)
    
    # Gaussian scoring: closer to 0 distance = higher score
    score = np.exp(-(min_dist**2) / (2 * sigma**2)).mean()
    
    return float(score)


def _compute_embeddings(eval_wrapper: EvaluatorMDMWrapper, motion_263_b1t: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """Compute motion embeddings using evaluator wrapper.

    motion_263_b1t: (B, 263, 1, T) or convertible
    Returns: (B, D)
    """
    motion_tn = _to_263_tn(motion_263_b1t)  # (B,T,263)
    with torch.no_grad():
        emb = eval_wrapper.get_motion_embeddings(motion_tn, lengths)
    return emb.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_results_file", type=str, required=True, help="Path to results.npy from sample/generate.py")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--fixed_len", type=int, default=60)
    parser.add_argument("--fps", type=float, default=60.0, help="Motion FPS (AIST++ uses 60fps)")
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--beat_sigma", type=float, default=0.08)
    parser.add_argument("--smooth_kinematic", action="store_true", 
                        help="Apply smoothing filter to kinematic speed before beat detection")
    parser.add_argument("--eval_gt", action="store_true",
                        help="Evaluate ground truth motions instead of generated motions (for baseline)")
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID/Diversity calculation (faster, skips GT dataloader loading)")
    parser.add_argument("--beat_samples", type=int, default=100,
                        help="Number of samples for BeatAlign calculation (default: 100, use -1 for all)")
    args = parser.parse_args()

    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # external motions
    motions_np, lengths_np, paired_audio = _load_external_results(args.external_results_file, use_gt=args.eval_gt)
    motions = torch.tensor(motions_np).float().to(device)
    lengths = torch.tensor(lengths_np).long().to(device)

    # If prefix mode used during generation, lengths may already include prefix.
    # For our AIST++ fixed-length setting, we assume all lengths are fixed_len.
    num_samples = motions.shape[0]
    
    # Detect actual motion length from the loaded data
    # motions shape: (B, T, 263) or (B, 263, 1, T)
    if motions.dim() == 3:
        actual_motion_len = motions.shape[1]
    else:
        actual_motion_len = motions.shape[-1]
    
    # Clamp lengths to actual motion length (PKL files may have original lengths > motion frames)
    if lengths.max() > actual_motion_len:
        print(f"[WARN] lengths max ({lengths.max().item()}) > actual_motion_len ({actual_motion_len}), clamping")
        lengths = lengths.clamp(max=actual_motion_len)
    
    # Determine audio source and whether to load GT data
    gt_motion = None
    gt_lengths = None
    
    if paired_audio is not None:
        # Use paired audio from the npy file (correct for beat alignment)
        print(f"[INFO] Using {len(paired_audio)} paired audio waveforms for beat alignment")
        gt_audio_meta = paired_audio
        # Only load GT motion for FID calculation if not skipping
        if not args.skip_fid:
            audio_fixed_len = actual_motion_len if actual_motion_len > args.fixed_len else args.fixed_len
            gt_motion, gt_lengths, _ = _get_gt_batch_with_audio(
                device=device,
                num_samples=num_samples,
                fixed_len=audio_fixed_len,
                pred_len=audio_fixed_len,
                seed=args.seed,
            )
    else:
        # Fallback: load audio from dataloader (may not be paired)
        print(f"[WARN] No paired audio in npy file, using random audio from dataloader")
        audio_fixed_len = actual_motion_len if actual_motion_len > args.fixed_len else args.fixed_len
        # Limit samples for faster audio loading
        audio_samples = args.beat_samples if args.beat_samples > 0 else num_samples
        audio_samples = min(audio_samples, num_samples)
        print(f"[INFO] Loading {audio_samples} audio samples for BeatAlign (fixed_len={audio_fixed_len})")
        gt_motion, gt_lengths, gt_audio_meta = _get_gt_batch_with_audio(
            device=device,
            num_samples=audio_samples,
            fixed_len=audio_fixed_len,
            pred_len=audio_fixed_len,
            seed=args.seed,
        )
        print("[DEBUG] Dataloader done!")

    fid = None
    diversity = None
    # evaluator wrapper: treat AIST++ as HumanML (263-d)
    # NOTE: If `results.npy` already stores recovered joints (B,22,3,T), the HumanML evaluator
    # expects 263-d representation and cannot be applied directly.
    if _is_joints_motion(motions):
        print(
            "[WARN] External results motion looks like joints (B,22,3,T). "
            "Skipping FID/Diversity (HumanML 263-d evaluator expects (B,T,263))."
        )
    elif args.skip_fid:
        print("[INFO] Skipping FID/Diversity calculation (--skip_fid)")
    else:
        eval_wrapper = EvaluatorMDMWrapper("humanml", device)

        # --- FID / Diversity ---
        gen_emb = _compute_embeddings(eval_wrapper, motions, lengths)
        
        if gt_motion is not None:
            gt_emb = _compute_embeddings(eval_wrapper, gt_motion, gt_lengths)
            gt_mu, gt_cov = calculate_activation_statistics(gt_emb)
            gen_mu, gen_cov = calculate_activation_statistics(gen_emb)
            fid = float(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov))

        # diversity is computed only on generated set
        div_times = min(args.diversity_times, gen_emb.shape[0] - 1)
        diversity = float(calculate_diversity(gen_emb, div_times)) if div_times > 1 else 0.0

    # --- joints-based physics metrics ---
    # Load mean/std for denormalization (AIST++ uses raw motion stats)
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    mean_path = os.path.join(dataset_dir, "aistpp_motion_mean_raw.npy")
    std_path = os.path.join(dataset_dir, "aistpp_motion_std_raw.npy")
    
    motion_mean = None
    motion_std = None
    if os.path.exists(mean_path) and os.path.exists(std_path):
        motion_mean = torch.tensor(np.load(mean_path)).float().to(device)
        motion_std = torch.tensor(np.load(std_path)).float().to(device)
        print(f"[INFO] Loaded motion mean/std from {dataset_dir}")
    else:
        print(f"[WARN] Could not find mean/std files at {dataset_dir}, joints may be incorrect scale")
    
    if _is_joints_motion(motions):
        gen_joints = motions.detach().cpu()
    else:
        gen_263_tn = _to_263_tn(motions)
        gen_joints = _recover_joints_from_263(
            gen_263_tn, n_joints=22, mean=motion_mean, std=motion_std
        ).detach().cpu()  # (B,22,3,T)

    lengths_cpu = lengths.detach().cpu().numpy().tolist()

    # Clamp lengths to the actual temporal length to avoid out-of-bounds in metric helpers.
    # Note: `calculate_foot_sliding` iterates t in range(length-1) and accesses t+1, so it
    # effectively requires length <= T-1.
    T = int(gen_joints.shape[-1])
    lengths_clamped = [min(int(l), T) for l in lengths_cpu]
    lengths_clamped_fs = [min(int(l), max(T - 1, 1)) for l in lengths_cpu]

    skating_ratio, _ = calculate_skating_ratio(gen_joints)
    skating_ratio_mean = float(np.mean(skating_ratio))

    penetration = float(calculate_penetration(gen_joints, lengths_clamped))
    floating = float(calculate_floating(gen_joints, lengths_clamped))
    foot_sliding = float(calculate_foot_sliding(gen_joints, lengths_clamped_fs))

    # --- beat align ---
    beat_scores = []
    n_with_audio = 0
    n_librosa_fallback = 0
    for i in range(min(num_samples, len(gt_audio_meta))):
        meta = gt_audio_meta[i]
        if meta is None:
            continue
        wf = meta.get("waveform", None)
        sr = meta.get("sample_rate", None)
        if wf is None or sr is None:
            continue
        n_with_audio += 1
        if torch.is_tensor(wf):
            wf = wf.detach().cpu().numpy()
        music_beats, used_librosa = _extract_beats_from_waveform(wf, int(sr))
        if not used_librosa:
            n_librosa_fallback += 1

        joints_i = gen_joints[i].permute(2, 0, 1).numpy()  # (T,22,3)
        kinematic_beats = _extract_kinematic_beats(joints_i, fps=args.fps, smooth=args.smooth_kinematic)
        
        # Calculate alignment: how well do kinematic beats align with music beats?
        beat_scores.append(_beat_align_score(music_beats, kinematic_beats, sigma=args.beat_sigma))

    if n_librosa_fallback > 0:
        print(f"[WARN] librosa not used for {n_librosa_fallback}/{n_with_audio} samples (fallback peak detection used)")
    
    beat_align = float(np.mean(beat_scores)) if len(beat_scores) > 0 else 0.0

    print("=== AIST++ external eval ===")
    print(f"external_results_file: {os.path.abspath(args.external_results_file)}")
    print(f"num_samples: {num_samples}")
    if fid is not None:
        print(f"FID: {fid:.4f}")
    else:
        print("FID: <skipped>")
    if diversity is not None:
        print(f"Diversity: {diversity:.4f}")
    else:
        print("Diversity: <skipped>")
    print(f"BeatAlign: {beat_align:.4f} (computed on {len(beat_scores)} samples with audio)")
    print(f"SkatingRatio(mean): {skating_ratio_mean:.4f}")
    print(f"FootSliding(mm): {foot_sliding:.4f}")
    print(f"Penetration(mm): {penetration:.4f}")
    print(f"Floating(mm): {floating:.4f}")


if __name__ == "__main__":
    main()
