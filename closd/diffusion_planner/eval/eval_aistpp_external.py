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


def _recover_joints_from_263(motion_tn_263: torch.Tensor, n_joints: int = 22, hml_type=None) -> torch.Tensor:
    """Recover joints from HumanML 263 features.

    Input: (B, T, 263) => output: (B, n_joints, 3, T)
    """
    # recover_from_ric expects (B, T, 263) (float)
    joints = recover_from_ric(motion_tn_263, n_joints, hml_type)  # (B, T, n_joints, 3)
    joints = joints.permute(0, 2, 3, 1).contiguous()  # (B, n_joints, 3, T)
    return joints


def _load_external_results(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if hasattr(arr, "item") else arr
    motions = data["motion"]
    lengths = data["lengths"]
    return motions, lengths


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
    motion, cond = next(iter(loader))
    y = cond["y"]
    lengths = y["lengths"].to(device)
    audio_meta = y.get("audio", None)
    if audio_meta is None:
        audio_meta = [None for _ in range(num_samples)]
    return motion.to(device), lengths, audio_meta


def _extract_beats_from_waveform(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Return beat times in seconds.

    Uses librosa if available; otherwise falls back to a simple onset envelope peak picking.
    """
    try:
        import librosa  # type: ignore

        tempo, beats = librosa.beat.beat_track(y=waveform.astype(np.float32), sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times
    except Exception:
        # minimal fallback: detect peaks on energy envelope
        x = waveform.astype(np.float32)
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
        return np.array(keep, dtype=np.float32) / float(sr)


def _motion_contact_onsets(joints: np.ndarray, fps: float = 20.0) -> np.ndarray:
    """Detect foot-contact onsets from joints.

    joints: (T, 22, 3) in meters (?) y-up.
    Returns onset times in seconds.
    """
    # HumanML convention in metrics.py assumes foot joints 10/11
    foot_idx = [10, 11]
    y = joints[:, foot_idx, 1]  # (T,2)
    vxz = np.linalg.norm(joints[1:, foot_idx][:, :, [0, 2]] - joints[:-1, foot_idx][:, :, [0, 2]], axis=-1) * fps

    # contact when low and slow
    height_thr = 0.05
    vel_thr = 0.2
    contact = (y[:-1] < height_thr) & (y[1:] < height_thr) & (vxz < vel_thr)
    contact_any = contact[:, 0] | contact[:, 1]  # (T-1,)

    # onset: rising edge
    onset = np.where((~contact_any[:-1]) & (contact_any[1:]))[0] + 1
    onset_t = onset.astype(np.float32) / fps
    return onset_t


def _beat_align_score(beat_times: np.ndarray, onset_times: np.ndarray, sigma: float = 0.08) -> float:
    """Gaussian kernel alignment between beat times and motion onset times."""
    if beat_times.size == 0 or onset_times.size == 0:
        return 0.0
    # for each beat, distance to nearest onset
    d = np.abs(beat_times[:, None] - onset_times[None, :])
    dmin = d.min(axis=1)
    score = np.exp(-(dmin**2) / (2 * sigma**2)).mean()
    return float(score)


def _compute_embeddings(eval_wrapper: EvaluatorMDMWrapper, motion_263_b1t: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """Compute motion embeddings using evaluator wrapper.

    motion_263_b1t: (B, 263, 1, T) or convertible
    Returns: (B, D)
    """
    motion_tn = _to_263_tn(motion_263_b1t)  # (B,T,263)
    # evaluator expects something it can do motions[..., :-4] on; (B,T,263) works.
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
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--diversity_times", type=int, default=300)
    parser.add_argument("--beat_sigma", type=float, default=0.08)
    args = parser.parse_args()

    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # external motions
    motions_np, lengths_np = _load_external_results(args.external_results_file)
    motions = torch.tensor(motions_np).float().to(device)
    lengths = torch.tensor(lengths_np).long().to(device)

    # If prefix mode used during generation, lengths may already include prefix.
    # For our AIST++ fixed-length setting, we assume all lengths are fixed_len.
    num_samples = motions.shape[0]

    # build GT batch (same sample count, but not index-aligned; distribution-aligned)
    gt_motion, gt_lengths, gt_audio_meta = _get_gt_batch_with_audio(
        device=device,
        num_samples=num_samples,
        fixed_len=args.fixed_len,
        pred_len=args.pred_len,
        seed=args.seed,
    )

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
    else:
        eval_wrapper = EvaluatorMDMWrapper("humanml", device)

        # --- FID / Diversity ---
        gen_emb = _compute_embeddings(eval_wrapper, motions, lengths)
        gt_emb = _compute_embeddings(eval_wrapper, gt_motion, gt_lengths)

        gt_mu, gt_cov = calculate_activation_statistics(gt_emb)
        gen_mu, gen_cov = calculate_activation_statistics(gen_emb)
        fid = float(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov))

        # diversity is computed only on generated set
        div_times = min(args.diversity_times, gen_emb.shape[0] - 1)
        diversity = float(calculate_diversity(gen_emb, div_times)) if div_times > 1 else 0.0

    # --- joints-based physics metrics ---
    if _is_joints_motion(motions):
        gen_joints = motions.detach().cpu()
    else:
        gen_263_tn = _to_263_tn(motions)
        gen_joints = _recover_joints_from_263(gen_263_tn, n_joints=22).detach().cpu()  # (B,22,3,T)

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
    # use GT audio meta as the conditioning reference (fixed_len audio)
    for i in range(min(num_samples, len(gt_audio_meta))):
        meta = gt_audio_meta[i]
        if meta is None:
            continue
        wf = meta.get("waveform", None)
        sr = meta.get("sample_rate", None)
        if wf is None or sr is None:
            continue
        if torch.is_tensor(wf):
            wf = wf.detach().cpu().numpy()
        beat_t = _extract_beats_from_waveform(wf, int(sr))

        joints_i = gen_joints[i].permute(2, 0, 1).numpy()  # (T,22,3)
        onset_t = _motion_contact_onsets(joints_i, fps=args.fps)
        beat_scores.append(_beat_align_score(beat_t, onset_t, sigma=args.beat_sigma))

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
