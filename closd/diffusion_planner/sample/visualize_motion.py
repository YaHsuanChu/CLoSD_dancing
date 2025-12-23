"""Visualize generated motions from npy files as MP4 videos with audio.

Usage:
    python visualize_motion.py --npy_path output/sample/generated_motions.npy \
        --output_dir output/visualizations --num_samples 5
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

import closd.diffusion_planner.data_loaders.humanml.utils.paramUtil as paramUtil
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric
from closd.diffusion_planner.data_loaders.humanml.utils.plot_script import plot_3d_motion


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generated motions as MP4")
    parser.add_argument("--npy_path", type=str, required=True,
                        help="Path to generated_motions.npy")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for videos")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="FPS for video (default: 60 for AIST++)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting sample index")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print(f"Loading {args.npy_path}...")
    data = np.load(args.npy_path, allow_pickle=True).item()
    
    motions = data.get("motion", None)
    gt_motions = data.get("gt_motion", None)
    lengths = data.get("lengths", None)
    audio_waveforms = data.get("audio_waveforms", None)
    
    print(f"Motion shape: {motions.shape}")
    print(f"Number of audio waveforms: {len(audio_waveforms) if audio_waveforms else 0}")
    
    # Setup output directory
    if args.output_dir == "":
        args.output_dir = os.path.join(os.path.dirname(args.npy_path), "visualizations")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Skeleton for HumanML/AIST++ (22 joints)
    kinematic_chain = paramUtil.t2m_kinematic_chain
    
    # Load dataset for inv_transform
    print("Loading dataset for denormalization...")
    from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
    data_loader = get_dataset_loader(
        name="aistpp", 
        batch_size=1, 
        num_frames=None,
        split="test",
        hml_mode="train",
    )
    dataset = data_loader.dataset
    
    # Visualize samples
    end_idx = min(args.start_idx + args.num_samples, motions.shape[0])
    
    for i in tqdm(range(args.start_idx, end_idx), desc="Rendering"):
        motion = motions[i]  # (T, 263) or (263, 1, T)
        
        # Convert to (1, T, 263) for processing
        if motion.ndim == 3:
            if motion.shape[0] == 263:
                motion = motion.squeeze(1).T  # (263, 1, T) -> (T, 263)
            else:
                motion = motion.squeeze(1)  # (T, 1, 263) -> (T, 263)
        elif motion.shape[0] == 263:
            motion = motion.T  # (263, T) -> (T, 263)
        
        n_frames = motion.shape[0]
        
        # Shape for inv_transform: (B, 1, T, 263) -> permute(0,2,3,1) -> (B, T, 263, 1) -> squeeze
        # Actually generate.py does: sample.cpu().permute(0, 2, 3, 1) on (B, 263, 1, T) -> (B, T, 1, 263)
        # Then inv_transform expects (B, T, 1, 263) or (B, T, 263)?
        
        # Let's follow generate.py exactly:
        # motion shape in generate.py before inv_transform: (B, 263, 1, T)
        # After permute(0, 2, 3, 1): (B, 1, T, 263)
        motion_t = torch.tensor(motion).unsqueeze(0).unsqueeze(1).float()  # (1, 1, T, 263)
        
        # Apply inv_transform (denormalization)
        try:
            denormed = dataset.t2m_dataset.inv_transform(motion_t).float()
            # denormed shape: (1, 1, T, 263)
        except Exception as e:
            print(f"[WARN] inv_transform failed for sample {i}: {e}, using raw motion")
            denormed = motion_t
        
        # Recover joints: expects (B, T, 263) -> (B, T, 22, 3)
        denormed = denormed.squeeze(1)  # (1, T, 263)
        joints = recover_from_ric(denormed, 22).numpy()[0]  # (T, 22, 3)
        
        # Create video
        video_path = os.path.join(args.output_dir, f"sample_{i:04d}.mp4")
        title = f"Sample {i}"
        
        ani = plot_3d_motion(
            save_path=video_path,
            kinematic_tree=kinematic_chain,
            joints=joints,
            title=title,
            dataset="aistpp",
            fps=args.fps,
        )
        
        # Save video (without audio first)
        temp_video_path = os.path.join(args.output_dir, f"sample_{i:04d}_temp.mp4")
        ani.write_videofile(temp_video_path, fps=args.fps, codec='libx264', audio=False, logger=None)
        
        # Combine with audio if available
        if audio_waveforms is not None and i < len(audio_waveforms):
            audio_meta = audio_waveforms[i]
            if audio_meta is not None:
                waveform = audio_meta.get("waveform", None)
                sample_rate = audio_meta.get("sample_rate", None)
                
                if waveform is not None and sample_rate is not None:
                    # Convert waveform to audio file
                    import scipy.io.wavfile as wavfile
                    
                    if hasattr(waveform, 'numpy'):
                        waveform = waveform.numpy()
                    
                    audio_path = os.path.join(args.output_dir, f"sample_{i:04d}_audio.wav")
                    
                    # Ensure waveform is 1D and in correct range
                    if waveform.ndim > 1:
                        waveform = waveform.flatten()
                    
                    # Normalize to int16
                    waveform_int16 = (waveform * 32767).astype(np.int16)
                    wavfile.write(audio_path, int(sample_rate), waveform_int16)
                    
                    # Combine video and audio using ffmpeg
                    final_path = video_path
                    os.system(
                        f'ffmpeg -y -i {temp_video_path} -i {audio_path} -c:v copy -c:a aac '
                        f'-shortest {final_path} -loglevel quiet'
                    )
                    
                    # Cleanup temp files
                    os.remove(temp_video_path)
                    os.remove(audio_path)
                    
                    print(f"[OK] Saved {video_path} (with audio)")
                    continue
        
        # If no audio, just rename temp to final
        os.rename(temp_video_path, video_path)
        print(f"[OK] Saved {video_path} (no audio)")
    
    print(f"\nDone! Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
