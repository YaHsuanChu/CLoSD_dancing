# Generate motions from audio samples using the diffusion model.
# This script samples 300 random audio clips from the dataloader, 
# generates corresponding motions, and saves them in 263-dim HumanML format.

from closd.diffusion_planner.utils.fixseed import fixseed
import os
import numpy as np
import torch
from closd.diffusion_planner.utils.parser_util import generate_args
from closd.diffusion_planner.utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
)
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.sampler_util import (
    ClassifierFreeSampleModel,
    AutoRegressiveSampler,
)
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.data_loaders import humanml_utils
import argparse
from tqdm import tqdm


def parse_args():
    """Parse command line arguments for audio-to-motion generation."""
    parser = argparse.ArgumentParser(description="Generate motions from audio samples")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="aistpp",
                        help="Dataset name (default: aistpp)")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="Number of audio samples to generate motions for (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation (default: 32)")
    parser.add_argument("--output_path", type=str, default="",
                        help="Output path for the generated motions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--n_frames", type=int, default=196,
                        help="Number of frames to generate (default: 196)")
    
    # Generation parameters
    parser.add_argument("--guidance_param", type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model weights")
    
    # Prefix completion parameters
    parser.add_argument("--context_len", type=int, default=30,
                        help="Context length for prefix completion (default: 30)")
    parser.add_argument("--pred_len", type=int, default=60,
                        help="Prediction length (default: 60)")
    
    # Audio concat mode
    parser.add_argument("--audio_concat_mode", type=str, default="none",
                        choices=["none", "concat"],
                        help="Audio concatenation mode (default: none)")
    
    # Model architecture arguments (for xattn models)
    parser.add_argument("--arch", type=str, default="trans_enc",
                        choices=["trans_enc", "trans_dec", "gru"],
                        help="Architecture type (default: trans_enc)")
    parser.add_argument("--text_encoder_type", type=str, default="clip",
                        choices=["clip", "bert", "none"],
                        help="Text encoder type (default: clip)")
    parser.add_argument("--per_frame_audio_xatten", action="store_true",
                        help="Use per-frame audio tokens as cross-attention memory")
    parser.add_argument("--text_uncond_all", action="store_true",
                        help="Set text_uncond=True for all batches (useful for pure audio conditioning)")
    
    return parser.parse_args()


def load_model_args(model_path):
    """Load the model's original training arguments from args.json."""
    import json
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            return json.load(f)
    return {}


def main():
    args = parse_args()
    fixseed(args.seed)
    
    # Setup distributed (single GPU)
    # Extract device ID as integer (e.g., "cuda:0" -> 0, "0" -> 0)
    if isinstance(args.device, str):
        if "cuda:" in args.device:
            device_id = int(args.device.split(":")[-1])
        elif args.device.isdigit():
            device_id = int(args.device)
        else:
            device_id = 0
    else:
        device_id = int(args.device)
    dist_util.setup_dist(device_id)
    
    # Load model training args - these take priority for model architecture
    model_train_args = load_model_args(args.model_path)
    
    # Create a namespace object that combines our args with model args
    class CombinedArgs:
        def __init__(self, cli_args, model_args):
            # First set CLI args as defaults
            for k, v in vars(cli_args).items():
                setattr(self, k, v)
            # Then override with model training args (model architecture must match)
            for k, v in model_args.items():
                setattr(self, k, v)
            # Allow some CLI overrides that don't affect model architecture
            self.num_samples = cli_args.num_samples
            self.batch_size = cli_args.batch_size
            self.output_path = cli_args.output_path
            self.seed = cli_args.seed
            self.guidance_param = cli_args.guidance_param
    
    full_args = CombinedArgs(args, model_train_args)
    
    # Ensure required attributes exist
    if not hasattr(full_args, 'hml_type'):
        full_args.hml_type = None
    if not hasattr(full_args, 'autoregressive'):
        full_args.autoregressive = False
    if not hasattr(full_args, 'audio_concat_mode'):
        full_args.audio_concat_mode = "none"
    if not hasattr(full_args, 'audio_dim'):
        full_args.audio_dim = 80
    
    # Use model's context_len
    context_len = getattr(full_args, 'context_len', 20)
    
    # Determine number of frames to generate
    # If using autoregressive, we can generate more than 196
    # If not using autoregressive, we are limited by model's max seq len (usually 196)
    
    n_frames_requested = args.n_frames
    
    # For dataloading, we need to know the total length (including context if needed, 
    # but usually loader just wants target length)
    # The existing code used fixed_len = max_frames
    
    # n_frames is the prediction part
    n_frames = n_frames_requested 
    
    # Calculate max_frames passed to loader
    # This should be enough to cover the requested n_frames
    max_frames = n_frames + context_len # Add context just in case loader needs it
    pred_len = n_frames # This is what we want to predict
    
    print(f"Loading dataset [{full_args.dataset}]...")
    print(f"Context length: {context_len}, Prediction length (generated): {pred_len}")
    print(f"Audio concat mode: {full_args.audio_concat_mode}")
    
    # Load dataset with prefix mode
    data = get_dataset_loader(
        name=full_args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames, # Request enough frames from dataset
        split="test",
        hml_mode="train",  # Need full motion data
        hml_type=full_args.hml_type,
        pred_len=pred_len,
        fixed_len=max_frames, # Using variable length, but passing max for buffer
        device=dist_util.dev(),
    )
    
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(full_args, data)
    
    sample_fn = diffusion.p_sample_loop
    if getattr(full_args, 'autoregressive', False):
        sample_cls = AutoRegressiveSampler(full_args, sample_fn, n_frames)
        sample_fn = sample_cls.sample
    
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()
    
    # Motion shape: (batch_size, n_joints, n_feats=1, n_frames)
    # For concat mode, the model expects motion_dim + audio_dim
    motion_dim = 263 if full_args.dataset in ["humanml", "aistpp"] else 251
    if full_args.audio_concat_mode == "concat":
        input_dim = motion_dim + full_args.audio_dim  # 263 + 80 = 343
    else:
        input_dim = motion_dim
    motion_shape = (args.batch_size, input_dim, 1, n_frames)
    
    # Collect samples
    all_motions = []
    all_gt_motions = []  # Ground truth for comparison
    all_lengths = []
    all_audio_embeds = []
    all_audio_waveforms = []  # Raw audio waveforms for beat alignment
    
    n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    data_iter = iter(data)
    
    print(f"Generating {args.num_samples} samples in {n_batches} batches...")
    
    collected = 0
    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        try:
            input_motion, model_kwargs = next(data_iter)
        except StopIteration:
            # Reset iterator if we run out of data
            data_iter = iter(data)
            input_motion, model_kwargs = next(data_iter)
        
        # Move to device
        input_motion = input_motion.to(dist_util.dev())
        model_kwargs["y"] = {
            key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        
        # Handle audio concat mode
        if full_args.audio_concat_mode == "concat":
            y = model_kwargs["y"]
            audio_prefix = y.get("audio_embed_prefix", None)
            
            if "prefix" in y and audio_prefix is not None:
                prefix_motion = y["prefix"]
                T_prefix = prefix_motion.shape[-1]
                
                audio_prefix_tf = audio_prefix.permute(0, 2, 1)
                Bp, T_ap, F_ap = audio_prefix_tf.shape
                
                if T_ap >= T_prefix:
                    audio_pref_trim = audio_prefix_tf[:, :T_prefix, :]
                else:
                    pad_len = T_prefix - T_ap
                    pad = torch.zeros(Bp, pad_len, F_ap, 
                                      device=audio_prefix.device, 
                                      dtype=audio_prefix.dtype)
                    audio_pref_trim = torch.cat([audio_prefix_tf, pad], dim=1)
                
                audio_feat_prefix = audio_pref_trim.permute(0, 2, 1).unsqueeze(2)
                y["prefix"] = torch.cat([prefix_motion, audio_feat_prefix], dim=1)
        
        # Add CFG scale
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            )
        
        # Generate samples
        with torch.no_grad():
            sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        
        # Extract only motion dimensions (remove audio channels if concat mode)
        if sample.shape[1] > motion_dim:
            sample = sample[:, :motion_dim, ...]
        
        # Store generated motions in HumanML 263-dim format
        # sample shape: (B, 263, 1, T_total) where T_total might include context
        # We only want the last pred_len frames (predicted part, not prefix)
        sample = sample[..., -n_frames:]  # Slice to get only pred part
        sample_hml = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()  # (B, 196, 263)
        
        # Get the ground truth motion for reference (pred part only)
        # input_motion is the pred part from dataloader, slice to match
        gt_motion_tensor = input_motion[..., -n_frames:] if input_motion.shape[-1] > n_frames else input_motion
        gt_motion = gt_motion_tensor.squeeze(2).permute(0, 2, 1).cpu().numpy()  # (B, 196, 263)
        
        # Get lengths - this is the prediction length
        lengths = np.full(sample_hml.shape[0], n_frames, dtype=np.int64)  # All have same pred length
        
        # Collect audio embeddings if available
        audio_embed_pred = model_kwargs["y"].get("audio_embed_pred", None)
        if audio_embed_pred is not None:
            all_audio_embeds.append(audio_embed_pred.cpu().numpy())
        
        # Collect raw audio waveforms for beat alignment
        audio_meta_list = model_kwargs["y"].get("audio", None)
        if audio_meta_list is not None:
            batch_waveforms = []
            for audio_meta in audio_meta_list:
                if audio_meta is not None:
                    wf = audio_meta.get("waveform", None)
                    sr = audio_meta.get("sample_rate", None)
                    if wf is not None:
                        if torch.is_tensor(wf):
                            wf = wf.detach().cpu().numpy()
                        batch_waveforms.append({"waveform": wf, "sample_rate": int(sr) if sr else 44100})
                    else:
                        batch_waveforms.append(None)
                else:
                    batch_waveforms.append(None)
            all_audio_waveforms.extend(batch_waveforms)
        
        all_motions.append(sample_hml)
        all_gt_motions.append(gt_motion)
        all_lengths.append(lengths)
        
        collected += sample_hml.shape[0]
        if collected >= args.num_samples:
            break
    
    # Concatenate all results
    all_motions = np.concatenate(all_motions, axis=0)[:args.num_samples]
    all_gt_motions = np.concatenate(all_gt_motions, axis=0)[:args.num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:args.num_samples]
    
    if all_audio_embeds:
        all_audio_embeds = np.concatenate(all_audio_embeds, axis=0)[:args.num_samples]
    else:
        all_audio_embeds = None
    
    # Setup output path
    if args.output_path == "":
        model_name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
        output_dir = os.path.join(
            os.path.dirname(args.model_path),
            f"audio_gen_{model_name}_{niter}_n{args.num_samples}_seed{args.seed}"
        )
    else:
        output_dir = args.output_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, "generated_motions.npy")
    
    result_dict = {
        "motion": all_motions,           # Generated motions (N, T_pred, 263) - HumanML format, pred only
        "gt_motion": all_gt_motions,     # Ground truth motions (N, T_pred, 263), pred only
        "lengths": all_lengths,           # Motion lengths (prediction part)
        "num_samples": args.num_samples,
        "context_len": context_len,
        "pred_len": pred_len,             # Actual prediction length used (196)
        "dataset": full_args.dataset,
        "model_path": args.model_path,
        "seed": args.seed,
    }
    
    if all_audio_embeds is not None:
        result_dict["audio_embed"] = all_audio_embeds
    
    # Save audio waveforms (trimmed to match num_samples)
    if all_audio_waveforms:
        result_dict["audio_waveforms"] = all_audio_waveforms[:args.num_samples]
    
    np.save(output_file, result_dict)
    print(f"\n[Done] Saved {args.num_samples} generated motions to [{output_file}]")
    print(f"  Motion shape: {all_motions.shape} (samples, frames, 263)")
    print(f"  GT Motion shape: {all_gt_motions.shape}")
    print(f"  Lengths shape: {all_lengths.shape}")
    if all_audio_embeds is not None:
        print(f"  Audio embed shape: {all_audio_embeds.shape}")
    
    return output_file


if __name__ == "__main__":
    main()
