# Motion Generation Scripts

This folder contains scripts for generating motions using trained diffusion models.

## Audio-to-Motion Generation

### Usage

```bash
python -m closd.diffusion_planner.sample.generate_from_audio \
    --model_path <path_to_model.pt> \
    --num_samples 300 \
    --batch_size 32 \
    --output_path <output_folder>
```

python -m closd.diffusion_planner.sample.generate_from_audio \
    --model_path  output/concat_ng40/model000200000.pt\
    --num_samples 300 \
    --n_frames 300 \
    --batch_size 32 \
    --output_path output/concat_ng40/npy

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to the trained model checkpoint |
| `--num_samples` | 300 | Number of samples to generate |
| `--batch_size` | 32 | Batch size for generation |
| `--output_path` | auto | Output directory (auto-generated if not specified) |
| `--seed` | 42 | Random seed |
| `--device` | cuda:0 | Device to use |
| `--guidance_param` | 1.0 | Classifier-free guidance scale |
| `--use_ema` | False | Use EMA model weights |

> **Note**: Model architecture parameters (context_len, pred_len, audio_concat_mode, etc.) are automatically loaded from `args.json` in the model directory.

---

## Output Format

The generated motions are saved as `generated_motions.npy` in the output directory.

### Data Structure

```python
{
    "motion": np.ndarray,       # (N, 196, 263) - Generated motions
    "gt_motion": np.ndarray,    # (N, 196, 263) - Ground truth motions
    "lengths": np.ndarray,      # (N,) - Motion lengths (all 196)
    "audio_embed": np.ndarray,  # (N, 80, 196) - Audio embeddings
    "num_samples": int,         # Number of samples
    "context_len": int,         # Context length used (e.g., 20)
    "pred_len": int,            # Prediction length (196)
    "dataset": str,             # Dataset name
    "model_path": str,          # Model checkpoint path
    "seed": int,                # Random seed used
}
```

### How to Load

```python
import numpy as np

# Load the file
data = np.load('generated_motions.npy', allow_pickle=True).item()

# Access data
motions = data['motion']           # (300, 196, 263)
gt_motions = data['gt_motion']     # (300, 196, 263)
lengths = data['lengths']          # (300,)
audio_embeds = data['audio_embed'] # (300, 80, 196)

# Single sample
motion_0 = motions[0]              # (196, 263)
gt_0 = gt_motions[0]               # (196, 263)
```

---

## HumanML 263-dim Format

Each frame is a 263-dimensional vector:

| Dimension | Content |
|-----------|---------|
| 0-3 | Root rotation and height |
| 4-66 | Joint positions (21 joints × 3) |
| 67-129 | Joint velocities (21 joints × 3) |
| 130-192 | Joint rotations (21 joints × 3, 6D rotation) |
| 193-262 | Foot contact and other features |

This format is consistent with the HumanML3D dataloader output.
