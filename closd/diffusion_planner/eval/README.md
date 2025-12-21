# AIST++ Motion Evaluation Script

Evaluate motion generation quality on AIST++ dataset, comparing generated motions against ground truth.

## Usage

```bash
# Evaluate generated motions
python closd/diffusion_planner/eval/eval_aistpp_external.py \
    --external_results_file <path_to_npy_file>

# Evaluate ground truth motions (baseline)
python closd/diffusion_planner/eval/eval_aistpp_external.py \
    --external_results_file <path_to_npy_file> \
    --eval_gt
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--external_results_file` | (required) | Path to `.npy` or `.pkl` file with generated motions |
| `--eval_gt` | False | Evaluate ground truth motions instead of generated |
| `--device` | 0 | GPU device ID |
| `--seed` | 10 | Random seed |
| `--fps` | 60 | Motion FPS (AIST++ uses 60) |
| `--beat_sigma` | 0.08 | Gaussian tolerance for beat alignment (seconds) |
| `--smooth_kinematic` | False | Apply smoothing filter to kinematic speed |

## Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **FID** | Fr?chet Inception Distance between generated and GT motion distributions | Lower ¡õ |
| **Diversity** | Variance of motion features | Higher ¡ô |
| **BeatAlign** | How well kinematic beats align with music beats (0-1) | Higher ¡ô |
| **SkatingRatio** | Ratio of frames with foot skating artifact | Lower ¡õ |
| **FootSliding** | Total foot sliding distance (mm) | Lower ¡õ |
| **Penetration** | Foot penetration into ground (mm) | Lower ¡õ |
| **Floating** | Distance feet float above ground (mm) | Lower ¡õ |

## Beat Alignment (BAS)

The Beat Alignment Score follows the **AIST++ standard**:

1. **Music Beats**: Extracted from audio waveform using `librosa.beat.beat_track`
2. **Kinematic Beats**: Local minima of body velocity (velocity dips when dancer "hits" a beat)
3. **Score**: For each kinematic beat, find distance to nearest music beat, apply Gaussian kernel

Formula:
$$\text{BAS} = \frac{1}{|B_{dance}|} \sum_{t_d \in B_{dance}} \exp\left(-\frac{(t_d - t_{music}^{nearest})^2}{2\sigma^2}\right)$$

> **Note**: This is a **precision** metric - dancers can skip beats, but when they move, they should be on beat.

## Input File Formats

### `.npy` (from `generate_from_audio.py`)
```python
{
    "motion": np.ndarray,           # (N, T, 263) generated motions
    "gt_motion": np.ndarray,        # (N, T, 263) ground truth motions
    "lengths": np.ndarray,          # (N,) motion lengths
    "audio_waveforms": List[Dict],  # [{waveform, sample_rate}, ...] paired audio
}
```

### `.pkl` (from CLoSD `run.py`)
```python
{
    "motion": np.ndarray,  # (N, T, 263) generated motions
    "length": np.ndarray,  # (N,) motion lengths (note: "length" not "lengths")
}
```

## Example Output

```
=== AIST++ external eval ===
external_results_file: /path/to/generated_motions.npy
num_samples: 1000
FID: 19.1027
Diversity: 4.4411
BeatAlign: 0.3338 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0547
FootSliding(mm): 2.0688
Penetration(mm): 1.2183
Floating(mm): 231.1720
```

## Typical Results

| Metric | Generated | Ground Truth |
|--------|-----------|--------------|
| FID | ~19 | ~0.05 |
| Diversity | ~4.4 | ~5.3 |
| BeatAlign | ~0.33 | ~0.33 |
| SkatingRatio | ~0.05 | ~0.008 |
| FootSliding(mm) | ~2 | ~0.05 |
| Penetration(mm) | ~1.2 | ~0 |
| Floating(mm) | ~230 | ~60 |
