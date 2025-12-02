AIST++ integration (audio-conditioned)

Overview
- Dataset loader: `closd/diffusion_planner/data_loaders/aistpp/data/dataset.py`
- Config: `closd/diffusion_planner/dataset/aistpp_opt.txt`
- Stats tool: `closd/diffusion_planner/dataset/generate_aistpp_stats.py`
- Audio builder: `closd/diffusion_planner/dataset/build_aistpp_audio_feats.py`

Setup steps
1) Copy motion features (263-d HumanML style) into:
   - `closd/diffusion_planner/data/aistpp/motions/*.npy`
   Each file should be shaped `(T, 263)`, basename matching the audio file.

2) Build audio features from wavs (log-mel):
   - `python closd/diffusion_planner/dataset/build_aistpp_audio_feats.py \
       --wav_root /path/to/raw_aistpp/wavs \
       --out_dir closd/diffusion_planner/data/aistpp/audio_feats \
       --stats_out closd/diffusion_planner/dataset \
       --recursive`
   This writes `aistpp_audio_mean.npy` and `aistpp_audio_std.npy` under `dataset/`.

3) Compute motion stats for the copied 263-d features:
   - `python closd/diffusion_planner/dataset/generate_aistpp_stats.py \
       --motion_dir closd/diffusion_planner/data/aistpp/motions \
       --out_dir closd/diffusion_planner/dataset \
       --representation raw`
   This writes `aistpp_motion_mean.npy` and `aistpp_motion_std.npy`.

4) Edit `closd/diffusion_planner/dataset/aistpp_opt.txt` if needed to match final paths.

Training
- Use dataset `aistpp` and disable text encoder (audio only):
  - `--dataset aistpp --text_encoder_type none`

Notes
- The loader always time-aligns audio features to motion length (linear) and returns per-frame audio embeddings.
- Collation enforces: `audio_emb` is per-frame `(B, T, F)`, while `text_embed` is a pooled global audio vector `(1, B, F)` to unify the text-conditioning interface.
- If you prefer HumanML conversion from raw SMPL joints, set `remap_joints=true` and compute HumanML stats. This requires PyTorch + SMPL + HumanML extractor.

Data Access & Batch Structure
The underlying per-sample tuple (before collation) returned by `AISTPPMotionAudioDataset.__getitem__` is:

```
(audio_embeddings, dummy_pos, caption_placeholder, audio_token_len, motion, motion_len, audio_tokens_joined)
```

Field meanings:
- `audio_embeddings`: shape `(N_audio_tokens, F_audio)`. If `pool_audio=true` this becomes `(1, F_audio)` (mean-pooled). Otherwise each time-aligned audio frame is a token.
- `dummy_pos`: zeros of shape `(N_audio_tokens, 1)` kept for interface compatibility (unused).
- `caption_placeholder`: empty string `''` (no text caption for audio-conditioned samples).
- `audio_token_len`: integer number of audio tokens (1 when pooled).
- `motion`: raw or HumanML-converted motion features shape `(T_motion, D_motion)`. If `remap_joints=true` then `D_motion=263` (HumanML feature dim). Otherwise `D_motion = n_joints*3` flattened.
- `motion_len`: original (pre-padding/cropping) motion length in frames.
- `audio_tokens_joined`: joined token string e.g. `a0_a1_...`; used downstream as a surrogate tokens field.

Collated batch (after `t2m_collate` in `data_loaders/tensors.py`) produces:

```
motion:  torch.Size([B, D_motion, 1, T_fixed])
cond['y'] keys:
   mask:       (B, 1, 1, T_fixed) boolean broadcast mask
   lengths:    (B,) original (or fixed_len) frame counts
   text:       list[str] (empty strings here)
   tokens:     list[str] token strings (audio token codes)
   audio_emb:  (B, N_audio_tokens, F_audio)
   text_embed: (N_audio_tokens, B, F_audio) transposed view for modules expecting text embeddings
   db_key:     list (None placeholders unless return_keys enabled)
```

Shape examples (current behavior):
- `fixed_len=196`, `batch_size=2` → `motion (2,263,1,196)`, `audio_emb (2,196,80)`, `text_embed (1,2,80)`
- `fixed_len=120`, `batch_size=2` → `motion (2,263,1,120)`, `audio_emb (2,120,80)`, `text_embed (1,2,80)`
- `fixed_len=100`, `batch_size=2` → `motion (2,263,1,100)`, `audio_emb (2,100,80)`, `text_embed (1,2,80)`

Important dimensional conventions:
- Motion ordering after collate is `(batch, D_motion, 1, T_frames)` rather than `(batch, 1, D_motion, T)`; the singleton second dimension is retained for historical compatibility with other data reps.
- Audio features use `(token, feature)` ordering before collate and become `(batch, N_audio_tokens, F_audio)` after collate.
- If `pool_audio=false`, `N_audio_tokens` equals time-aligned motion length (or `fixed_len`) and `audio_emb` expands accordingly.

Access Patterns:
```python
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.utils import dist_util
loader = get_dataset_loader(name='aistpp', batch_size=4, num_frames=None, split='train',
                                          hml_mode='train', fixed_len=196, pred_len=0,
                                          abs_path='CLoSD_dancing/closd/diffusion_planner', device=dist_util.dev())
motion, cond = next(iter(loader))
audio_emb = cond['y']['audio_emb']      # (B, N_audio_tokens, F_audio)
motion_tensor = motion                  # (B, D_motion, 1, T_frames)
mask = cond['y']['mask']                # (B, 1, 1, T_frames)
lengths = cond['y']['lengths']          # (B,)
```


