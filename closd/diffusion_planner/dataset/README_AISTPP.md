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
- Collation enforces: `audio_emb` is per-frame `(B, F, T)`, while `text_embed` is a pooled global audio vector `(1, B, F)` to unify the text-conditioning interface.
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

Collated batch (after `t2m_collate` / `t2m_prefix_collate` in `data_loaders/tensors.py`) produces:

```
motion:  torch.Size([B, D_motion, 1, pred_len])
cond['y'] keys:
   mask:               (B, 1, 1, pred_len) boolean broadcast mask
   lengths:            (B,) original (or fixed_len) frame counts
   text:               list[str] (empty strings here)
   tokens:             list[str] token strings (audio token codes)
   prefix:             (B, D_motion, 1, context_len)  
   audio_embed_prefix: (B, F_audio, context_len)      
   audio_embed_pred:   (B, F_audio, pred_len)      
   text_embed:         (1, B, F_audio)                # pooled global audio vector from pred window
   db_key:             list (None placeholders unless return_keys enabled)
```

Shape examples (current behavior):
- Prefix mode: `context_len=20`, `pred_len=40`, `batch_size=64` → `motion prefix (64,263,1,20)`, `motion inp (64,263,1,40)`, `audio_embed_prefix (64,80,20)`, `audio_embed_pred (64,80,40)`, `text_embed (1,64,80)`

Important dimensional conventions:
- Motion ordering after collate is `(batch, D_motion, 1, pred_len)` rather than `(batch, 1, D_motion, pred_len)`; the singleton second dimension is retained for historical compatibility with other data reps.
- Audio features use `(token, feature)` ordering before collate and become `(batch, N_audio_tokens, F_audio)` after collate.
- In prefix mode, audio is split into two windows: `audio_embed_prefix (context_len)` and `audio_embed_pred (pred_len)`. In non-prefix mode, only `audio_embed_pred` is provided and spans the full `T_fixed`.

Access Patterns:
```python
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.utils import dist_util

loader = get_dataset_loader(name='aistpp', batch_size=64, num_frames=None, split='train',
                            hml_mode='train', fixed_len=None, pred_len=40, context_len=20,
                            abs_path='CLoSD_dancing/closd/diffusion_planner', device=dist_util.dev())
motion, cond = next(iter(loader))
audio_prefix = cond['y']['audio_embed_prefix']  # (B, F_audio, context_len)
audio_pred   = cond['y']['audio_embed_pred']    # (B, F_audio, pred_len)
text_embed   = cond['y']['text_embed']          # (1, B, F_audio)
motion_prefix = cond['y']['prefix']             # (B, D_motion, 1, context_len)
```


## 中文補充：AIST++ 的 audio concat / cross-attention 模式

這一節說明在 AIST++ 上，音訊特徵怎麼進入 DiP（MDM）模型，以及兩種實驗模式要怎麼設定 config。

### 1. 資料流回顧

- `motion`: 經過 collate 後為 `motion: (B, D_motion, 1, pred_len)`，AIST++ / HumanML 目前 `D_motion=263`。
- `audio_embeddings` 經過 collate 後會被拆成：
   - `cond['y']['audio_embed_prefix']: (B, F_audio, context_len)`
   - `cond['y']['audio_embed_pred']:   (B, F_audio, pred_len)`
- `text_embed`: `cond['y']['text_embed']: (1, B, F_audio)`，為 `audio_embed_pred` 在時間維度上的平均，做為「全域 audio token」，與原有 text-conditioning 介面對齊。

在訓練 loop 中（`train/training_loop.py`），若 `audio_concat_mode='concat'`，會將 per-frame audio 特徵 concat 到 motion channel：

- pred 段：
   - `audio_embed_pred (B, F_audio, pred_len)` 會先轉成 `(B, F_audio, 1, pred_len)`；
   - `x = concat(motion, audio_feat_pred, dim=1)` 變成 `(B, 263+F_audio, 1, pred_len)` 再送進 diffusion；
- prefix 段（若有 prefix / completion 模式）：
   - `audio_embed_prefix (B, F_audio, context_len)` 同樣轉成 `(B, F_audio, 1, context_len)`；
   - concat 到 `cond['y']['prefix']`，讓 prefix 也包含 audio channel，與 pred 段的 channel 數對齊。

在 diffusion loss 裡（`GaussianDiffusion.training_losses`），若偵測到：

- `model_kwargs['y']['audio_embed_pred']` 存在，且 `audio_concat_mode='concat'`；
- dataset name 包含 `humanml` 或 `aist`；

則會只對前 `D_motion` 個 channel 計算 rot MSE：

```python
motion_dim = 263  # for humanml / aistpp
target_motion = target[:, :motion_dim, ...]
model_output_motion = model_output[:, :motion_dim, ...]
rot_mse = masked_l2(target_motion, model_output_motion, mask)
```

也就是說，concat 上去的 audio channel 僅視為「輸入特徵」，不強迫模型重建它。

為了修正 prefix-completion + `trans_enc` 架構下的時間維 off-by-one 問題，在 AIST++ + audio concat + 有 prefix 時，loss 端會對 model_output 在時間維度做一次 slicing，使其與 target 的 `pred_len` 對齊，只對指定的預測區間 supervision。

### 2. 兩種常用實驗模式

目前推薦在 AIST++ 上使用以下兩種設定來對照：

#### 模式 A：純 concat（沒有 cross-attention）

目標：

- 模型只從輸入上的 `[motion, audio_channel]` 學習，不再透過任何 text/audio cross-attention 取條件。

建議訓練指令（範例）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m closd.diffusion_planner.train.train_mdm \
   --save_dir output/CLoSD/CLoSD_aistpp_concat_only \
   --dataset aistpp \
   --text_encoder_type none \
   --audio_concat_mode concat \
   --audio_dim 80 \
   --audio_feat_dim 80 \
   --batch_size 64 \
   --num_frames 60 \
   --num_steps 200000 \
   --lr 1e-4 \
   --text_uncond_all \
   --lambda_target_loc 0.0 \
   --context_len 20 \
   --pred_len 40
```

說明：

- `--text_encoder_type none`：不載入 CLIP/BERT，MDM 只使用 `y['text_embed']`（此處為 pooled audio）作為介面，但我們會在下面關掉它。
- `--audio_concat_mode concat`：開啟 per-frame audio concat 到 motion channel 的路徑。
- `--audio_dim` / `--audio_feat_dim`：必須與 `build_aistpp_audio_feats.py` 產生的 audio feature 維度一致（預設 80）。
- `--text_uncond_all`：訓練時每個 batch 都設 `cond['y']['text_uncond']=True`，
   讓 MDM 的 `mask_cond(..., force_mask=True)` 把所有 text/audio/action cond 向量歸零，相當於關掉 cross-attention；
   此時模型唯一「看到」的 audio 資訊就是輸入 $x$ 上 concat 的 per-frame audio channel。

#### 模式 B：per-frame audio cross-attention（不做 concat）

目標：

- 不在 $x$ 上 concat audio channel，而是把 prefix/pred 的每一幀 audio 特徵當作 cross-attention 的 memory token。
- 透過 `--text_uncond_all` 關閉 pooled audio/text cond，避免與 per-frame token 重複。

建議訓練指令（範例）：

```bash
python -m closd.diffusion_planner.train.train_mdm \
   --save_dir output/diffusion/aistpp_per_frame_cross_attn_only_ng20 \
   --dataset aistpp \
   --text_encoder_type none \
   --audio_concat_mode none --text_uncond_all --per_frame_audio_xatten \
   --arch trans_dec \
   --audio_dim 80 --audio_feat_dim 80 \
   --batch_size 64 --num_steps 200000 \
   --num_frames 60 --context_len 20 --pred_len 40 \
   --lambda_target_loc 0.0 --device 6
```

與模式 A 的差別在於：

- `--audio_concat_mode none`：$x$ 輸入只包含 motion，沒有 audio channel。
- `--per_frame_audio_xatten`：`audio_embed_prefix/pred` 會經 `embed_text` + `sequence_pos_encoder` 後，拼到 decoder 的 memory 序列，讓跨注意力看到每一幀 audio token。
- `--text_uncond_all`：pooled audio（`text_embed`）會被 mask_cond 清為 0，不再進入 cross-attention，避免與 per-frame audio token 重複。
- `--arch tans_dec` : 原本 concat 模式是使用 trans_enc, 而 cross attention 模式必須使用 trans_dec 模式，因為 nn.TransformerDecoder 才有 cross attention 的機制


### 3. 總結與建議

- 若你只想要「讓音樂訊號長在 motion 上」，不希望有任何 cross-attention 分支干擾，建議使用：
   - **模式 A（純 concat）**：`audio_concat_mode='concat'` + `text_uncond_all`。

- 若你想比較「只用 per-frame audio token 透過 cross-attention」對品質的影響，則使用：
   - **模式 B（per-frame cross-attention）**：`audio_concat_mode='none'` + `per_frame_audio_xatten` + `text_uncond_all`。

兩種模式共用同一套 data loader 與 stats/builder 流程，只透過 CLI 旗標與 cond flag 切換，方便在同一訓練程式中做 ablation 與比較。


### 4. Sampling 範例（生成音訊條件動作）
Run the following command to generate samples with `generate.py`
**generate command (xatten)**
```
python -m closd.diffusion_planner.sample.generate \
    --model_path output/diffusion/aistpp_per_frame_cross_attn_only_ng40/model000200000.pt \
    --output_dir output/diffusion/aistpp_per_frame_cross_attn_only_ng40/samples/ \
    --dataset aistpp \
    --text_encoder_type none --audio_concat_mode none --text_uncond_all \
    --per_frame_audio_xatten \
    --arch trans_dec \
    --audio_dim 80 --audio_feat_dim 80 \
    --context_len 20 --pred_len 40 \
    --autoregressive --motion_length 9.8 \
    --num_samples 8 --device 0
```

**generate command (concat)**
```
python -m closd.diffusion_planner.sample.generate \
    --model_path output/diffusion/aistpp_concat_ng40/model000200000.pt \
    --output_dir output/diffusion/aistpp_concat_ng40/samples/ \
    --dataset aistpp \
    --text_encoder_type none --audio_concat_mode concat --text_uncond_all \
    --audio_dim 80 --audio_feat_dim 80 \
    --context_len 20 --pred_len 40 \
    --autoregressive --motion_length 9.8 \
    --num_samples 8 --device 0
```

## 中文補充：AIST++ 的 audio concat / cross-attention 模式

這一節說明在 AIST++ 上，音訊特徵怎麼進入 DiP（MDM）模型，以及兩種實驗模式要怎麼設定 config。

### 1. 資料流回顧

- `motion`: 經過 collate 後為 `motion: (B, D_motion, 1, pred_len)`，AIST++ / HumanML 目前 `D_motion=263`。
- `audio_emb`: `cond['y']['audio_emb_pred']: (B, N_audio_tokens, F_audio)`，在不 pool 的設定下 `N_audio_tokens=pred_len`，每一幀一個 audio token。
- `text_embed`: `cond['y']['text_embed']: (N_tokens, B, F_audio)`，目前會把 `audio_emb` 在時間維度做平均，得到一個 global audio token（`N_tokens=1`），提供給 MDM 原本的 text-conditioning 介面使用。

訓練與推論時，`training_loop` 會依照 `audio_concat_mode` 決定是否把 `audio_emb` concat 進輸入 $x$：

- 當 `audio_concat_mode='concat'` 時：
   - `audio_embed_pred (B, T, F)` 會被轉成 `(B, F_audio, 1, T)`，並在 channel 維度與 `motion` concat：
      - `x = concat(motion, audio_feat)  # [B, D_motion + F_audio, 1, T]`
   - DiP/MDM 的 `njoints` 也會依據 `audio_dim` 自動變成 `263 + audio_dim`（只在 AIST++ 下生效）。
- 當 `audio_concat_mode='none'`（預設）時：
   - `x = motion`，模型和原本 HumanML / DiP 行為相同，不會在輸入上看到 audio channel。

在 diffusion loss 裡（`gaussian_diffusion.training_losses`），若偵測到有 `model_kwargs['y']['audio_embed_pred']`（即 AIST++ concat 模式），會只對前 `D_motion` 個 channel（263）計算 reconstruction loss，視 concat 上去的 audio channel 為「附帶 feature、不強迫重建」。

在 sampling 時（`sample/generate.py`），若 `audio_concat_mode='concat'` 且通道數大於 motion 維度，會先砍掉後面的 audio channel，再進行 `inv_transform` / `recover_from_ric` 等後處理，保證輸出仍然是純 motion：

```python
if args.dataset in ['humanml', 'aistpp']:
      motion_dim = 263
...
if getattr(args, 'audio_concat_mode', 'none') == 'concat' and sample.shape[1] > motion_dim:
      sample = sample[:, :motion_dim, ...]
```

### 2. 兩種常用實驗模式

目前推薦在 AIST++ 上使用以下兩種設定來對照：

#### 模式 A：純 concat（沒有 cross-attention）

目標：
- 模型只從輸入上的 `[motion, audio_channel]` 學習，不再透過任何 text/audio cross-attention 取條件。

建議設定：

- 在訓練腳本中（`train_mdm.py`）：
   - `--dataset aistpp`
   - `--text_encoder_type none`  （沿用本檔最上方的建議，關閉文字編碼器）
   - `--audio_concat_mode concat`
   - `--audio_dim 80`  （或你的 audio feature 維度）
   - `--text_uncond_all`  （**新參數**：會讓訓練時每一個 batch 都走 text_uncond=True）

- 在程式內部，`TrainLoop` 會偵測 `args.text_uncond_all`，若為 True 則自動：
   - 對每個 batch 設定 `cond['y']['text_uncond'] = True`，
   - 讓 MDM 的 `mask_cond(..., force_mask=True)` 直接把 text/audio/action cond 統一 mask 成 0。

邏輯上可理解為：

$$
	ext{emb} = \text{time\_emb} \quad (\text{因為 text/audio cond 被強制為 0})
$$

模型看到的唯一「條件」就是輸入 $x$ 上的 motion + audio channel，沒有額外的 cross-attention 訊息。

#### 模式 B：per-frame audio cross-attention（不做 concat）

目標：
- $x$ 只保留 motion；prefix/pred 的每一幀 audio 特徵改由 cross-attention 訪問。

建議設定：

- `--dataset aistpp`
- `--text_encoder_type none`
- `--audio_concat_mode none`
- `--per_frame_audio_xatten`
- `--arch trans_dec`
- `--text_uncond_all`（遮蔽 pooled audio/text cond，只保留 per-frame audio token）

條件流說明：

1. `audio_embed_prefix/pred` 經 `embed_text` + `sequence_pos_encoder`，拼到 decoder memory，跨注意力逐幀使用。
2. 由於 `text_uncond_all=True`，pooled audio（`text_embed`）會被 mask_cond 清為 0，不再進入 cross-attention。
3. 輸入 $x$ 沒有 audio channel，因此 diffusion loss 仍只覆蓋 motion channel。

### 3. 總結與建議

- 若你只想要「讓音樂長在 motion 上」，不希望有任何 cross-attention 分支干擾：
   - 選擇 **模式 A（純 concat）**：`audio_concat_mode='concat'` + `text_uncond=True`。

- 若你想比較「只靠 per-frame audio token 的 cross-attention」對品質的影響：
   - 選擇 **模式 B（per-frame cross-attention）**：`audio_concat_mode='none'` + `per_frame_audio_xatten` + `text_uncond_all`。

兩種模式共用同一套 data loader 与 stats/builder 流程，僅透過 config / cond flag 來切換，方便在同一訓練程式中做 ablation 與比較。其中：

- 是否 concat audio：由 `--audio_concat_mode` 控制；
- 是否完全關掉 cross-attention：由 `--text_uncond_all`（訓練）或自行在 y 裡設 `text_uncond=True`（其他使用情境）來控制。
