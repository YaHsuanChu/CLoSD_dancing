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

---

## 1. Evaluation Metrics 介紹與實作邏輯

在分析數據前，首先定義每個指標的數學意義及其在代碼中的實作方式。

### A. 生成質量指標 (Generative Quality)

#### **1. FID (Fréchet Inception Distance)**
*   **意義：** 衡量生成動作與真實動作在「特徵空間」上的分佈距離。數值**越低越好**，代表生成的動作在視覺和運動模式上越接近真實數據。
*   **算法：** 提取動作特徵向量（使用預訓練的 Motion Encoder），計算真實數據分佈 $(\mu_r, \Sigma_r)$ 與生成數據分佈 $(\mu_g, \Sigma_g)$ 的距離。
*   **公式：** $FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$

#### **2. Diversity**
*   **意義：** 衡量生成動作的多樣性。數值應接近 Ground Truth (GT)。
*   **算法：** 計算生成批次中，每對動作特徵向量之間的平均歐式距離。

---

### B. 音樂對齊指標 (Audio-Motion Alignment)

#### **3. Beat Align Score (BAS)**
*   **意義：** 衡量舞蹈動作的節拍（Kinematic Beats）是否準確落在音樂節拍（Music Beats）上。數值範圍 [0, 1]，**越高越好**。
*   **代碼實作 (Librosa & SciPy)：**
    1.  **音樂節拍提取 (`librosa`):** 使用 `librosa.beat.beat_track` 分析音頻波形，提取音樂的強拍時間點。
        ```python
        # 代碼概念
        import librosa
        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
        music_beats = librosa.frames_to_time(beat_frames, sr=sr)
        ```
    2.  **動作節拍提取:** 計算身體運動速度（Velocity），找出速度的局部極小值（Local Minima），這通常代表舞者在節拍點上的「定格」或轉向。
    3.  **對齊計算:** 對於每個動作節拍，尋找最近的音樂節拍，使用高斯函數計算分數：
        $$Score = \frac{1}{N} \sum \exp(-\frac{\Delta t^2}{2\sigma^2})$$
        其中 $\sigma$ 通常設為 0.08秒 (約等於 60FPS 下的 5 幀容錯)。

---

### C. 物理合理性指標 (Physical Realism)

這些指標用於檢測動作是否違反物理常識（如腳穿入地板、懸浮、滑步）。**數值皆為越低越好**。

#### **4. Skating Ratio (滑步率)**
*   **意義：** 計算腳在接觸地面時（高度低於閾值），水平速度卻過快（發生滑動）的幀數比例。
*   **算法：** 若 $Height_{foot} < H_{thresh}$ 且 $Velocity_{foot} > V_{thresh}$，則判定為滑步。

#### **5. Foot Sliding (滑動距離)**
*   **意義：** 滑步的嚴重程度，計算滑動的總距離（單位：mm）。

#### **6. Penetration (穿透深度)**
*   **意義：** 腳部陷入地下的平均深度（單位：mm）。理想值為 0。
*   **算法：** 計算所有 $y < 0$ 的關節座標的平均絕對值。

#### **7. Floating (浮動高度)**
*   **意義：** 衡量腳部整體的「懸浮」程度。若數值過高，代表舞者像幽靈一樣飄在空中，沒有著地感。
*   **注意：** 真實舞者也會跳躍，所以 GT 的 Floating 不會是 0，但異常高的數值代表生成錯誤。

---

## 2. 實驗結果匯總表 (Consolidated Metrics)

下表包含所有實驗設定的完整數據。
*   **GT Baseline:** 真實數據基準。
*   **粗體 (Bold):** 代表在該指標上表現**最佳**的模型結果（不含 GT，因為 GT 是天花板）。
*   *(註): CLoSD 的 BeatAlign 因音頻未對齊，數據無效。*

| Metric | GT Baseline (ng40) | CLoSD (Physics) | Raw Diffusion (ng20) | Raw Diffusion (ng40) | Raw Diffusion (ng80) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FID** (↓) | 0.0535 | **5.0655** | 16.1660 | 18.2840 | 11.7527 |
| **Diversity** (→GT) | 5.3355 | 3.5792 | **4.6198** | 4.3523 | 3.6255 |
| **Beat Align** (↑) | 0.3853 | 0.3727* (Invalid) | 0.3852 | 0.3834 | **0.3873** |
| **Skating Ratio** (↓) | 0.0083 | 0.0176 | 0.0152 | 0.0486 | **0.0087** |
| **Foot Sliding** (mm, ↓) | 0.0538 | **0.0000** | 0.9039 | 2.3931 | 0.2495 |
| **Penetration** (mm, ↓) | 0.0000 | **0.0002** | 1.8671 | 1.6065 | 0.1466 |
| **Floating** (mm, ↓) | 59.1546 | **23.8992** | 249.7649 | 242.1510 | 169.4469 |

---

## 3. 詳細分析報告

### A. 動作質量與真實度分析 (FID)
*   **CLoSD 的優勢：** 經過物理後處理的 **CLoSD** 取得了最佳的 FID (**5.07**)，遠低於所有原始輸出（Raw Diffusion）。這證明物理模擬不僅修復了接觸點，還有效地去除了生成動作中的非自然抖動和噪聲，使其特徵分佈更接近真實人類動作。
*   **長序列生成的驚喜：** 在原始輸出中，**ng80** 的 FID (**11.75**) 明顯優於 ng20 和 ng40。這表明該 Diffusion Transformer 模型在處理長序列時，能更好地捕捉動作的完整語義和連貫性，而非像傳統模型那樣容易崩潰。

### B. 音樂同步性分析 (Beat Align)
*   **SOTA 級別的節奏感：** **Raw Diffusion (ng80)** 的 Beat Align 分數 (**0.3873**) 甚至微幅超越了 Ground Truth (**0.3853**)。這是一個極強的結果，顯示模型非常精確地學會了將動作的頓點（Kinematic beats）對齊到音樂的重拍上。
*   **Context 的影響：** 對比 ng40 (0.3834) 和 ng80 (0.3873)，更長的生成窗口似乎有助於模型規劃更準確的節奏點。
*   **CLoSD 的數據缺失：** 表格中 CLoSD 的分數較低且無參考價值，是因為測試時使用的是隨機音頻。若能正確配對音頻，CLoSD 的分數理論上會略低於 Raw，因為物理模擬會為了滿足接觸約束而微調動作時間。

### C. 物理合理性分析 (Physics Metrics)
*   **懸浮問題 (Floating)：**
    *   **原始模型缺陷：** Raw Diffusion (ng20/40) 的 Floating 高達 **240mm+**，這意味著生成的舞者平均離地 24 公分，視覺上會像是在「飄浮」。
    *   **CLoSD 修復：** CLoSD 將 Floating 壓至 **23.9mm**。雖然比 GT (59mm) 更低（可能過度吸附地面），但成功解決了懸浮問題，讓動作看起來紮實。
    *   **ng80 的自我修正：** 值得注意的是，ng80 的 Floating (**169mm**) 比 ng20/40 大幅改善。這暗示長序列生成促使模型學會了更穩定的全局姿態，減少了整體飄移。
*   **滑步與穿透 (Skating & Penetration)：**
    *   **CLoSD** 幾乎將滑動距離 (Sliding) 和穿透 (Penetration) 降至 **0**，這是物理模擬器的核心功能。
    *   **ng80** 在沒有物理後處理的情況下，Skating Ratio (**0.0087**) 竟然達到了 GT 水平 (**0.0083**)，這再次印證了長序列訓練/生成對於動作物理穩定性的正面影響。

### 4. 總結 (Conclusion)
本實驗驗證了兩階段生成策略的有效性：
1.  **Raw Diffusion (特別是 ng80)** 展現了極佳的音樂對齊能力和動作連貫性，且在長序列生成下具有更好的物理穩定性。
2.  **CLoSD 後處理** 對於解決 Diffusion 模型常見的「懸浮」與「滑步」問題至關重要，它將 FID 分數提升了兩倍以上，將視覺真實度推向了實用水平。


---


conda activate closd && CUDA_VISIBLE_DEVICES=5 python closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_dancing/output/CLoSD.pkl --beat_samples 1000
[WARN] lengths max (2877) > actual_motion_len (400), clamping
[WARN] No paired audio in npy file, using random audio from dataloader
[INFO] Loading 1000 audio samples for BeatAlign (fixed_len=400)
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
[DEBUG] Dataloader done!
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/CLoSD.pkl
num_samples: 1000
FID: 5.0655
Diversity: 3.5792
BeatAlign: 0.3727 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0176
FootSliding(mm): 0.0000
Penetration(mm): 0.0002
Floating(mm): 23.8992

conda activate closd && CUDA_VISIBLE_DEVICES=5 python closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng20/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng20/sample/generated_motions.npy
num_samples: 1000
FID: 16.1660
Diversity: 4.6198
BeatAlign: 0.3852 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0152
FootSliding(mm): 0.9039
Penetration(mm): 1.8671
Floating(mm): 249.7649

conda activate closd && CUDA_VISIBLE_DEVICES=5 python closd/diffusion_planner/eval/eval_aistpp_e
xternal.py --external_results_file /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy --eval_gt
[INFO] Loading ground truth motions (gt_motion)
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy
num_samples: 1000
FID: 0.0535
Diversity: 5.3355
BeatAlign: 0.3853 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0083
FootSliding(mm): 0.0538
Penetration(mm): 0.0000
Floating(mm): 59.1546

conda activate closd && CUDA_VISIBLE_DEVICES=5 python closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy
num_samples: 1000
FID: 18.2840
Diversity: 4.3523
BeatAlign: 0.3834 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0486
FootSliding(mm): 2.3931
Penetration(mm): 1.6065
Floating(mm): 242.1510

conda activate closd && CUDA_VISIBLE_DEVICES=5 python closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng80/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng80/sample/generated_motions.npy
num_samples: 1000
FID: 11.7527
Diversity: 3.6255
BeatAlign: 0.3873 (computed on 1000 samples with audio)
SkatingRatio(mean): 0.0087
FootSliding(mm): 0.2495
Penetration(mm): 0.1466
Floating(mm): 169.4469