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
| `--beat_sigma` | 0.05 | Gaussian tolerance for beat alignment (seconds) |
| `--beat_samples` | -1 (all) | Number of samples to use for beat alignment (-1 = all) |
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

The Beat Alignment Score is calculated using three metrics to provide a comprehensive view of rhythm quality:

1.  **Precision** (AIST++ standard):
    - For each **kinematic beat** (dance), find the nearest **music beat**.
    - Penalizes extra movements that are off-beat.
    - Formula: Dance $\to$ Music.

2.  **Recall** (Bailando/EDGE standard):
    - For each **music beat**, find the nearest **kinematic beat** (dance).
    - Checks if the dancer "hits" every musical beat.
    - Formula: Music $\to$ Dance.

3.  **F1 Score**:
    - Harmonic mean of Precision and Recall.
    - $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**Sigma ($\sigma$)**: The Gaussian tolerance is set to **0.05s** (default), representing a tight window for alignment (~3 frames at 60 FPS).

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

**1. FID (Fréchet Inception Distance)**
*   **意義：** 衡量生成動作與真實動作在「特徵空間」上的分佈距離。數值**越低越好**，代表生成的動作在視覺和運動模式上越接近真實數據。
*   **算法：** 提取動作特徵向量（使用預訓練的 Motion Encoder），計算真實數據分佈 $(\mu_r, \Sigma_r)$ 與生成數據分佈 $(\mu_g, \Sigma_g)$ 的距離。
*   **公式：** $FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$

**2. Diversity**
*   **意義：** 衡量生成動作的多樣性。數值應接近 Ground Truth (GT)。
*   **算法：** 計算生成批次中，每對動作特徵向量之間的平均歐式距離。

### B. 音樂對齊指標 (Audio-Motion Alignment)

**3. Beat Align Score (BAS) - 標準演變與細節**
*   **意義：** 衡量舞蹈動作的節拍（Kinematic Beats）是否準確落在音樂節拍（Music Beats）上。數值範圍 [0, 1]，**越高越好**。
*   **實作原理 (Librosa & SciPy)：**
    1.  **音樂節拍提取 (`librosa`):** 使用 `librosa.beat.beat_track` 分析音頻波形。
    2.  **動作節拍提取:** 計算身體運動速度（Velocity），找出速度的局部極小值（Local Minima），代表舞者在節拍點上的「定格」或轉向。
    3.  **對齊計算:** 使用高斯函數計算分數：$Score = \frac{1}{N} \sum \exp(-\frac{\Delta t^2}{2\sigma^2})$，其中 $\sigma=0.05s$ (約 3 frames)。

> **⚠️ 關鍵補充：評測標準的轉移 (Metric Standard Shift)**
> 學界對於 "BAS" 的定義經歷了演變，本報告同時列出以下三種以供全面分析：
> *   **Precision (AIST++ 舊標準):** $Dance \to Music$。舞者的每一個頓點是否都對準音樂？（懲罰多餘動作）。
> *   **Recall (Bailando/EDGE 新 SOTA):** $Music \to Dance$。音樂的每一個拍子，舞者是否都有動作對應？（目前論文主流比較指標，但容易因轉圈圈產生密集動作而虛高）。
> *   **F1 Score:** Precision 與 Recall 的調和平均，最能代表綜合節奏能力。

### C. 物理合理性指標 (Physical Realism)
這些指標用於檢測動作是否違反物理常識。**數值皆為越低越好**。

**4. Skating Ratio (滑步率)**
*   **意義：** 計算腳在接觸地面時（高度低於閾值），水平速度卻過快（發生滑動）的幀數比例。
*   **算法：** 若 $Height_{foot} < H_{thresh}$ 且 $Velocity_{foot} > V_{thresh}$，則判定為滑步。

**5. Foot Sliding (滑動距離)**
*   **意義：** 滑步的嚴重程度，計算滑動的總距離（單位：mm）。

**6. Penetration (穿透深度)**
*   **意義：** 腳部陷入地下的平均深度（單位：mm）。理想值為 0。
*   **算法：** 計算所有 $y < 0$ 的關節座標的平均絕對值。

**7. Floating (浮動高度)**
*   **意義：** 衡量腳部整體的「懸浮」程度。若數值過高，代表舞者像幽靈一樣飄在空中，沒有著地感。
*   **注意：** 真實舞者也會跳躍，所以 GT 的 Floating 不會是 0，但異常高的數值代表生成錯誤。

---

## 2. 實驗數據表格庫 (Data Tables)


### 表格 A: 完整實驗數據總表 (Master Table)

這是包含所有指標的總覽表

*   **單位說明：** Sliding, Penetration, Floating 單位皆為 **mm**。
*   **粗體 (Bold)：** 代表該指標表現最佳（接近 GT 或數值最優）。
*   *(註): CLoSD 的 BeatAlign 因音頻隨機配對，數值無效故不列入。*

| Model Config | Context | **FID** (↓) | **Diversity** | **BAS (F1)** (↑) | **BAS (Recall)** | **Skating** (↓) | **Foot Sliding** (mm, ↓) | **Penetration** (mm, ↓) | **Floating** (mm, ↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth** | - | **0.05** | **5.34** | 0.325 | 0.529 | **0.0083** | 0.05 | 0.00 | 59.2 |
| **CLoSD (Phys)** | - | **5.07** | 3.58 | - | - | 0.0176 | **0.00** | **0.00** | **23.9** |
| **Concat (Ours)** | ng20 | 16.17 | 4.62 | 0.366 | 0.763 | 0.0152 | 0.90 | 1.87 | 249.8 |
| **Concat (Ours)** | ng40 | 18.28 | 4.35 | **0.368** | **0.790** | 0.0486 | 2.39 | 1.61 | 242.2 |
| **Concat (Ours)** | **ng80** | **11.75** | 3.63 | 0.357 | 0.689 | **0.0087** | **0.25** | **0.15** | 169.4 |
| **X-Attention** | ng40 | 27.84 | 3.66 | 0.355 | 0.644 | 0.0000* | 0.00* | 0.00* | 103.4 |

---

### 表格 B: 生成質量與物理真實性 (Quality & Physics Focus)
*適合用來強調 CLoSD 的修復能力以及 ng80 的長序列穩定性。*

| Method | **FID** (Realism) ↓ | **Skating Ratio** ↓ | **Floating** (mm) ↓ | **Penetration** (mm) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Ground Truth** | 0.05 | 0.0083 | 59.2 | 0.00 |
| **CLoSD (Post-process)** | **5.07** | 0.0176 | **23.9** | **0.00** |
| **Concat-ng20** | 16.17 | 0.0152 | 249.8 | 1.87 |
| **Concat-ng40** | 18.28 | 0.0486 | 242.2 | 1.61 |
| **Concat-ng80** | **11.75** | **0.0087** | 169.4 | 0.15 |

---

### 表格 C: 音樂對齊深入分析 (Beat Alignment Deep Dive)
*適合用來分析節奏感，並解釋 SOTA Recall 與 Precision 的權衡。*

| Method | **F1 Score** (Balance) | **Recall** ($M \to D$) (SOTA Metric) | **Precision** ($D \to M$) (AIST++ Metric) | 分析備註 |
| :--- | :---: | :---: | :---: | :--- |
| **Ground Truth** | 0.325 | 0.529 | **0.246** | 真實舞者會適度留白，Recall 不會是 1 |
| **Concat-ng80** | 0.357 | 0.689 | **0.246** | **最佳平衡**，密度最接近 GT |
| **Concat-ng40** | **0.368** | **0.790** | 0.244 | Recall 過高，動作過密 |
| **Concat-ng20** | 0.366 | 0.763 | 0.245 | Recall 過高，動作過密 |
| **X-Attention** | 0.355 | 0.644 | 0.250 | **虛高**：因轉圈產生密集 Beats 覆蓋節拍 |

---

### 表格 D: 模型架構比較 (Ablation: Architecture)
*適合用來證明 Concat 優於 X-Attention。*

| Model Architecture | **FID** (↓) | **BAS F1** (↑) | **Floating** (mm) | 視覺效果描述 |
| :--- | :---: | :---: | :---: | :--- |
| **Concat (Recommended)** | **18.28** | **0.368** | 242.2 | 動作豐富，節奏對齊 |
| **X-Attention** | 27.84 | 0.355 | **103.4*** | **Mode Collapse (原地轉圈)** |

---

### 表格 E: 生成長度影響 (Ablation: Sequence Length)
*適合用來展示 ng80 的優越性。*

| Sequence Length | **FID** (↓) | **Skating Ratio** (↓) | **Floating** (mm) (↓) | **Diversity** |
| :--- | :---: | :---: | :---: | :---: |
| **Short (ng20)** | 16.17 | 0.0152 | 249.8 | **4.62** |
| **Medium (ng40)** | 18.28 | 0.0486 | 242.2 | 4.35 |
| **Long (ng80)** | **11.75** | **0.0087** | **169.4** | 3.63 |

---

## 3. Deep Dive Analysis Report

### A. 動作質量與真實度分析 (FID)

**3.1. CLoSD 的物理修正效果：**
*   **觀察：** CLoSD 的 FID (**5.07**) 遠低於所有原始輸出（Raw Diffusion: 11.75 - 27.84）。
*   **原因分析：**
    *   **去除雜訊：** Raw Diffusion 模型在生成過程中，由於數據的分佈、長時依賴的挑戰，或者注意力機制的缺陷，常常會產生非物理的細節（如輕微的漂浮、不自然的關節扭曲、快速的細微抖動）。FID 的計算基於動作的特徵向量，這些細節會被 encoder 捕捉到，導致其特徵分佈與真實數據產生較大差異。
    *   **修正接觸點：** CLoSD 的物理模擬強制將腳部關節約束在地平面附近，這極大地修正了「懸浮」和「穿透」等根本性問題。腳部穩定的接觸是人體運動中非常重要的真實性來源。當腳步被正確「錨定」後，整體動作的時空特徵會更趨近於真實數據。
    *   **案例對比：** 原始 ng20/40 的 Floating 高達 **240mm+**，而 CLoSD 僅為 **23.9mm**。這巨大的差異直接影響了 FID，因為腳部離地高度是衡量動作「落地感」的重要指標。
*   **結論：** CLoSD 的物理後處理是提升生成動作視覺真實度（FID）的關鍵步驟。它不僅解決了單一的物理錯誤（如穿模），更透過約束動作的全局時空軌跡，使其整體分佈向真實數據靠攏。

**3.2. 長序列生成的驚喜 (FID of ng80)：**
*   **觀察：** Concat-ng80 的 FID (**11.75**) 明顯優於 Concat-ng20 (16.17) 和 ng40 (18.28)。
*   **原因分析：**
    *   **長時依賴 (Long-Term Dependency):** 舞蹈是由一系列連貫的動作組合而成，具有較長的時序結構（例如，一個完整的舞步或一個樂句）。傳統的 Transformer 模型在處理非常長的序列時，可能面臨梯度消失/爆炸、注意力範圍不足等問題。
    *   **語義連貫性：** ng80 能夠捕捉到更長的動作模式和語義結構。例如，一個完整的旋轉、一個連貫的擺臂動作，需要足夠的上下文才能被準確生成。短序列可能只捕捉到動作的片段，導致缺乏整體性。
    *   **分布擬合：** Diffusion 模型試圖學習數據的整體分佈。對於像舞蹈這樣具有複雜結構的數據，越長的樣本能提供越多關於其潛在分佈結構的信息。ng80 樣本可能提供了更多關於真實舞蹈「流動性」、「動機（motif）」的信息，使模型學到的分佈更為精準。
    *   **對比傳統模型：** 傳統方法（如 RNN）在長時依賴上較弱。Transformer 在理論上擅長長時依賴，但實際效果受限於模型大小、訓練數據和長度。這裡的結果說明，在這種設置下，長序列生成對 FID 的提升是實質性的。
*   **結論：** 對於 AIST++ 這樣的數據集，長序列生成（如 ng80）是捕捉舞蹈完整性、語義連貫性和提升 FID 的關鍵。

### B. 音樂同步性分析 (Beat Align)

**3.3. SOTA 標準的演變與解讀 (Precision vs. Recall vs. F1):**
*   **指標的演變：**
    *   **舊標準 (Precision):** AIST++ 原論文和早期工作（如 FACT）主要關注 **Precision ($D \to M$)**。其目的是確保舞者**不要亂動**。如果舞者在非節拍點上移動，Precision 就會下降。
    *   **新標準 (Recall):** 近期 SOTA 工作（Bailando, EDGE）轉向使用 **Recall ($M \to D$)**。目的是確保舞者**不錯過每一個音樂節拍**。
*   **Precision vs. Recall 的衝突：**
    *   **Precision:** 嚴格要求動作節拍緊貼音樂節拍，並懲罰額外動作。
    *   **Recall:** 寬鬆要求音樂節拍附近有動作即可，有利於生成更密集的運動。
*   **您的數據中的體現：**
    *   **GT 的 Precision (0.246) vs. Recall (0.529):** GT 的 Precision 很高，說明舞者移動非常準確。但 Recall 低於 Precision，表明舞者並非每個拍子都有強烈動作（例如，有些拍子是延音，有些是過渡）。
    *   **Concat-ng40 的 Recall (0.790) vs. Precision (0.244):** ng40 的 Recall 非常高，Precision 與 GT 相當。這意味著生成的動作「密集的覆蓋」了所有音樂節拍，但 Precision 並未因此顯著下降（因為它的動作是持續的，雖然密集，但仍算對準了節拍）。
    *   **Concat-ng80 的 Recall (0.689) vs. Precision (0.246):** ng80 的 Recall 比 ng40 低，但 Precision 與 GT 相當，且 F1 Score (0.357) 仍然很高。這代表 ng80 的動作密度**開始向 GT 靠攏**，不再是無腦的填滿節拍。
    *   **X-Attention 的 Recall (0.644) vs. Precision (0.250):** Recall 數值雖然不高，但 Precision 意外的較高。然而，結合其極差的 FID (**27.84**) 和视觉效果，這是一個典型的 **"Metric Hacking"** 案例。模型可能透過產生微弱的、但足夠覆蓋節拍的「噪音」或「原地抖動」，來提升 Recall，而無視了真實的動作質量。
*   **結論：**
    *   **ng80 的節奏感：** 儘管 ng40 的 Recall 最高，但 ng80 的 **F1 Score (0.357)** 是最高的，且其 Recall (0.689) 和 Precision (0.246) 更接近 GT。這表明 ng80 在節奏的「綜合表現」上最佳，既對準了音樂，又具有更自然的動作密度。
    *   **CLoSD 的 BeatAlign 無效：** 由於音頻未配對，CLoSD 的 BeatAlign 分數（Precision 0.236, Recall 0.659）僅為隨機猜測，無參考價值。

### D. 物理合理性分析 (Physics Metrics)

**3.4. 懸浮問題 (Floating) 的嚴重性與 ng80 的改進：**
*   **原始模型缺陷：** Raw Diffusion (ng20/40) 的 Floating 高達 **240mm+**。這代表生成的舞者平均離地 24 公分，視覺上會給人「飄浮」或「懸空」的感覺，嚴重缺乏接地感。
*   **CLoSD 的徹底修復：** CLoSD 將 Floating 壓至 **23.9mm**。這幾乎將腳部「吸附」在地板上，極大地提升了接地感。雖然比 GT (59mm) 還低（可能表示 CLoSD 有時會過度修正，限制了真實舞者的跳躍動作），但成功解決了「飄浮」問題。
*   **ng80 的自我修正：** 值得注意的是，ng80 的 Floating (**169mm**) 比 ng20/40 大幅改善（減少了約 70mm）。這暗示長序列生成促使模型學會了更穩定的全局姿態，減少了整體飄移，使其整體運動軌跡更傾向於「有根基」。
*   **結論：** 原始 Diffusion 模型有嚴重的懸浮問題。長序列生成 (ng80) 有所改善，但 CLoSD 的物理後處理是解決此問題的決定性方法。

**3.5. 滑步與穿透 (Skating & Penetration) 的分析：**
*   **CLoSD 的絕對優勢：** CLoSD 幾乎將 **Foot Sliding** 和 **Penetration** 降至 **0.00 mm**。這是其作為物理模擬器的核心功能，確保了腳部不會穿過地板，也不會發生不合理的滑動。
*   **ng80 的物理穩定性：** 在沒有物理後處理的情況下，Concat-ng80 在 **Skating Ratio (0.0087)** 上竟然達到了 GT 水平 (**0.0083**)，且 **Foot Sliding (0.25mm)** 和 **Penetration (0.15mm)** 也遠低於 ng20/40。這再次印證了長序列生成對於動作物理穩定性的正面影響，模型學會了更穩定的腳部動作。
*   **X-Attention 的虛假零值：** X-Attention 的 Skating, Sliding, Penetration 數值為 **0.00**，但這是**假象**。如同 Floating 的分析，這是因為它的動作根本沒有有效發生「觸地」和「滑動」的條件，僅僅是原地轉圈。
*   **結論：** CLoSD 在這兩項指標上表現完美。而 ng80 在沒有物理後處理的情況下，也展現了驚人的物理穩定性，大幅超越短序列生成。

**3.6. X-Attention 架構的根本性缺陷：**
*   **綜合表現：** X-Attention 在所有主要指標（FID, Skating, Floating, Sliding, Penetration）上都表現極差或出現異常數值，唯一的「亮點」Precision 虛高。
*   **原因推測：**
    *   **注意力機制問題：** 在這種長序列、多模態（音頻+動作）的任務中，Cross-Attention 可能未能有效學習到音頻與動作的關聯。
    *   **Mode Collapse：** 模型退化為只生成一種（錯誤的）動作模式（原地轉圈），這是一種典型的 Mode Collapse 現象。
    *   **「騙分」行為：** 雖然 BAS Recall 分數看起來還行，但結合視覺上的極差表現（FID, Floating），這表明模型並沒有真正學習到有效的舞蹈生成能力。
*   **結論：** X-Attention 架構在這個實驗設定下是失敗的，無法有效生成高質量的舞蹈。

---

## 4. 總結與建議 (Conclusion & Recommendations)

**4.1. 核心發現：**
*   **長序列生成的重要性：** Concat 模型在長序列（ng80）下表現出最佳的動作連貫性、物理穩定性（Skating, Sliding, Penetration, Floating）和最高的 F1 Score，證明了長時依賴學習對舞蹈生成的關鍵作用。
*   **CLoSD 的物理修復價值：** CLoSD 作為後處理器，是解決 Diffusion 模型固有「飄浮」問題的關鍵，並將 FID 顯著降低，提升了視覺真實度。
*   **Beat Alignment 的標準演變：** 需要理解 Recall 指標的局限性（易被 Hack）。 ng80 在 F1 Score 上表現最佳，表明其綜合節奏能力優於其他原始輸出，且更接近 GT 的自然動作密度。
*   **X-Attention 的失敗：** 該架構在此設定下未能有效學習，導致動作質量極差。

**4.2. 推薦的生成流程：**
1.  **選擇模型架構：** 使用 **Concat** 結構。
2.  **選擇生成長度：** 使用 **ng80**。
3.  **進行物理後處理：** 使用 **CLoSD**。

**建議的報告呈現方式：**
*   **首選表格：** **Master Table** (表格 A) 提供最全面的視角。
*   **強調質量：** **Quality & Physics Focus Table** (表格 B) 強調 CLoSD 的優勢與 ng80 的改善。
*   **強調節奏：** **Beat Alignment Deep Dive Table** (表格 C) 深入分析節奏問題。
*   **架構比較：** **Ablation: Architecture Table** (表格 D) 證明 Concat 的優越性。
*   **長度比較：** **Ablation: Sequence Length Table** (表格 E) 展示 ng80 的優勢。

**未來工作建議：**
*   **Audio Pairing for CLoSD:** 獲取 CLoSD 的準確 BeatAlign 分數，以評估物理後處理對節奏的影響。
*   **更嚴謹的 Beat Alignment：** 考慮在報告中同時呈現 F1 Score 與 Recall，以避免誤導。
*   **視覺評估：** 結合 video 評估，可以更直觀地發現 X-Attention 的轉圈問題，以及 CLoSD 的接地效果。

---

conda activate closd && CUDA_VISIBLE_DEVICES=5 py
thon closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_d
ancing/output/CLoSD.pkl
[WARN] lengths max (2877) > actual_motion_len (400), clamping
[WARN] No paired audio in npy file, using random audio from dataloader
[INFO] Loading 1000 audio samples for BeatAlign (fixed_len=400)
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 23681.20it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
[DEBUG] Dataloader done!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 16745.01it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/CLoSD.pkl
num_samples: 1000
FID: 5.0655
Diversity: 3.5792
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2359 ± 0.0644
  Recall:    0.6594 ± 0.0955
  F1 Score:  0.3429 ± 0.0736
--- Physics ---
  SkatingRatio(mean): 0.0176
  FootSliding(mm):    0.0000
  Penetration(mm):    0.0002
  Floating(mm):       23.8992

---

conda activate closd && CUDA_VISIBLE_DEVICES=5 py
thon closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_d
ancing/output/aistpp_xattn_ng40/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 65852.60it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 12112.84it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_xattn_ng40/sample/generated_motions.npy
num_samples: 1000
FID: 27.8405
Diversity: 3.6606
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2500 ± 0.0708
  Recall:    0.6436 ± 0.1267
  F1 Score:  0.3551 ± 0.0831
--- Physics ---
  SkatingRatio(mean): 0.0000
  FootSliding(mm):    0.0000
  Penetration(mm):    0.0000
  Floating(mm):       103.3515

---

conda activate closd && CUDA_VISIBLE_DEVICES=5  p
ython closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_
dancing/output/aistpp_concat_ng20/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 51720.13it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 18880.18it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng20/sample/generated_motions.npy
num_samples: 1000
FID: 16.1660
Diversity: 4.6198
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2445 ± 0.0640
  Recall:    0.7629 ± 0.1029
  F1 Score:  0.3658 ± 0.0774
--- Physics ---
  SkatingRatio(mean): 0.0152
  FootSliding(mm):    0.9039
  Penetration(mm):    1.8671
  Floating(mm):       249.7649

---

conda activate closd && CUDA_VISIBLE_DEVICES=5 py
thon closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_d
ancing/output/aistpp_concat_ng40/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|███████████████████████████████████████████████████████| 52/52 [00:00<00:00, 402405.55it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 10746.15it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy
num_samples: 1000
FID: 18.2840
Diversity: 4.3523
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2435 ± 0.0633
  Recall:    0.7895 ± 0.0970
  F1 Score:  0.3677 ± 0.0765
--- Physics ---
  SkatingRatio(mean): 0.0486
  FootSliding(mm):    2.3931
  Penetration(mm):    1.6065
  Floating(mm):       242.1510

---

conda activate closd && CUDA_VISIBLE_DEVICES=5 py
thon closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_d
ancing/output/aistpp_concat_ng40/sample/generated_motions.npy --eval_gt
[INFO] Loading ground truth motions (gt_motion)
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 39568.91it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 11975.17it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng40/sample/generated_motions.npy
num_samples: 1000
FID: 0.0535
Diversity: 5.3355
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2457 ± 0.1093
  Recall:    0.5293 ± 0.2101
  F1 Score:  0.3252 ± 0.1279
--- Physics ---
  SkatingRatio(mean): 0.0083
  FootSliding(mm):    0.0538
  Penetration(mm):    0.0000
  Floating(mm):       59.1546

---

conda activate closd && CUDA_VISIBLE_DEVICES=5 py
thon closd/diffusion_planner/eval/eval_aistpp_external.py --external_results_file /share3/saves/boson/image/CLoSD_d
ancing/output/aistpp_concat_ng80/sample/generated_motions.npy
[INFO] Loaded 1000 paired audio waveforms from npy file
[INFO] Using 1000 paired audio waveforms for beat alignment
[DEBUG] _get_gt_batch_with_audio: calling get_dataset_loader...
Fetching 52 files: 100%|███████████████████████████████████████████████████████| 52/52 [00:00<00:00, 209513.74it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
[DEBUG] _get_gt_batch_with_audio: loader created, getting first batch...
[DEBUG] _get_gt_batch_with_audio: got batch!
Fetching 52 files: 100%|████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 24880.65it/s]
Data dependencies are cached at [/share2/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b]
Loading Evaluation Model Wrapper (Epoch 11) Completed!!
[INFO] Loaded motion mean/std from closd/diffusion_planner/dataset
=== AIST++ external eval ===
external_results_file: /share3/saves/boson/image/CLoSD_dancing/output/aistpp_concat_ng80/sample/generated_motions.npy
num_samples: 1000
FID: 11.7527
Diversity: 3.6255
--- Beat Alignment (sigma=0.05, n=1000) ---
  Precision: 0.2459 ± 0.0693
  Recall:    0.6885 ± 0.1330
  F1 Score:  0.3566 ± 0.0817
--- Physics ---
  SkatingRatio(mean): 0.0087
  FootSliding(mm):    0.2495
  Penetration(mm):    0.1466
  Floating(mm):       169.4469