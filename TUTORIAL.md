# 90° 矽光子彎曲波導：模擬與反向設計 Tutorial

本文件說明如何使用 Meep 建立 90° 彎曲波導的二維 FDTD 模型，並執行本 Repository 提供的三種結構編碼與反向設計流程：

1. 2D Fourier Basis
2. Gaussian Upsampling（GU）
3. β-Variational Autoencoder（β-VAE）

三種方法皆以 Meep 評估候選結構的 TE₀ 模態穿透率，並使用 Factorization Machine（FM）代理模型與 QUBO 最佳化流程搜尋候選結構。

> **重要範圍說明：** 目前程式使用二維 Meep 模型，波導材料區域沿 z 方向以無限延伸（mp.inf）表示。這是用於研究平面內彎曲結構的簡化模型，不等同於具有有限矽核心厚度、埋氧層與上包覆層的完整 3D SOI 波導模擬。本 Tutorial 中的模擬數值也不代表實際製作或量測結果。

---

## 1. 專案目標與目錄

本專案研究如何以不同的結構參數化方式表示 90° 彎曲波導設計區域，再透過 FM、QUBO 與退火搜尋進行反向設計。

Repository 主要包含：

~~~text
Bend_Waveguide_Tutorial/
├── README.md
├── TUTORIAL.md
├── LICENSE
├── Tutorial/
│   ├── 2D_Fourier_Basis/
│   │   ├── 2D_Fourier_10x10.ipynb
│   │   ├── 2D_Fourier_10x10.py
│   │   └── factorization_machine.py
│   ├── Gaussian_Upsampling/
│   │   ├── WG_GU_1000_10x10.ipynb
│   │   ├── WG_GU_1000_10x10.py
│   │   └── factorization_machine.py
│   └── beta_VAE/
│       ├── bVAE_gu.ipynb
│       ├── bVAE_gu.py
│       └── factorization_machine.py
└── shared_modules/
    ├── __init__.py
    ├── factorization_machine.py
    └── meep_utils.py
~~~

每個 Tutorial 資料夾包含對應的 Notebook、Python 主程式及 FM 模型檔案。請從對應資料夾執行程式，讓 Python 能找到該資料夾內的 factorization_machine.py。

---

## 2. 模擬幾何與座標

### 2.1 幾何配置

模擬區域中央為 4.0 µm × 4.0 µm 的可設計區域，輸入與輸出波導臂分別連接在設計區域的左側與上側：

- 輸入波導：位於 −X 側，光由左向右沿 +X 方向入射。
- 輸出波導：由設計區域沿 +Y 方向傳播。
- 設計區域：位於座標原點附近，尺寸為 4.0 µm × 4.0 µm。
- 輸入／輸出波導寬度：0.5 µm。
- PML：配置於模擬計算區域邊界，吸收離開計算域的電磁波。

座標概念如下：

~~~text
                 Output (+Y)
                      ↑
                      │
             ┌────────────────┐
             │                │
             │  Design Region │
             │    4 × 4 µm²   │
             │                │
             └────────────────┘
                      │
                      │
Input (−X) ──────────→
~~~

上圖僅表示輸入與輸出方向，以及中央設計區域的相對位置；實際結構由各編碼方法產生，並非固定的圓弧半徑幾何。

### 2.2 共用物理參數

目前三種主程式採用下列基準參數：

| 參數 | 數值 | 說明 |
|---|---:|---|
| RESOLUTION | 40 | 每 µm 的網格數 |
| DPML | 1.0 µm | PML 厚度 |
| MDM_LC | 4.0 µm | 中央設計區域邊長 |
| WG_LENGTH | 1.0 µm | 固定波導臂長度參數 |
| WG_WIDTH | 0.5 µm | 輸入／輸出波導寬度 |
| SX | 8.0 µm | x 方向計算域尺寸 |
| SY | 8.0 µm | y 方向計算域尺寸 |
| N_SI | 3.48 | 矽折射率 |
| N_SIO2 | 1.44 | 二氧化矽折射率 |
| WL_CEN | 1.55 µm | 中心波長 |
| FCEN | 1 / WL_CEN | Meep 中心頻率 |
| DF | 0.1 × FCEN | Gaussian source 頻寬參數 |
| NFREQ | 1 | 模態監測頻率點數 |

長度單位採 µm。Meep 的頻率使用與幾何長度互相一致的倒長度單位，因此中心波長 1.55 µm 對應的中心頻率為：

\[
f_\mathrm{cen} = \frac{1}{1.55} \approx 0.64516\ \mathrm{\mu m}^{-1}
\]

目前矽與二氧化矽使用固定折射率，未加入波長色散模型。若要將結果用於實際元件設計，應依材料資料、製程堆疊及工作波段重新定義材料模型。

### 2.3 二維模型的適用範圍

主程式中的波導與設計區域使用 z 方向無限延伸的幾何物件。這代表模擬只處理 x-y 平面內的場分佈，沒有明確描述有限厚度矽核心中的垂直場侷限。

因此：

- 可用於學習二維彎曲波導的幾何表示與 FDTD 最佳化流程。
- 可用於比較不同結構編碼方法在同一套二維模型下的結果。
- 不可直接將二維結果視為有限高度 SOI strip waveguide 的完整 3D 模擬結果。
- 若需評估垂直模態分佈、實際 TE/TM 模態或有限厚度效應，應建立 3D 模型或明確採用並驗證有效折射率近似。

---

## 3. Meep 模擬流程

### 3.1 材料與幾何

模擬以二氧化矽作為預設背景材料，並將矽材料配置於輸入波導、輸出波導及候選設計區域。

對於不含色散的簡化材料模型，折射率 n 對應的相對介電常數為：

\[
\varepsilon_r = n^2
\]

設計區域由各方法的解碼器轉換成二維密度矩陣，再透過 Meep MaterialGrid 映射為材料分佈。

### 3.2 TE₀ 模態激發

程式使用 Meep EigenModeSource 激發輸入導波模態，主要設定包含：

- Source：GaussianSource。
- 中心頻率：FCEN。
- 頻寬參數：DF。
- 傳播方向：mp.X。
- 模態編號：eig_band = 1。
- 對稱條件：eig_parity = mp.EVEN_Y。

程式以 Ez 場分量作為衰減判斷及場分佈分析的主要觀察量。專案程式將目標基模標記為 TE₀；但二維模型中的 TE/TM 命名會受到模擬平面與場分量定義影響，因此延伸至其他模擬維度或撰寫論文時，應同時註明場分量及偏振慣例。模態編號與場分量的物理意義應依目前的二維模型及波導截面確認，不應在不同模擬維度間直接假設 band 1 永遠等同於相同的實際 TE₀ 模態。

### 3.3 輸入與輸出監測器

程式使用 mode monitor 擷取模態係數：

- 輸入規範化監測器：位於輸入波導直線段，用於估算入射模態功率。
- 輸出監測器：位於 +Y 方向的輸出波導，用於估算輸出模態功率。

程式以模態係數 alpha 計算穿透率：

\[
T = \frac{|\alpha_\mathrm{out}|^2}
         {|\alpha_\mathrm{in}|^2 + \epsilon}
\]

其中 \(\epsilon\) 為避免分母為零的小常數。各主程式目前使用相同的主要監測概念，但詳細分析與輸出欄位仍依各程式實作為準。

模擬使用場衰減條件停止計算，程式中的代表性設定為：

~~~python
sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, MON_CENTER, 1e-5
    )
)
~~~

### 3.4 目標函數

三種方法的最佳化方向是提高輸出端 TE₀ 模態穿透率。

部分程式定義：

\[
FOM = -T
\]

並以最小化 FOM 的方式搜尋；β-VAE 程式的部分流程則直接以穿透率作為最大化目標。因此閱讀或修改最佳化程式時，必須確認該程式目前採用的目標函數正負號與排序方向。

---

## 4. 三種結構編碼方法

### 4.1 2D Fourier Basis

#### 原理

2D Fourier Basis 以二維正弦基底組合表示設計區域。每個二元變數決定是否啟用一組空間基底，所有啟用基底相加形成連續場，再透過閾值轉成二元材料分佈。

目前程式使用：

| 參數 | 數值 |
|---|---:|
| M_MODES | 10 |
| N_MODES | 10 |
| 二元變數總數 | 100 |
| 中央設計區域 | 4.0 × 4.0 µm² |
| 物理網格解析度 | 160 × 160 |

程式利用 m、n 兩個方向的正弦基底建立結構，並以 0 作為二值化閾值。最終矩陣映射到 Meep MaterialGrid。

#### 最佳化參數

| 參數 | 預設值 |
|---|---:|
| 初始樣本數 | 1,000 |
| 迭代次數 | 100 |
| 每輪新增候選數 | 30 |
| FM 因數維度 K | 10 |
| FM 訓練 epochs | 1,000 |
| Learning rate | 0.001 |
| SA num_reads | 10,000 |
| SA num_sweeps | 1,000 |
| 預設取樣器 | SA |

程式預設使用 SA。程式中保留 QA 取樣路徑，但需具備 D-Wave 相關套件、帳戶憑證與可用 Solver 才能執行。

#### 執行檔案

- Notebook：Tutorial/2D_Fourier_Basis/2D_Fourier_10x10.ipynb
- Python：Tutorial/2D_Fourier_Basis/2D_Fourier_10x10.py

### 4.2 Gaussian Upsampling（GU）

#### 原理

Gaussian Upsampling 先將低解析度二元向量重組為二維網格，再透過 Gaussian convolution 對相鄰設計變數進行平滑，最後使用 Tanh projection 將材料密度推向接近 0 或 1 的值。

主要步驟：

1. 將 100 個二元變數重組成 10 × 10 網格。
2. 使用 Gaussian kernel 進行二維卷積。
3. 使用 Tanh projection 調整密度值。
4. 將密度限制在 [0, 1]。
5. 將密度矩陣映射到 Meep MaterialGrid。

#### 結構編碼參數

| 參數 | 預設值 |
|---|---:|
| GRID_ROWS | 10 |
| GRID_COLS | 10 |
| 二元變數總數 | 100 |
| Gaussian kernel | 5 × 5 |
| Gaussian 標準差 σ | 1.0 |
| Tanh β | 50 |
| Tanh η | 0.5 |
| MaterialGrid smoothing_radius | 1.0 |

Tanh projection 的概念可表示為：

\[
\tilde{\rho} =
\frac{
\tanh(\beta\eta)+\tanh[\beta(\rho-\eta)]
}{
\tanh(\beta\eta)+\tanh[\beta(1-\eta)]
}
\]

其中 \(\rho\) 為 Gaussian 平滑後的密度，\(\beta\) 控制投影陡峭程度，\(\eta\) 為投影閾值。

#### 最佳化參數

| 參數 | 預設值 |
|---|---:|
| 初始樣本數 | 1,000 |
| 迭代次數 | 100 |
| 每輪新增候選數 | 30 |
| FM 因數維度 K | 8 |
| FM 訓練 epochs | 1,000 |
| Learning rate | 0.001 |
| SA num_reads | 1,500 |
| SA num_sweeps | 1,000 |
| 預設取樣器 | SA |

程式保留 D-Wave QA 的條件式程式碼；若 QA 執行失敗，程式會嘗試退回 SA。實際使用前應檢查 D-Wave 套件、帳戶設定及 Solver 權限。

#### 執行檔案

- Notebook：Tutorial/Gaussian_Upsampling/WG_GU_1000_10x10.ipynb
- Python：Tutorial/Gaussian_Upsampling/WG_GU_1000_10x10.py

### 4.3 β-Variational Autoencoder（β-VAE）

#### 原理

β-VAE 先使用合成的二維二元結構建立資料集，再訓練生成式模型學習結構分佈。完成訓練後，將潛在變數交給 FM 代理模型與 SA 搜尋，再使用 Decoder 還原為 64 × 64 設計結構，最後交由 Meep 評估。

目前資料集由隨機雜訊經 Gaussian smoothing 後，再透過閾值二值化產生。程式採用 categorical latent representation，每個潛在變數有兩個類別。

#### VAE 設定

| 參數 | 預設值 |
|---|---:|
| IMG_SIZE | 64 |
| 輸入維度 | 64 × 64 = 4,096 |
| HIDDEN_LAYERS_VAE | [512, 256] |
| LATENT_N | 100 |
| LATENT_K | 2 |
| BVAE_DATA_SAMPLES | 10,000 |
| BVAE_BATCH_SIZE | 64 |
| BVAE_EPOCHS | 30 |
| BVAE_LR | 0.001 |
| β-KL 權重 | 0.01 |
| Gumbel-Softmax 初始溫度 | 1.0 |
| Gumbel-Softmax 最低溫度 | 0.1 |

Decoder 輸出後會進行 Gaussian smoothing（5 × 5、σ = 1.0）及 Tanh projection（β = 30、η = 0.5），再將結果二值化並建立 Meep 幾何。

#### FM 與退火設定

| 參數 | 預設值 |
|---|---:|
| 初始模擬樣本數 | 2,000 |
| 最佳化迭代次數 | 200 |
| 每輪新增候選數 | 30 |
| FM 因數維度 K | 10 |
| FM 訓練 epochs | 1,000 |
| FM Learning rate | 0.001 |
| SA num_reads | 2,000 |
| SA num_sweeps | 1,000 |

目前 β-VAE 程式的候選搜尋流程使用 SA。

#### 執行檔案

- Notebook：Tutorial/beta_VAE/bVAE_gu.ipynb
- Python：Tutorial/beta_VAE/bVAE_gu.py

---

## 5. 環境需求

程式使用的主要套件如下：

| 套件 | 用途 |
|---|---|
| Meep / pymeep | FDTD 電磁模擬 |
| NumPy | 陣列與數值運算 |
| SciPy | Gaussian filter、卷積及插值 |
| Matplotlib | 模擬結果視覺化 |
| PyTorch | Factorization Machine 與 β-VAE |
| mpi4py | Python MPI 介面 |
| MPI implementation | 多行程平行運算 |
| dimod | Binary Quadratic Model / QUBO |
| dwave-samplers | Simulated Annealing |
| dwave-system | 選用；D-Wave Quantum Annealing 連線 |

目前 Repository 尚未提供固定版本的 requirements.txt 或 Conda environment 檔案，因此安裝時應依自己的作業系統、Python 版本及 MPI 環境選擇相容版本。

Meep 官方安裝說明：

- https://meep.readthedocs.io/en/latest/Installation/

建議先在 Python 環境中確認必要套件可以正常匯入，再開始執行完整最佳化。

---

## 6. 執行方式

### 6.1 MPI 平行架構

三種程式都使用 MPI 將候選結構的 FDTD 評估工作分派給 Worker。典型架構為：

- Rank 0（Master）：產生候選解、訓練 FM、建立 QUBO、執行退火搜尋、彙整模擬結果。
- Rank 1 及以上（Workers）：接收候選結構，執行 Meep FDTD，並將結果回傳 Master。

例如：

~~~bash
mpirun -np 31 python your_script.py
~~~

其中 31 代表總 MPI 行程數，通常是 1 個 Master 加上 30 個 Workers。請根據實際 CPU 核心數、記憶體及模擬資源調整，不建議任意提高行程數。

### 6.2 執行 Gaussian Upsampling

~~~bash
cd Tutorial/Gaussian_Upsampling
mpirun -np 31 python WG_GU_1000_10x10.py --name bend_gu
~~~

GU 程式支援透過 --name 指定此次工作的名稱。

### 6.3 執行 2D Fourier Basis

~~~bash
cd Tutorial/2D_Fourier_Basis
mpirun -np 31 python 2D_Fourier_10x10.py --name bend_fourier
~~~

Fourier 程式亦支援透過 --name 指定工作名稱。

### 6.4 執行 β-VAE

~~~bash
cd Tutorial/beta_VAE
mpirun -np 31 python bVAE_gu.py
~~~

β-VAE 程式目前使用程式內的時間戳記建立輸出資料夾，沒有沿用前兩種程式的 --name 參數。

### 6.5 執行 Jupyter Notebook

若要逐步閱讀程式與觀察中間結果，可以在對應環境啟動 Jupyter：

~~~bash
jupyter lab
~~~

再開啟各方法資料夾中的 ipynb 檔案。Notebook 與 Python 主程式的執行方式及輸出路徑可能不同，請依 Notebook 內容確認工作目錄。

---

## 7. 輸出結果

各主程式會依方法建立結果資料夾，保存最佳化歷程、候選結構與模擬分析圖。

| 方法 | 程式中的輸出資料夾 |
|---|---|
| 2D Fourier Basis | Results/ |
| Gaussian Upsampling | Results_Bend/ |
| β-VAE | bVAE_GU_FMQA_時間戳記資料夾 |

依程式設定與執行流程，輸出可能包含：

- 最佳二元結構或最佳潛在變數。
- 最佳化目標值與迭代歷程。
- 候選結構的傳輸率紀錄。
- Meep 電磁場分佈與材料分佈圖。
- 時間統計及訓練紀錄 JSON。

請以實際執行產生的結果檔案為準。不同程式輸出檔名與紀錄內容並不完全相同；若要進行方法比較，建議將原始參數、程式版本、隨機種子及結果一併保存。

---

## 8. 結果比較與數值驗證

若要比較三種編碼方式，建議至少檢查下列項目：

1. **相同物理模型：** 確認計算區域、材料折射率、波導寬度、波長與輸入輸出方向一致。
2. **相同評估方式：** 確認輸入與輸出模態監測器及穿透率定義一致。
3. **一致的 FDTD 設定：** 確認 resolution、PML 厚度及模擬停止條件一致。
4. **一致的模擬預算：** 比較時應控制實際 Meep 評估次數，而不只比較迭代輪數。
5. **隨機性：** 記錄 NumPy、PyTorch 與退火程序的隨機種子；必要時重複多次實驗。
6. **數值收斂：** 提高空間解析度、調整 PML 厚度與計算域尺寸，確認穿透率不會因數值設定改變而大幅波動。

目前三種程式的預設資料量及迭代預算不同，不能直接將各程式單次執行的最佳結果視為完全公平的比較。

此外，模擬解析度 40 pixels/µm 是目前程式的基準值，不代表已經完成網格收斂驗證。正式報告應補充解析度與 PML 收斂測試。

---

## 9. 常見問題

| 現象 | 可能原因 | 建議檢查 |
|---|---|---|
| 找不到 factorization_machine | 從 Repository 根目錄執行，或 Python 搜尋路徑不包含腳本資料夾 | 先 cd 到對應 Tutorial 子資料夾再執行 |
| 找不到 meep | Meep 未安裝，或目前 Python 環境不是安裝 Meep 的環境 | 確認 Python、Conda 環境及 Meep 安裝 |
| MPI 啟動失敗 | MPI implementation 或 mpi4py 環境不相容 | 確認 mpirun 與 mpi4py 使用相同 MPI 環境 |
| 模擬時間過長 | FDTD 網格過大、Worker 數量不合適或模擬停止條件過嚴 | 檢查 resolution、行程數與場衰減條件 |
| 穿透率對解析度敏感 | 空間網格不足以描述材料邊界或幾何特徵 | 提高 resolution 並比較結果 |
| D-Wave QA 無法執行 | 未安裝 D-Wave 套件、憑證未設定或 Solver 不可用 | 檢查 D-Wave Ocean 環境；可先使用預設 SA |

---

## 10. 研究限制與注意事項

- 本專案目前採用二維模型，並未完整描述有限厚度 SOI 波導的垂直場侷限。
- 材料折射率使用常數近似，未納入完整色散、材料吸收與製程變異。
- 共用模擬設定的 NFREQ 為 1，主要對應單一中心頻率的模態監測，不等同完整連續頻譜掃描。
- 初始候選結構與 SA 搜尋具有隨機性，執行結果可能不同。
- 退火搜尋不保證找到全域最佳解；本文所稱的最佳結構是目前已評估樣本中的 best-found solution。
- 所有輸出皆為數值模擬資料，未經實際元件製作或量測驗證。

---

## 11. 參考資料

- Meep Documentation：https://meep.readthedocs.io/en/latest/
- Meep Python Tutorial — Basics：https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
- Meep Eigenmode Source：https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/
- Meep Installation Guide：https://meep.readthedocs.io/en/latest/Installation/
- D-Wave Ocean Documentation：https://docs.dwavequantum.com/en/latest/ocean/index.html
- PyTorch Documentation：https://pytorch.org/docs/stable/index.html

---

**文件狀態：** 本文件依照 Repository 目前的三套程式整理。若後續修改物理參數、模型架構或輸出格式，請同步更新 README 與本 Tutorial。
