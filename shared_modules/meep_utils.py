"""
90° Bend Waveguide — 共用 MEEP 模擬核心模組

此模組封裝所有三種 Encoding 方法（bVAE, 2D Fourier, Gaussian Upsampling）
共用的 MEEP 物理常數、模擬函式。

標準輸入介面：geometry_matrix（list of mp.GeometricObject）
  - 由各主程式中的 encoding-specific 函式負責將優化變數轉換為此格式
  - 轉換結果傳入 evaluate_fom_from_geometry() 或 run_bend_simulation()
"""

import numpy as np
import meep as mp
from scipy.ndimage import zoom

# ==============================================================================
# SECTION 1: 物理常數（所有腳本共用，唯一來源）
# ==============================================================================

# 解析度（MEEP 格式：pixels per µm）
RESOLUTION = 40

# PML 厚度（µm）
DPML = 1.0

# 設計區域邊長（µm）
MDM_LC = 4.0

# 波導尺寸（µm）
WG_LENGTH = 1.0   # 直波導臂長
WG_WIDTH  = 0.5   # 波導寬度（單模 SOI 0.5 µm）

# 模擬胞大小（µm）
# 結構：[PML | WG_ARM | DESIGN_REGION | WG_ARM | PML]
SX = 2 * DPML + MDM_LC + 2 * WG_LENGTH
SY = 2 * DPML + MDM_LC + 2 * WG_LENGTH
CELL = mp.Vector3(SX, SY, 0)

# 材料折射率（SOI）
N_SIO2 = 1.44
N_SI   = 3.48
SIO2_MEDIUM = mp.Medium(index=N_SIO2)
SI_MEDIUM   = mp.Medium(index=N_SI)

# 光學頻率參數（λ = 1.55 µm）
WL_CEN = 1.55
FCEN   = 1.0 / WL_CEN
DF     = 0.1 * FCEN
NFREQ  = 1           # 單頻點監測，避免取錯頻率索引

# [Fix-1] 統一波導中心座標計算（消除 bVAE 腳本中的 DPML/2 偏差）
# 輸入波導（X 方向，左側）：
#   從設計區域左邊界向左延伸 (WG_LENGTH + DPML) / 2
#   確保波導臂完整覆蓋至 PML 內
WG_INPUT_CENTER_X  = -(MDM_LC / 2 + (WG_LENGTH + DPML) / 2)
WG_OUTPUT_CENTER_Y =  (MDM_LC / 2 + (WG_LENGTH + DPML) / 2)

# 源點（距左側 PML 內邊界 0.2 µm）— X 座標先算為純量，再包成 Vector3
SRC_CENTER_X    = -SX / 2 + DPML + 0.2
SRC_CENTER      = mp.Vector3(SRC_CENTER_X, 0)
SRC_SIZE        = mp.Vector3(0, WG_WIDTH)

# 輸出監測器（距上側 PML 內邊界 0.2 µm）
MON_CENTER  = mp.Vector3(0, SY / 2 - DPML - 0.2)
MON_SIZE    = mp.Vector3(WG_WIDTH * 3, 0)

# 規範化監測器（距源點右側 0.5 µm）
NORM_MON_CENTER = mp.Vector3(SRC_CENTER_X + 0.5, 0)

# ==============================================================================
# SECTION 2: 幾何矩陣驗證與調整（相容性保護層）
# ==============================================================================

def validate_geometry_matrix(
    matrix: np.ndarray,
    target_rows: int,
    target_cols: int,
    clip_to_01: bool = True,
) -> np.ndarray:
    """
    驗證並自動調整 geometry matrix 使其可安全傳入 MEEP MaterialGrid。

    參數
    ----
    matrix : np.ndarray
        任意形狀的 2D 陣列，形狀 (rows, cols)。
        - 2D Fourier：嚴格 0/1 二值，形狀 = (PHYSICAL_GRID_ROWS, PHYSICAL_GRID_COLS)
        - Gaussian Upsampling：0~1 連續值，形狀 = (grid_rows, grid_cols)（可變）
        - bVAE：0/1 二值，形狀 = (64, 64) 固定
    target_rows : int
        MEEP MaterialGrid 期望的列數（Y 方向 pixel 數）。
    target_cols : int
        MEEP MaterialGrid 期望的欄數（X 方向 pixel 數）。
    clip_to_01 : bool
        是否將數值範圍強制限制在 [0, 1]，預設 True。

    回傳
    ----
    np.ndarray
        形狀為 (target_rows, target_cols) 的 float64 陣列，值域 [0, 1]。

    修復說明
    --------
    [Fix-2] 不同 encoding 產生的矩陣邊長可能不一致：
      - 2D Fourier 16x16 → PHYSICAL_GRID = int(4.0 * 40) = 160×160
      - GU 8x8 → create_projected_geometry 使用 8×8，傳入 MaterialGrid(8,8)
      - 兩者都能正常工作，因為 MEEP MaterialGrid 會自行插值到物理解析度
      但若有邊長不符預期的情況，此函式執行 zoom 插值修正。
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim == 1:
        # 嘗試推斷為正方形
        side = int(np.round(np.sqrt(matrix.size)))
        if side * side == matrix.size:
            matrix = matrix.reshape(side, side)
        else:
            raise ValueError(
                f"[meep_utils] validate_geometry_matrix: "
                f"無法將長度 {matrix.size} 的 1D 向量重塑為 2D 矩陣。"
                f"請在呼叫前先 reshape。"
            )

    if matrix.ndim != 2:
        raise ValueError(
            f"[meep_utils] validate_geometry_matrix: "
            f"期望 2D 矩陣，收到 {matrix.ndim}D 陣列。"
        )

    rows, cols = matrix.shape

    # [Fix-2] 自動縮放：若尺寸不符目標，用雙線性插值調整
    if rows != target_rows or cols != target_cols:
        zoom_r = target_rows / rows
        zoom_c = target_cols / cols
        matrix = zoom(matrix, (zoom_r, zoom_c), order=1)
        matrix = matrix[:target_rows, :target_cols]  # 防止浮點進位多出 1 pixel

    if clip_to_01:
        matrix = np.clip(matrix, 0.0, 1.0)

    return matrix.astype(np.float64)


# ==============================================================================
# SECTION 3: MEEP 模擬核心
# ==============================================================================

def run_bend_simulation(
    geometry: list,
    fcen: float = FCEN,
    df: float = DF,
    res: int = RESOLUTION,
) -> float:
    """
    執行 90° Bend 波導 FDTD 模擬，回傳 TE0 穿透率（正值 0~1）。

    此函式為所有 encoding 方法的共用模擬核心。
    呼叫前，各主程式需自行完成「優化變數 → geometry list」的轉換。

    參數
    ----
    geometry : list of mp.GeometricObject
        設計區域的 MEEP 幾何物件列表。
    fcen : float
        中心頻率（預設為 WL_CEN=1.55 µm 對應之頻率）。
    df : float
        頻寬。
    res : int
        模擬解析度。

    回傳
    ----
    float
        TE0 穿透率，範圍 [0, 1]。模擬失敗時回傳 0.0。

    波導拓撲說明
    -----------
    輸入：從 -X 方向（左側）注入 TE0（Ez 偏振，mp.EVEN_Y 奇偶）
    輸出：從 +Y 方向（上側）量測 TE0 穿透率

        ↑ (output, +Y)
        |
        |  [Design Region]  ─── (input, -X) →
        |
    座標原點在設計區域中心。

    [Fix-1] 波導中心計算：
      輸入臂中心 X = -(MDM_LC/2 + (WG_LENGTH+DPML)/2)
      輸出臂中心 Y = +(MDM_LC/2 + (WG_LENGTH+DPML)/2)
      此公式確保波導臂完整延伸至 PML 邊界，無座標偏移。
    """
    mp.verbosity(0)

    # 附加標準固定波導臂（若呼叫方沒有附加）
    fixed_geometry = [
        mp.Block(
            size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf),
            center=mp.Vector3(WG_INPUT_CENTER_X, 0),
            material=SI_MEDIUM,
        ),
        mp.Block(
            size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf),
            center=mp.Vector3(0, WG_OUTPUT_CENTER_Y),
            material=SI_MEDIUM,
        ),
    ]
    full_geometry = geometry + fixed_geometry

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=SRC_CENTER,
            size=SRC_SIZE,
            direction=mp.X,
            eig_band=1,
            eig_parity=mp.EVEN_Y,
        )
    ]

    sim = mp.Simulation(
        cell_size=CELL,
        boundary_layers=[mp.PML(DPML)],
        geometry=full_geometry,
        sources=sources,
        resolution=res,
        default_material=SIO2_MEDIUM,
    )

    # 規範化監測器（量測入射功率 & 反射）
    norm_flux = sim.add_mode_monitor(
        fcen, df, NFREQ,
        mp.ModeRegion(center=NORM_MON_CENTER, size=SRC_SIZE),
    )

    # 輸出監測器（量測穿透功率）
    flux_out = sim.add_mode_monitor(
        fcen, df, NFREQ,
        mp.ModeRegion(center=MON_CENTER, size=MON_SIZE),
        direction=mp.Y,
    )

    try:
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                50, mp.Ez, MON_CENTER, 1e-5
            )
        )
    except Exception as e:
        print(f"[meep_utils] run_bend_simulation: MEEP 運行時錯誤: {e}")
        return 0.0

    try:
        res_input = sim.get_eigenmode_coefficients(
            norm_flux, [1], eig_parity=mp.EVEN_Y
        )
        input_power = np.abs(res_input.alpha[0, 0, 0]) ** 2 + 1e-12

        res_out = sim.get_eigenmode_coefficients(flux_out, [1])
        output_power = np.abs(res_out.alpha[0, 0, 0]) ** 2

        transmission = float(np.clip(output_power / input_power, 0.0, 1.0))
    except Exception as e:
        print(f"[meep_utils] run_bend_simulation: 模態係數提取失敗: {e}")
        return 0.0

    return transmission


def evaluate_fom_from_geometry(
    geometry: list,
    fcen: float = FCEN,
    df: float = DF,
    res: int = RESOLUTION,
) -> float:
    """
    FOM 封裝函式，供「最小化」優化器（SA/QUBO）使用。

    回傳值
    ------
    float
        FOM = -transmission，範圍 [-1, 0]。
    """
    transmission = run_bend_simulation(geometry, fcen=fcen, df=df, res=res)
    return -transmission


# ==============================================================================
# SECTION 4: 對外公開的模組級符號（供 `from shared_modules.meep_utils import *` 使用）
# ==============================================================================

__all__ = [
    # 物理常數
    "RESOLUTION", "DPML", "MDM_LC", "WG_LENGTH", "WG_WIDTH",
    "SX", "SY", "CELL",
    "N_SIO2", "N_SI", "SIO2_MEDIUM", "SI_MEDIUM",
    "WL_CEN", "FCEN", "DF", "NFREQ",
    # 幾何座標（統一計算，消除各腳本中的重複定義差異）
    "WG_INPUT_CENTER_X", "WG_OUTPUT_CENTER_Y",
    "SRC_CENTER", "SRC_SIZE", "MON_CENTER", "MON_SIZE", "NORM_MON_CENTER",
    # 核心函式
    "run_bend_simulation",
    "evaluate_fom_from_geometry",
    "validate_geometry_matrix",
]
