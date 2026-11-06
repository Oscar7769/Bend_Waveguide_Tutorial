import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dimod
from dwave.samplers import SimulatedAnnealingSampler

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Warning: dwave-ocean-sdk not installed. Running in Simulated Annealing (SA) mode only.")

import meep as mp

from factorization_machine import FactorizationMachine

import meep as mp
import numpy as np
from scipy.ndimage import zoom

RESOLUTION = 40
DPML = 1.0
MDM_LC = 4.0
WG_LENGTH = 1.0
WG_WIDTH  = 0.5
SX = 2 * DPML + MDM_LC + 2 * WG_LENGTH
SY = 2 * DPML + MDM_LC + 2 * WG_LENGTH
CELL = mp.Vector3(SX, SY, 0)
N_SIO2 = 1.44
N_SI   = 3.48
SIO2_MEDIUM = mp.Medium(index=N_SIO2)
SI_MEDIUM   = mp.Medium(index=N_SI)
WL_CEN = 1.55
FCEN   = 1.0 / WL_CEN
DF     = 0.1 * FCEN
NFREQ  = 1

WG_INPUT_CENTER_X  = -(MDM_LC / 2 + (WG_LENGTH + DPML) / 2)
WG_OUTPUT_CENTER_Y =  (MDM_LC / 2 + (WG_LENGTH + DPML) / 2)
SRC_CENTER_X    = -SX / 2 + DPML + 0.2
SRC_CENTER      = mp.Vector3(SRC_CENTER_X, 0)
SRC_SIZE        = mp.Vector3(0, WG_WIDTH)
MON_CENTER  = mp.Vector3(0, SY / 2 - DPML - 0.2)
MON_SIZE    = mp.Vector3(WG_WIDTH * 3, 0)
NORM_MON_CENTER = mp.Vector3(SRC_CENTER_X + 0.5, 0)

def validate_geometry_matrix(matrix, target_rows, target_cols, clip_to_01=True):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        side = int(np.round(np.sqrt(matrix.size)))
        if side * side == matrix.size:
            matrix = matrix.reshape(side, side)
        else:
            raise ValueError(f"無法將長度 {matrix.size} 的 1D 向量重塑為 2D 矩陣。")
    if matrix.ndim != 2:
        raise ValueError(f"期望 2D 矩陣，收到 {matrix.ndim}D 陣列。")
    rows, cols = matrix.shape
    if rows != target_rows or cols != target_cols:
        zoom_r = target_rows / rows
        zoom_c = target_cols / cols
        matrix = zoom(matrix, (zoom_r, zoom_c), order=1)
        matrix = matrix[:target_rows, :target_cols]
    if clip_to_01:
        matrix = np.clip(matrix, 0.0, 1.0)
    return matrix.astype(np.float64)

def run_bend_simulation(geometry, fcen=FCEN, df=DF, res=RESOLUTION):
    mp.verbosity(0)
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf), center=mp.Vector3(WG_INPUT_CENTER_X, 0), material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf), center=mp.Vector3(0, WG_OUTPUT_CENTER_Y), material=SI_MEDIUM),
    ]
    full_geometry = geometry + fixed_geometry
    sources = [
        mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=df), center=SRC_CENTER, size=SRC_SIZE, direction=mp.X, eig_band=1, eig_parity=mp.EVEN_Y)
    ]
    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)], geometry=full_geometry, sources=sources, resolution=res, default_material=SIO2_MEDIUM)
    norm_flux = sim.add_mode_monitor(fcen, df, NFREQ, mp.ModeRegion(center=NORM_MON_CENTER, size=SRC_SIZE))
    flux_out = sim.add_mode_monitor(fcen, df, NFREQ, mp.ModeRegion(center=MON_CENTER, size=MON_SIZE), direction=mp.Y)
    try:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, MON_CENTER, 1e-5))
    except Exception as e:
        print(f"run_bend_simulation: MEEP 運行時錯誤: {e}")
        return 0.0
    try:
        res_input = sim.get_eigenmode_coefficients(norm_flux, [1], eig_parity=mp.EVEN_Y)
        input_power = np.abs(res_input.alpha[0, 0, 0]) ** 2 + 1e-12
        res_out = sim.get_eigenmode_coefficients(flux_out, [1])
        output_power = np.abs(res_out.alpha[0, 0, 0]) ** 2
        transmission = float(np.clip(output_power / input_power, 0.0, 1.0))
    except Exception as e:
        print(f"run_bend_simulation: 模態係數提取失敗: {e}")
        return 0.0
    return transmission

def evaluate_fom_from_geometry(geometry, fcen=FCEN, df=DF, res=RESOLUTION):
    transmission = run_bend_simulation(geometry, fcen=fcen, df=df, res=res)
    return -transmission

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

WORK_TAG = 1
STOP_TAG = 2

PHYSICAL_GRID_ROWS = int(MDM_LC * RESOLUTION)
PHYSICAL_GRID_COLS = int(MDM_LC * RESOLUTION)

THRESHOLD_C = 0.0

def spectral_reconstruction(alpha_vector, m_modes, n_modes, grid_rows, grid_cols):
    
    x = np.linspace(0, 1, grid_cols)
    y = np.linspace(0, 1, grid_rows)
    X, Y = np.meshgrid(x, y)
    
    field_sum = np.zeros((grid_rows, grid_cols))
    
    idx = 0
    for m in range(1, m_modes + 1):
        for n in range(1, n_modes + 1):
            if idx < len(alpha_vector):
                coef = alpha_vector[idx]
                if coef == 1:
                    field_sum += np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
            idx += 1
            
    binary_structure = np.where(field_sum > THRESHOLD_C, 1.0, 0.0)
    return binary_structure

def create_geometry_from_spectrum(alpha_vector, m_modes, n_modes):
    
    raw_structure = spectral_reconstruction(alpha_vector, m_modes, n_modes, PHYSICAL_GRID_ROWS, PHYSICAL_GRID_COLS)
    
    final_structure = raw_structure
    
    weights = final_structure.flatten()
    
    material_grid = mp.MaterialGrid(mp.Vector3(PHYSICAL_GRID_COLS, PHYSICAL_GRID_ROWS),
                                    SIO2_MEDIUM, SI_MEDIUM, weights=weights)

    design_block = mp.Block(size=mp.Vector3(MDM_LC, MDM_LC, mp.inf), center=mp.Vector3(), material=material_grid)
    return [design_block]

def generate_random_spectrum(m_modes, n_modes):
    num_vars = m_modes * n_modes
    return np.random.randint(0, 2, size=num_vars)

def evaluate_mdm_fom(binary_vector, m_modes, n_modes):
    
    design_geometry = create_geometry_from_spectrum(binary_vector, m_modes, n_modes)
    return evaluate_fom_from_geometry(design_geometry)

def perform_detailed_final_analysis(best_config, m_modes, n_modes, output_folder):
    print(f"\n>>> Starting Final Detailed MEEP Analysis (TE0 Bend) <<<")
    
    mdm_structure = create_geometry_from_spectrum(best_config, m_modes, n_modes)
    
    input_wg_center = mp.Vector3(-MDM_LC/2 - WG_LENGTH/2 - DPML/2, 0)
    output_wg_center = mp.Vector3(0, MDM_LC/2 + WG_LENGTH/2 + DPML/2)
    
    fixed_geometry = [
        mp.Block(size=mp.Vector3(WG_LENGTH + DPML + 0.1, WG_WIDTH, mp.inf),
                 center=input_wg_center, material=SI_MEDIUM),
        mp.Block(size=mp.Vector3(WG_WIDTH, WG_LENGTH + DPML + 0.1, mp.inf),
                 center=output_wg_center, material=SI_MEDIUM),
    ]
    full_geometry = mdm_structure + fixed_geometry
    
    src_center = mp.Vector3(-SX/2 + DPML + 0.2, 0)
    src_size = mp.Vector3(0, WG_WIDTH)
    monitor_center = mp.Vector3(0, SY/2 - DPML - 0.2)
    monitor_size = mp.Vector3(WG_WIDTH * 3, 0)
    
    sources = [mp.EigenModeSource(src=mp.GaussianSource(FCEN, fwidth=DF),
                                  center=src_center, size=src_size, direction=mp.X,
                                  eig_band=1, eig_parity=mp.EVEN_Y)]

    sim = mp.Simulation(cell_size=CELL, boundary_layers=[mp.PML(DPML)],
                        geometry=full_geometry, sources=sources,
                        resolution=RESOLUTION, default_material=SIO2_MEDIUM)
    
    norm_flux = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=mp.Vector3(src_center.x + 0.5, 0), size=src_size))
    flux_out = sim.add_mode_monitor(FCEN, DF, NFREQ, mp.ModeRegion(center=monitor_center, size=monitor_size), direction=mp.Y)
    
    dft_monitor = sim.add_dft_fields([mp.Ez], FCEN, FCEN, 1, center=mp.Vector3(), size=mp.Vector3(SX, SY))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, monitor_center, 1e-5))
    
    ez_dft_data = sim.get_dft_array(dft_monitor, mp.Ez, 0)
    eps_data = sim.get_epsilon()
    intensity = np.abs(ez_dft_data)**2
    
    res_input = sim.get_eigenmode_coefficients(norm_flux, [1], eig_parity=mp.EVEN_Y)
    input_power = np.abs(res_input.alpha[0, 0, 0])**2 + 1e-12
    refl_power = np.abs(res_input.alpha[0, 0, 1])**2
    
    res_out = sim.get_eigenmode_coefficients(flux_out, [1])
    output_power = np.abs(res_out.alpha[0, 0, 0])**2
    
    trans = output_power / input_power
    refl = refl_power / input_power
    loss = 1.0 - trans - refl
    trans_db = 10 * np.log10(trans + 1e-9)
    
    print(f"  [Result] Transmission: {trans:.4f} ({trans_db:.2f} dB)")
    print(f"  [Result] Reflection:   {refl:.4f}")
    print(f"  [Result] Loss/Scatter: {loss:.4f}")
    
    x = np.linspace(-SX/2, SX/2, intensity.shape[0])
    y = np.linspace(-SY/2, SY/2, intensity.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(intensity.T, extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap='jet', origin='lower')
    
    ax.contour(eps_data.T, extent=[x.min(), x.max(), y.min(), y.max()],
               levels=[(N_SI**2+N_SIO2**2)/2], colors='white', alpha=0.5, linewidths=1, origin='lower')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label=r'Intensity $|E_z|^2$')
    ax.set_title(f"Optimized Final Structure")
    ax.set_xlabel(r"x ($\mu$m)", fontsize=14)
    ax.set_ylabel(r"y ($\mu$m)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "final_analysis_TE0_field.png"), dpi=150)
    plt.close(fig)

    x_mask = (x >= -MDM_LC/2) & (x <= MDM_LC/2)
    y_mask = (y >= -MDM_LC/2) & (y <= MDM_LC/2)
    eps_design = eps_data[np.ix_(x_mask, y_mask)]
    
    threshold_eps = (N_SI**2 + N_SIO2**2) / 2
    binary_design = (eps_design > threshold_eps).astype(int)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_design.T, extent=[-MDM_LC/2, MDM_LC/2, -MDM_LC/2, MDM_LC/2],
              cmap='gray_r', origin='lower')
    ax.set_title("Smoothed Binary Structure", fontsize=16)
    ax.set_xlabel(r"x ($\mu$m)", fontsize=14)
    ax.set_ylabel(r"y ($\mu$m)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Smoothed_Binary_Structure.png"), dpi=300)
    plt.close(fig)

    return {"Transmission": float(trans), "Reflection": float(refl)}

def plot_optimization_trajectory(foms, split_index, adding_num, output_folder):
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(foms))
    transmissions = np.abs(foms)
    
    plt.scatter(indices[:split_index], transmissions[:split_index],
                s=5, c='red', alpha=0.6, label='Initial Samples')
                
    if len(foms) > split_index:
        plt.scatter(indices[split_index:], transmissions[split_index:],
                    s=5, c='blue', alpha=0.6, label='FMQA Optimized Samples')
    
    plt.axvline(x=split_index, color='red', linestyle='--', linewidth=2, label='Opt Start')
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Transmission (TE0)', fontsize=14)
    plt.title('Optimization Trajectory', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ax = plt.gca()
    if adding_num > 0 and len(foms) > split_index:
        max_iter = max(0, (len(foms) - split_index) // adding_num)
        if max_iter > 0:
            step = max(1, max_iter // 10)
            iter_ticks = np.arange(0, max_iter + 1, step)
            
            physical_ticks = split_index + iter_ticks * adding_num
            ax.set_xticks(physical_ticks)
            ax.set_xticklabels([str(t) for t in iter_ticks])
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "transmission_trajectory.png"), dpi=150)
    plt.close()

def plot_fom_history(fom_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fom_history)), fom_history, marker='o', linestyle='-', color='b')
    plt.title("Best FOM vs Iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Best FOM (Transmission)", fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fom_evolution.png"), dpi=300)
    plt.close()

def plot_time_statistics_pie_chart(total_fm, total_sa, total_fdtd, other, output_folder):
    labels = ['FM Training', 'Annealing Sampling', 'FDTD Simulation', 'Other']
    sizes = [total_fm, total_sa, total_fdtd, max(0, other)]
    all_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    total = sum(sizes)
    filtered_sizes = []
    filtered_labels = []
    filtered_colors = []
    for s, l, c in zip(sizes, labels, all_colors):
        if s > 0.1:
            filtered_sizes.append(s)
            pct = (s / total) * 100 if total > 0 else 0
            filtered_labels.append(f"{l} ({pct:.1f}%)")
            filtered_colors.append(c)
            
    fig, ax = plt.subplots(figsize=(10, 8))
    explode = [0.01] * len(filtered_sizes)
    
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct > 3 else ''
        
    wedges, texts, autotexts = ax.pie(filtered_sizes,
                                      autopct=my_autopct,
                                      startangle=140,
                                      colors=filtered_colors,
                                      explode=explode,
                                      textprops={'fontsize': 16})
    plt.setp(autotexts, size=16, weight="bold")
    
    lgd = ax.legend(wedges, filtered_labels, title="Runtime Components",
                    loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=14, title_fontsize=16)

    ax.set_title("Total Runtime Distribution", fontsize=18)
    plt.savefig(os.path.join(output_folder, "time_distribution.png"),
                dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plot_chain_break_history(history, output_folder):
    
    if not history.get('chain_break_trends'):
        return

    avg_breaks = history['chain_break_trends']['avg']
    max_breaks = history['chain_break_trends']['max']
    iterations = range(1, len(avg_breaks) + 1)

    if len(iterations) == 0: return

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, avg_breaks, 'o-', label='Avg Chain Break')
    plt.plot(iterations, max_breaks, 'x--', label='Max Chain Break')
    plt.xlabel('Iteration')
    plt.ylabel('Chain Break Fraction')
    plt.title('Quantum Annealing Chain Break History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "chain_break_history.png"))
    plt.close()

def train_fm_model(model, X_train, Y_train, num_epoch, learning_rate, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    device = next(model.parameters()).device
    X_tensor = torch.from_numpy(X_train).float().to(device)
    
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-8
    Y_scaled = (Y_train - y_mean) / y_std
    Y_tensor = torch.from_numpy(Y_scaled).float().view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    bs = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
    loss_history = []
    start_time = time.time()
    model.train()
    
    print(f"  [FM Training] Dataset Size: {len(X_train)} samples")
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch == 0 or (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    - Epoch [{epoch + 1:3d}/{num_epoch}], Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
    return model, loss_history, time.time() - start_time, (y_mean, y_std)

def worker_node():
    grid_dims = comm.bcast(None, root=0)
    mp.verbosity(0)
    while True:
        status = MPI.Status()
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG: break
        task_idx, config = data
        fom = evaluate_mdm_fom(config, grid_dims['m'], grid_dims['n'])
        comm.send((task_idx, fom), dest=0, tag=WORK_TAG)

def parallel_evaluate(tasks, fom_cache, m_modes, n_modes):
    num_tasks = len(tasks)
    results = [None] * num_tasks
    tasks_to_run = []
    
    for i, config in enumerate(tasks):
        c_tuple = tuple(config)
        if c_tuple in fom_cache:
            results[i] = fom_cache[c_tuple]
        else:
            tasks_to_run.append((i, config))

    if not tasks_to_run: return results

    num_to_run = len(tasks_to_run)
    sent_jobs = 0
    jobs_done = 0
    
    active_workers = min(size - 1, num_to_run)
    if size == 1:
        print("Warning: Running in Serial mode (Size=1). Use 'mpirun -np >1' for parallel.")
        for task_idx, config in tasks_to_run:
            results[task_idx] = evaluate_mdm_fom(config, m_modes, n_modes)
            fom_cache[tuple(config)] = results[task_idx]
        return results

    for worker_rank in range(1, active_workers + 1):
        comm.send(tasks_to_run[sent_jobs], dest=worker_rank, tag=WORK_TAG)
        sent_jobs += 1

    while jobs_done < num_to_run:
        status = MPI.Status()
        task_idx, fom = comm.recv(source=MPI.ANY_SOURCE, tag=WORK_TAG, status=status)
        results[task_idx] = fom
        fom_cache[tuple(tasks[task_idx])] = fom
        jobs_done += 1
        
        if sent_jobs < num_to_run:
            comm.send(tasks_to_run[sent_jobs], dest=status.Get_source(), tag=WORK_TAG)
            sent_jobs += 1
            
    return results

def master_node():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='bend_te0', help='Job Name')
    args = parser.parse_args()
    job_name = args.name

    start_total = time.time()
    device = torch.device("cpu")
    print(f"--- Master Node Started on {device.type} (Job: {job_name}) ---")

    PARAMS = {
        'M_MODES': 10,
        'N_MODES': 10,
        'INIT_DATASET_SIZE': 1000,
        'ITERATIONS': 100,
        'ADDING_NUM': 30,
        'NUM_EPOCHS': 1000,
        'LEARNING_RATE': 0.001,
        'NUM_READS': 10000,
        'K_FACTOR': 10,
        'SWEEP_NUM': 1000,
        'SAMPLER_TYPE': 'SA'
    }
    NUM_VARS = PARAMS['M_MODES'] * PARAMS['N_MODES']
    print(f"--- Vars: {NUM_VARS}, Sampler: {PARAMS['SAMPLER_TYPE']} ---")

    BASE_PATH = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"2D_fourier_{PARAMS['M_MODES']}x{PARAMS['N_MODES']}_{PARAMS['INIT_DATASET_SIZE']}_{PARAMS['ITERATIONS']}X{PARAMS['ADDING_NUM']}_{timestamp}"
    RESULTS_DIR = os.path.join(BASE_PATH, "Results", folder_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if size > 1:
        comm.bcast({'m': PARAMS['M_MODES'], 'n': PARAMS['N_MODES']}, root=0)
    
    print(f"--- Generating Initial Dataset ({PARAMS['INIT_DATASET_SIZE']}) ---")
    configs = np.array([generate_random_spectrum(PARAMS['M_MODES'], PARAMS['N_MODES']) for _ in range(PARAMS['INIT_DATASET_SIZE'])])
    fom_cache = {}
    foms = np.array(parallel_evaluate(configs, fom_cache, PARAMS['M_MODES'], PARAMS['N_MODES']))
    
    best_fom = np.min(foms)
    best_config = configs[np.argmin(foms)]
    print(f"Initial Best Transmission: {-best_fom * 100:.2f}%\n")
    
    history = {
        'chain_break_trends': {'avg': [], 'max': []},
        'timing_metrics': {'fm_train_time': [], 'sa_time': [], 'new_data_sim_time': []},
        'best_fom_history': [float(-best_fom)]
    }

    model = FactorizationMachine(input_size=NUM_VARS, factorization_size=PARAMS['K_FACTOR']).to(device)

    for i in range(PARAMS['ITERATIONS']):
        print(f"\n=== Iteration {i+1}/{PARAMS['ITERATIONS']} ===")
        
        model, losses, t_train, stats = train_fm_model(model, configs, foms, PARAMS['NUM_EPOCHS'], PARAMS['LEARNING_RATE'])
        history['timing_metrics']['fm_train_time'].append(t_train)
        
        bias, h, Q = model.get_bhQ()
        Q_dict = {(r, c): Q[r, c] for r in range(Q.shape[0]) for c in range(r+1, Q.shape[1]) if Q[r, c] != 0}
        bqm = dimod.BinaryQuadraticModel(h, Q_dict, bias, dimod.BINARY)
        
        print("  [Sampling]...")
        t_sa_start = time.time()
        if PARAMS['SAMPLER_TYPE'] == 'QA':
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], label=f"{job_name}_{i}")
                
                if 'chain_break_fraction' in sampleset.record.dtype.names:
                    breaks = sampleset.record['chain_break_fraction']
                    history['chain_break_trends']['avg'].append(float(np.mean(breaks)))
                    history['chain_break_trends']['max'].append(float(np.max(breaks)))
            except Exception as e:
                print(f"  QA Failed ({e}), falling back to SA.")
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=PARAMS['SWEEP_NUM'])
        else:
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=PARAMS['NUM_READS'], num_sweeps=PARAMS['SWEEP_NUM'])
        history['timing_metrics']['sa_time'].append(time.time() - t_sa_start)
            
        sampleset = sampleset.aggregate()
        best_samples = sampleset.record['sample'][np.argsort(sampleset.record['energy'])]
        
        unique_new = [s for s in best_samples if tuple(s) not in fom_cache]
        new_configs_list = [list(x) for x in unique_new[:PARAMS['ADDING_NUM']]]
        
        shortfall = PARAMS['ADDING_NUM'] - len(new_configs_list)
        if shortfall > 0:
            print(f"  [Notice] Sampler only yielded {len(new_configs_list)} unique configs. Auto-filling {shortfall} random configs.")
            existing_set = set(fom_cache.keys()) | {tuple(x) for x in new_configs_list}
            attempts = 0
            while len(new_configs_list) < PARAMS['ADDING_NUM'] and attempts < shortfall * 10:
                rand_c = generate_random_spectrum(PARAMS['M_MODES'], PARAMS['N_MODES'])
                if tuple(rand_c) not in existing_set:
                    new_configs_list.append(rand_c.tolist())
                    existing_set.add(tuple(rand_c))
                attempts += 1
                
        new_configs = np.array(new_configs_list)
        
        if len(new_configs) > 0:
            print(f"  Evaluating {len(new_configs)} new candidates...")
            t_sim_start = time.time()
            new_foms = parallel_evaluate(new_configs, fom_cache, PARAMS['M_MODES'], PARAMS['N_MODES'])
            history['timing_metrics']['new_data_sim_time'].append(time.time() - t_sim_start)
            
            configs = np.vstack([configs, new_configs])
            foms = np.concatenate([foms, new_foms])
            
            curr_best = np.min(foms)
            if curr_best < best_fom:
                best_fom = curr_best
                best_config = configs[np.argmin(foms)]
                
            print(f"  Added {len(new_configs)} samples. Global Best T: {-best_fom:.4f}")
        else:
            print("  No new unique candidates found.")
            
        history['best_fom_history'].append(float(-best_fom))

    for i in range(1, size): comm.send(None, dest=i, tag=STOP_TAG)
    
    print("\n === Finalizing ===\n")
    total_time_so_far = time.time() - start_total
    print(f"Elapsed run time = {total_time_so_far:.4f} s")
    
    plot_optimization_trajectory(foms, PARAMS['INIT_DATASET_SIZE'], PARAMS['ADDING_NUM'], RESULTS_DIR)
    plot_chain_break_history(history, RESULTS_DIR)
    
    final_metrics = perform_detailed_final_analysis(best_config, PARAMS['M_MODES'], PARAMS['N_MODES'], RESULTS_DIR)
    
    plot_fom_history(history['best_fom_history'], RESULTS_DIR)
    np.save(os.path.join(RESULTS_DIR, "best_config.npy"), best_config)
    np.save(os.path.join(RESULTS_DIR, "all_foms_history.npy"), np.array(foms))
    
    total_time = time.time() - start_total
    
    total_fm_time = np.sum(history['timing_metrics']['fm_train_time']) if history['timing_metrics']['fm_train_time'] else 0
    total_sa_time = np.sum(history['timing_metrics']['sa_time']) if history['timing_metrics']['sa_time'] else 0
    total_fdtd_time = np.sum(history['timing_metrics']['new_data_sim_time']) if history['timing_metrics']['new_data_sim_time'] else 0
    
    other_time = total_time - total_fm_time - total_sa_time - total_fdtd_time
    plot_time_statistics_pie_chart(total_fm_time, total_sa_time, total_fdtd_time, other_time, RESULTS_DIR)
    
    avg_fm_time = np.mean(history['timing_metrics']['fm_train_time']) if history['timing_metrics']['fm_train_time'] else 0
    avg_sa_time = np.mean(history['timing_metrics']['sa_time']) if history['timing_metrics']['sa_time'] else 0
    avg_fdtd_time = np.mean(history['timing_metrics']['new_data_sim_time']) if history['timing_metrics']['new_data_sim_time'] else 0
    
    report = {
        "parameters": {
            **PARAMS,
            "RESOLUTION": RESOLUTION,
            "DPML": DPML,
            "MDM_LC": MDM_LC,
            "WG_LENGTH": WG_LENGTH,
            "WG_WIDTH": WG_WIDTH,
            "N_SIO2": N_SIO2,
            "N_SI": N_SI,
            "WL_CEN": WL_CEN,
            "FCEN": FCEN,
            "DF": DF,
            "NFREQ": NFREQ,
            "PHYSICAL_GRID_ROWS": PHYSICAL_GRID_ROWS,
            "PHYSICAL_GRID_COLS": PHYSICAL_GRID_COLS,
            "THRESHOLD_C": THRESHOLD_C
        },
        "time_statistics": {
            "total_time_s": float(total_time),
            "total_fm_train_time_s": float(total_fm_time),
            "total_sa_time_s": float(total_sa_time),
            "total_fdtd_sim_time_s": float(total_fdtd_time),
            "avg_fm_train_time_s": float(avg_fm_time),
            "avg_sa_time_s": float(avg_sa_time),
            "avg_fdtd_sim_time_s": float(avg_fdtd_time)
        },
        "final_results": {
            "best_fom": float(-best_fom),
            "best_config": best_config.tolist(),
            "final_analysis_transmission": final_metrics.get("Transmission", 0.0),
            "final_analysis_reflection": final_metrics.get("Reflection", 0.0)
        },
        "history": {
            "fom_trajectory": history['best_fom_history'],
            "all_foms": foms.tolist()
        }
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Done. Results in {RESULTS_DIR}")
    print(f"\nElapsed run time = {time.time() - start_total:.4f} s")

if __name__ == "__main__":
    if size > 1 and rank != 0:
        worker_node()
    else:
        master_node()