# Tutorial: Modeling a 90° Silicon Photonic Waveguide Bend with Meep

This tutorial describes a reproducible workflow for studying a 90° waveguide bend with Meep, an open-source finite-difference time-domain (FDTD) solver.

It is written as a modeling guide. The repository does not yet include a validated, project-specific simulation script, so the numerical values below are baseline inputs rather than guaranteed simulation results. Do not interpret them as an optimized design or measured device performance.

## 1. Learning objectives

By the end of this tutorial, you should be able to:

1. Define the physical structure and its material model.
2. Choose a simulation dimensionality that matches the question being studied.
3. Set up a mode source, absorbing boundaries, and field monitors.
4. Run a reference simulation and a bend simulation consistently.
5. Calculate normalized transmission and reflection.
6. Check whether the result is converged with respect to resolution and simulation time.

## 2. Physical structure and coordinate system

A 90° bend redirects a guided optical mode from an input waveguide to an output waveguide whose propagation direction is perpendicular to the input.

For a top-view model, place the bend in the (x-y) plane:

- The input waveguide propagates along (+x).
- The output waveguide propagates along (+y).
- The bend connects the two straight sections.
- The waveguide width is measured in the plane of the model.

Define the bend using a centerline radius (R) and waveguide width (w). For a circular quarter bend, the centerline follows a 90° arc. The inner and outer radii are approximately (R-w/2) and (R+w/2), respectively.

Keep the input and output straight sections long enough to place the source and flux monitors away from the curved region. This reduces contamination from evanescent fields and makes the reference comparison easier.

### Important: 2D versus 3D

A 2D simulation is useful for learning the FDTD workflow and comparing simplified in-plane geometries. However, a 2D high-index core is effectively invariant in the out-of-plane direction. It is not automatically equivalent to a finite-height silicon strip waveguide.

For an SOI strip waveguide, the vertical stack (silicon core, buried oxide, and upper cladding) affects the effective index, mode profile, polarization, and bend loss.

Choose one of these approaches and state it clearly in any report:

- **2D effective-index model:** replace the vertical stack with a justified effective-index approximation. This can be useful for fast in-plane studies, but the approximation must be documented.
- **3D full-vector model:** explicitly model the silicon thickness and surrounding materials when vertical confinement and realistic TE/TM behavior are important.

Do not compare a 2D effective-index result directly with a 3D device result without explaining the model differences.

## 3. Baseline parameters

The following values are a starting point for a tutorial case. Update them to match the actual device and simulation script.

| Parameter | Baseline | Notes |
| --- | ---: | --- |
| Free-space wavelength | 1.55 µm | Common telecom-band reference |
| Bend angle | 90° | Input and output are perpendicular |
| Waveguide width | 1.0 µm | In-plane width |
| Silicon thickness | 0.5 µm | Relevant to a 3D SOI model |
| Silicon refractive index | 3.48 | Constant-index approximation |
| Silicon dioxide refractive index | 1.44 | Constant-index approximation |
| Spatial resolution | 40 pixels/µm | Initial mesh; convergence must be checked |
| PML thickness | 1.0 µm | Initial value; verify by increasing it |

The corresponding frequency in Meep's length-unit convention (µm) is:

[
f = \frac{1}{\lambda_0}
  = \frac{1}{1.55\;\mu\mathrm{m}}
  \approx 0.64516\;\mu\mathrm{m}^{-1}.
]

If material dispersion is important, use a dispersive material model instead of a constant refractive index.

## 4. Install and verify Meep

Meep's Python interface is imported as `meep`. The package name used by pip is `pymeep`.

### Conda (recommended)

```bash
conda create -n bend-tutorial -c conda-forge pymeep
conda activate bend-tutorial
python -c "import meep as mp; print(mp.__version__)"
```

### pip (supported environments only)

```bash
python -m pip install pymeep
python -c "import meep as mp; print(mp.__version__)"
```

Check the official installation guide for supported operating systems and dependencies: <https://meep.readthedocs.io/en/latest/Installation/>.

On Windows, use WSL2 with a supported Linux distribution rather than assuming that native Windows Python installation is supported.

## 5. Build the simulation model

A Meep simulation is assembled from several components:

1. **Cell:** the finite computational domain.
2. **Geometry:** the waveguide core and surrounding media.
3. **Materials:** permittivity or refractive-index definitions.
4. **Boundary layers:** PML regions that absorb outgoing radiation.
5. **Source:** a time-dependent excitation, preferably a guided eigenmode for a waveguide input.
6. **Monitors:** flux planes and field sampling regions.
7. **Run condition:** a simulation-time or field-decay stopping criterion.

### 5.1 Computational cell

Make the cell large enough to contain:

- the complete bend,
- straight input and output waveguide sections,
- source and monitor locations,
- sufficient distance between the structure and PML.

Place PML at the outer boundaries. The PML should not overlap the waveguide or the source/monitor planes. Increase the PML thickness and the distance to the structure if results change noticeably.

### 5.2 Geometry

Parameterize the geometry rather than hard-coding unrelated coordinates. At minimum, expose:

- bend radius (R),
- waveguide width (w),
- input straight length,
- output straight length,
- cladding and substrate dimensions (for a 3D model).

For a circular 90° bend, construct a quarter-annular core joined smoothly to the straight input and output sections. Ensure that the geometry has no unintended gaps or overlaps at the interfaces.

A sharp, square L-shaped corner is not the same structure as a circular bend. If using a square corner as a first debugging model, label it accordingly and do not report its result as the loss of a circular bend.

### 5.3 Material definitions

For a simple nondispersive model, refractive index (n) corresponds to relative permittivity:

[
\varepsilon_r = n^2.
]

Using the baseline values:

- Silicon: (\varepsilon_r \approx 3.48^2).
- Silicon dioxide: (\varepsilon_r \approx 1.44^2).

The actual material assignment depends on the selected dimensionality. A 3D SOI model should explicitly define the silicon core, buried oxide, and upper cladding. A 2D model must explain how the vertical stack is represented.

## 6. Excite the guided mode

A waveguide bend should be excited with a guided input mode, rather than an arbitrary point source, when the goal is to evaluate guided-mode transmission.

Meep provides `EigenModeSource` for launching a mode calculated from a cross-section of the waveguide. Configure the source with:

- a source plane perpendicular to the input propagation direction,
- the intended frequency or frequency range,
- the desired mode band,
- the correct propagation direction,
- a source region that covers the waveguide cross-section.

For a single-frequency baseline, use the frequency calculated from the target wavelength. For a broadband study, define a frequency range and ensure that the source and monitor settings cover the entire range.

The mode index or band number should be verified for the actual cross-section. Do not assume that a particular band number always corresponds to TE0 across different geometries or dimensionalities.

See the official Meep eigenmode-source tutorial: <https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/>.

## 7. Add flux monitors

Use flux monitors to measure power in the input and output waveguides.

Recommended monitor locations:

- **Input monitor:** on the straight input section, between the source and the bend.
- **Output monitor:** on the straight output section, after the bend.

Keep both planes far enough from the bend that the guided mode is well formed. Ensure that each monitor spans the full transverse region needed to capture the guided mode and relevant radiated power.

A useful procedure is to run two simulations:

### Simulation A — reference waveguide

Replace the bend with a straight waveguide while keeping the input/output cross-section, source, cell, resolution, monitor normalization, and frequency settings consistent.

Record the incident power spectrum at the input reference plane and the transmitted power spectrum at the output reference plane.

### Simulation B — bent waveguide

Use the same source and numerical settings, but replace the straight reference section with the bend geometry.

Normalize the bent-waveguide output against the straight-waveguide reference. This helps separate bend-related effects from source normalization and ordinary propagation through the straight sections.

The exact normalization procedure depends on monitor placement and propagation direction. Document whether the reported quantity is total flux, forward guided-mode power, or a ratio relative to the straight reference.

## 8. Calculate transmission and bend loss

A common normalized transmission definition is:

[
T_{\mathrm{norm}}(f)
= \frac{P_{\mathrm{out,bend}}(f)}
       {P_{\mathrm{out,straight}}(f)}.
]

Here, both output powers must be obtained using consistent source, monitor, and simulation settings. If the reference and bend have different straight-section lengths, account for the additional propagation loss.

The corresponding bend insertion loss is:

[
IL_{\mathrm{bend}}(f)
= -10\log_{10}\left(T_{\mathrm{norm}}(f)\right)
\quad\mathrm{dB}.
]

For example, a normalized transmission of 0.90 corresponds to approximately 0.46 dB insertion loss. This is only a mathematical illustration, not a result from this repository.

Reflection can be estimated using an input-side flux monitor with a consistent incident/reflected-wave separation method. A raw net flux value at the input plane is not automatically the reflected power. Use Meep's flux-data subtraction workflow or a suitable modal decomposition, and explain the sign and normalization convention.

If the output monitor includes radiation as well as the guided mode, total output flux may overestimate useful guided transmission. For mode-to-mode transmission, use an eigenmode decomposition or another explicitly defined modal-power measurement.

## 9. Visualize the electromagnetic fields

Plot field components such as (E_x), (E_y), (E_z), or the magnetic field component appropriate to the model. For a 2D model, the available field components depend on the chosen polarization.

Useful plots include:

- field amplitude or a signed field component around the bend,
- a snapshot after the source has entered the structure,
- a steady-state or frequency-domain field distribution,
- output mode profile compared with the input mode profile.

A field snapshot is useful for diagnosing geometry and propagation, but it is not by itself a quantitative measurement of transmission. Use flux or modal monitors for power calculations.

## 10. Numerical convergence and validation

A simulation result is not reliable merely because the fields look smooth. Check convergence systematically.

### 10.1 Spatial resolution

Repeat the simulation with progressively finer spatial resolution. For example, compare 30, 40, and 50 pixels/µm as an initial study, then refine further if the result has not stabilized.

Track at least:

- normalized transmission,
- reflection,
- bend insertion loss,
- runtime and memory use.

A converged result should change by less than a stated tolerance when the resolution is increased. Choose and report the tolerance based on the accuracy required by the study.

### 10.2 PML and cell-size checks

Increase the PML thickness and/or the distance between the waveguide and PML. The measured transmission should not change materially. If it does, the simulation may be affected by boundary reflections or insufficient separation.

### 10.3 Simulation time

Run long enough for the source pulse to pass through the device and for relevant fields to decay. Check the field-decay criterion and repeat with a longer run time to verify that the flux result is stable.

### 10.4 Reference consistency

Use identical numerical settings for the straight reference and bend simulations. Confirm that the reference waveguide itself has negligible reflection and that the monitor planes are positioned consistently.

### 10.5 Geometry checks

Before a parameter sweep, inspect the material distribution and confirm:

- the bend is connected to both straight sections,
- the core width is consistent,
- the intended bend radius is measured from the centerline,
- no geometry is unintentionally clipped by the cell or PML.

## 11. Parameter sweep

After the baseline case is validated, study the effect of bend radius.

A simple sweep can vary (R) while keeping the following fixed:

- waveguide cross-section,
- material model,
- wavelength,
- source mode,
- cell and monitor definitions,
- spatial resolution,
- convergence criteria.

For every radius, run both the bend and its straight reference under consistent settings. Plot normalized transmission or insertion loss against bend radius.

Do not compare different radii using a mesh that is too coarse to represent the smallest radius accurately. Recheck convergence at the most demanding geometry.

Other useful parameters include waveguide width, wavelength, polarization, and material index. Change one parameter at a time for an interpretable baseline study.

## 12. Recommended result presentation

A clear report or repository update should include:

1. A diagram of the geometry with (R), (w), and monitor positions.
2. The dimensionality and material stack.
3. Source wavelength, mode, and polarization convention.
4. Cell size, resolution, and PML thickness.
5. Field-distribution plots.
6. Straight-reference and bend transmission.
7. The definition used for insertion loss and reflection.
8. Resolution, PML, and simulation-time convergence checks.

Always include units and state whether a value is simulated, calculated, or measured.

## 13. Troubleshooting

| Symptom | Possible cause | Checks |
| --- | --- | --- |
| Strong field reflection at the input | Mode mismatch, abrupt geometry, or boundary reflection | Inspect the input cross-section, source mode, and PML |
| Output power changes with cell size | PML too close or cell too small | Increase cell dimensions and repeat |
| Transmission changes with resolution | Mesh is insufficient | Increase resolution and compare |
| Field does not decay | Run time too short, trapped resonance, or poor absorbing boundary | Inspect decay and extend simulation |
| Output field is not guided | Bend radiation or incorrect output alignment | Inspect output mode profile and use modal decomposition |
| 2D and 3D results differ | Different confinement and polarization physics | Document the model assumptions; compare equivalent modes |

## 14. Further reading

- [Meep documentation](https://meep.readthedocs.io/en/latest/)
- [Meep Python Tutorial: Basics](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/)
- [Meep Eigenmode Source Tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/)
- [Meep installation guide](https://meep.readthedocs.io/en/latest/Installation/)

---

**Status:** This document defines the intended learning workflow. Add links to runnable scripts, exact environment versions, and verified simulation figures as those project artifacts become available.
