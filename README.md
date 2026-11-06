# Bend Waveguide Tutorial

A step-by-step tutorial for modeling and simulating silicon photonic waveguide bends with the **Meep FDTD solver**.

This repository is intended as a practical learning resource: start from the electromagnetic model, build a 90° waveguide bend, excite the input mode, inspect the fields, and evaluate transmission, reflection, and bend-related loss.

## What you will learn

- How to define a waveguide bend and its surrounding simulation domain.
- How to choose materials, boundary conditions, PML layers, and spatial resolution.
- How to launch a guided mode (for example, the fundamental TE-like mode).
- How to inspect the electromagnetic field distribution.
- How to calculate transmission and reflection using a straight-waveguide reference simulation.
- How to check numerical convergence before comparing designs.

## Design baseline

The following values are a starting point for the tutorial, not a claim that a particular bend design has already been optimized or experimentally validated.

| Parameter | Starting value |
| --- | ---: |
| Operating wavelength | 1.55 µm |
| Bend angle | 90° |
| Silicon refractive index | 3.48 |
| Silicon dioxide refractive index | 1.44 |
| Waveguide width | 1.0 µm |
| Silicon core thickness | 0.5 µm |
| FDTD resolution | 40 pixels/µm |

The refractive indices are treated as constant in this introductory setup. For quantitative device design, use wavelength-dependent material data and verify the geometry, polarization, mesh, and boundary conditions against the intended fabrication stack.

> **Modeling note:** Meep's introductory 90° bend example is a 2D high-index waveguide model. A 2D model with an infinitely extruded core does not represent the finite height of an SOI strip waveguide. Use a justified effective-index approximation for a reduced model, or a 3D simulation when the vertical silicon/SiO₂ structure and polarization behavior matter.

## Getting started

Read the [Tutorial](docs/TUTORIAL.md) for the modeling workflow, setup instructions, and simulation checks.

Meep is available through Conda and, on supported platforms, PyPI. Native Windows installation is not supported by the official Meep documentation; on Windows, use WSL2 with Ubuntu or another supported Linux environment.

### Conda

```bash
conda create -n bend-tutorial -c conda-forge pymeep
conda activate bend-tutorial
python -c "import meep as mp; print(mp.__version__)"
```

### Linux / supported macOS Python environments

```bash
python -m pip install pymeep
python -c "import meep as mp; print(mp.__version__)"
```

The import name is `meep`, although the PyPI package is named `pymeep`. Check the official [Meep installation guide](https://meep.readthedocs.io/en/latest/Installation/) for platform and Python-version requirements.

## Repository contents

```text
Bend_Waveguide_Tutorial/
├── README.md
├── LICENSE
└── docs/
    └── TUTORIAL.md
```

This initial version contains the documentation workflow. Runnable project-specific simulation scripts and generated results will be added after their geometry, source configuration, and output format have been finalized and tested.

## References

- [Meep documentation — Python Tutorial: Basics](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/)
- [Meep documentation — Eigenmode Source](https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/)
- [Meep documentation — Installation](https://meep.readthedocs.io/en/latest/Installation/)

## License

This project is released under the [MIT License](LICENSE).
