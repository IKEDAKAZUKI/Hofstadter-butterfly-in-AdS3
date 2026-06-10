<div align="center">

# Hofstadter’s Butterfly in AdS<sub>3</sub> Black Holes

### Numerical data and reproducibility code for the BTZ-derived curved Harper model

[![arXiv](https://img.shields.io/badge/arXiv-2604.14335-b31b1b.svg)](https://arxiv.org/abs/2604.14335)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2FJHEP06%282026%29038-blue)](https://doi.org/10.1007/JHEP06%282026%29038)
[![JHEP](https://img.shields.io/badge/JHEP-2026%2C%20038-red)](https://link.springer.com/article/10.1007/JHEP06%282026%29038)
[![Python](https://img.shields.io/badge/Python-Research%20Code-3776AB)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br>

**BTZ black holes · Curved Harper equation · Hofstadter butterfly · AdS/CMT · Hyperbolic band theory · Aharonov–Bohm response**

<br>

<img src="btz_state_colored_spectra.png" width="82%" alt="State-colored spectra for the BTZ-derived curved Harper model">

</div>

---

## Overview

This repository contains the numerical code and released data for the work:

> K. Ikeda and Y. Oz,  
> **“Hofstadter’s Butterfly in AdS<sub>3</sub> Black Holes,”**  
> *Journal of High Energy Physics* **2026**, 038 (2026).  
> DOI: [10.1007/JHEP06(2026)038](https://doi.org/10.1007/JHEP06%282026%29038)  
> arXiv: [2604.14335](https://arxiv.org/abs/2604.14335)

The project studies a **BTZ-derived curved Harper model** on the constant-time cylinder of a non-rotating AdS<sub>3</sub> black hole. The code generates spectra, local-density-of-states maps, magnetic-flux diagnostics, Aharonov–Bohm response, parameter scans, and robustness checks for the effective single-band lattice model introduced in the paper.

The central physics question is how black-hole geometry modifies the Hofstadter spectrum. In this model:

- the AdS radius `L` controls the local curvature scale,
- the horizon radius `r_h` controls the throat size and near-horizon redshift,
- the magnetic flux controls the Harper-like spectral fragmentation,
- the Aharonov–Bohm flux probes spectral flow around the BTZ angular cycle.

---

## Scientific background

The ordinary Hofstadter butterfly arises from Bloch electrons on a lattice in a magnetic field. This repository studies a curved, gravitationally motivated analogue obtained from a reduced Dirac problem on a non-rotating BTZ black-hole background.

The effective lattice Hamiltonian retains:

- BTZ redshifted inverse proper bond lengths,
- magnetic Peierls phases,
- a curved Harper equation after angular Fourier transformation,
- a dimensionless angular quasi-momentum,
- state-resolved radial diagnostics.

The resulting spectra reveal how curvature and horizons deform the butterfly-like band structure.

---

## Key results reproduced by this repository

The scripts and released data reproduce the main numerical diagnostics of the paper, including:

- **BTZ-curved Hofstadter spectra** as a function of magnetic flux.
- **State-colored spectra** showing radial localization of eigenstates.
- **Local density of states** near and away from the horizon.
- **Flux-response versus radius** correlations.
- **Aharonov–Bohm spectral flow** on the BTZ cycle.
- **Persistent-current diagnostics** from flux insertion.
- **Annulus and full-exterior parameter scans** over `L` and `r_h`.
- **Robustness checks** against lattice size, flux resolution, and radial cutoff.

---

## Results preview

<p align="center">
  <img src="btz_ldos_maps.png" width="48%" alt="Local density of states maps">
  <img src="btz_ab_response.png" width="48%" alt="Aharonov-Bohm response">
</p>

<p align="center">
  <img src="btz_full_heatmaps.png" width="48%" alt="Full exterior heatmaps">
  <img src="btz_robustness.png" width="48%" alt="Robustness diagnostics">
</p>

---

## Repository contents

| File | Description |
|---|---|
| [`btz_curved_hofstadter.py`](btz_curved_hofstadter.py) | Main numerical analysis script for the BTZ-derived effective lattice model |
| [`run_btz_curved_hofstadter.ipynb`](run_btz_curved_hofstadter.ipynb) | Jupyter notebook interface for reproducing and modifying the analysis |
| [`btz_curved_hofstadter_metrics.csv`](btz_curved_hofstadter_metrics.csv) | Table of parameter-scan diagnostics |
| [`btz_curved_hofstadter_summary.json`](btz_curved_hofstadter_summary.json) | Summary of derived diagnostics used in the analysis |
| [`btz_curved_hofstadter_config_used.json`](btz_curved_hofstadter_config_used.json) | Configuration snapshot used to generate the released figures |
| [`btz_state_colored_spectra.png`](btz_state_colored_spectra.png) | State-colored Hofstadter spectra |
| [`btz_ldos_maps.png`](btz_ldos_maps.png) | Local-density-of-states maps |
| [`btz_flux_radius_scatter.png`](btz_flux_radius_scatter.png) | Flux response versus radial localization |
| [`btz_ab_response.png`](btz_ab_response.png) | Aharonov–Bohm spectral flow and persistent-current response |
| [`btz_annulus_heatmaps.png`](btz_annulus_heatmaps.png) | Annulus parameter-scan heatmaps |
| [`btz_full_heatmaps.png`](btz_full_heatmaps.png) | Full-exterior parameter-scan heatmaps |
| [`btz_robustness.png`](btz_robustness.png) | Robustness checks |
| [`LICENSE`](LICENSE) | MIT License |

---

## Quick start

Clone the repository:

```bash
git clone https://github.com/IKEDAKAZUKI/Hofstadter-butterfly-in-AdS3.git
cd Hofstadter-butterfly-in-AdS3
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the required Python packages:

```bash
python3 -m pip install numpy scipy pandas matplotlib jupyter
```

Run the full numerical analysis:

```bash
python3 btz_curved_hofstadter.py
```

This regenerates the numerical outputs in the current working directory.

---

## Notebook workflow

For an interactive workflow, open:

```bash
jupyter notebook run_btz_curved_hofstadter.ipynb
```

The notebook provides a convenient interface for:

- modifying the lattice size,
- changing the parameter grid,
- adjusting the magnetic-flux resolution,
- changing plotting options,
- regenerating selected figures,
- exploring additional BTZ parameter regimes.

---

## Default numerical configuration

The released data were generated with the configuration stored in:

```text
btz_curved_hofstadter_config_used.json
```

Main default parameters include:

| Parameter | Meaning | Default |
|---|---|---|
| `Nx` | Number of radial lattice sites | `48` |
| `Nphi` | Number of angular lattice sites / angular modes | `48` |
| `ax` | Radial lattice spacing in equal-area coordinate | `0.10` |
| `alpha_min` | Minimum magnetic flux parameter | `0.0` |
| `alpha_max` | Maximum magnetic flux parameter | `1.0` |
| `Nalpha` | Number of magnetic-flux samples | `141` |
| `L_grid` | AdS-radius scan values | `8, 10, 12, 14, 16, 18, 20` |
| `rh_grid` | Horizon-radius scan values | `0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0` |
| `x0_annulus` | Inner cutoff for annulus scans | `2.0` |
| `x0_full` | Inner cutoff for full-exterior scans | `0.0` |

---

## Diagnostics

The main script computes several diagnostics designed to quantify how the BTZ geometry modifies the curved Hofstadter spectrum.

| Diagnostic | Description |
|---|---|
| `D_box` | Box-counting estimate of spectral fragmentation |
| `flux_sensitivity` | Mean spectral sensitivity to magnetic flux |
| `rho_zero` | Density of states near zero energy |
| `avg_horizon_weight_mid` | Average near-horizon weight of mid-spectrum states |
| `avg_xbar_mid` | Mean radial position of mid-spectrum states |
| `charge_stiffness` | Aharonov–Bohm charge stiffness / persistent-current proxy |
| `half_bandwidth` | Half-bandwidth used for normalization |

These quantities are written to:

```text
btz_curved_hofstadter_metrics.csv
btz_curved_hofstadter_summary.json
```

---

## Physical interpretation

The code is designed to make the geometric effects visible in the spectrum.

In the BTZ-derived curved Harper model:

- weaker curvature, corresponding to larger `L`, sharpens butterfly-like spectral fragmentation;
- larger horizon radius `r_h` enhances near-horizon localization;
- near-horizon states become weakly dispersing;
- magnetic-flux and Aharonov–Bohm responses are suppressed when low-energy states are strongly localized near the horizon;
- the annulus and full-exterior geometries provide complementary probes of curvature and horizon effects.

---

## Repository structure

```text
.
├── btz_curved_hofstadter.py
├── run_btz_curved_hofstadter.ipynb
├── btz_curved_hofstadter_metrics.csv
├── btz_curved_hofstadter_summary.json
├── btz_curved_hofstadter_config_used.json
├── btz_state_colored_spectra.png
├── btz_ldos_maps.png
├── btz_flux_radius_scatter.png
├── btz_ab_response.png
├── btz_annulus_heatmaps.png
├── btz_full_heatmaps.png
├── btz_robustness.png
├── LICENSE
└── README.md
```

---

## Citation

If you use this code or data in academic work, please cite:

```bibtex
@article{Ikeda:2020guk,
    author = "Ikeda, Kazuki and Oz, Yaron",
    title = "{Hofstadter{\textquoteright}s butterfly in AdS$_{3}$ black holes}",
    eprint = "2604.14335",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1007/JHEP06(2026)038",
    journal = "JHEP",
    volume = "2026",
    pages = "038",
    year = "2026"
}
```

Official publication:

> K. Ikeda and Y. Oz,  
> **“Hofstadter’s butterfly in AdS<sub>3</sub> black holes,”**  
> *Journal of High Energy Physics* **2026**, 038 (2026).  
> DOI: [10.1007/JHEP06(2026)038](https://doi.org/10.1007/JHEP06%282026%29038)

---

## License

This repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 Kazuki Ikeda.

---

<div align="center">

**Hofstadter Butterfly · BTZ Geometry · AdS<sub>3</sub> Black Holes · Curved Harper Model**

</div>
