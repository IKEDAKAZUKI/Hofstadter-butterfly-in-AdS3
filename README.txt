BTZ-derived curved Harper package

Contents
- btz_curved_hofstadter.py: numerical analysis script for the BTZ-derived effective lattice model.
- run_btz_curved_hofstadter.ipynb: notebook interface for reproducing the analysis.
- btz_curved_hofstadter_metrics.csv: table of scan diagnostics.
- btz_curved_hofstadter_summary.json: summary of derived diagnostics.
- btz_curved_hofstadter_config_used.json: configuration used for the generated figures.
- btz_state_colored_spectra.png
- btz_ldos_maps.png
- btz_flux_radius_scatter.png
- btz_ab_response.png
- btz_annulus_heatmaps.png
- btz_full_heatmaps.png
- btz_robustness.png

Python requirements
- numpy
- scipy
- pandas
- matplotlib

Command-line usage
Run
    python btz_curved_hofstadter.py
from the project directory to regenerate the numerical outputs in the current working directory.

Notebook usage
Open run_btz_curved_hofstadter.ipynb and edit the CONFIG cell to change lattice sizes, parameter grids, or plotting options.
