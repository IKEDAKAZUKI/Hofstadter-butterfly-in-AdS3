"""Numerical analysis for the BTZ-derived curved Harper model.

This script generates spectra, local-density-of-states maps, flux-response diagnostics,
Aharonov-Bohm response, parameter scans, and robustness checks for the effective
single-band lattice model introduced in the manuscript.

The numerical model is a BTZ-derived effective single-band lattice Hamiltonian on the
constant-time BTZ cylinder. It retains the BTZ redshifted inverse proper bond lengths
and magnetic Peierls phases within a numerically tractable one-band description.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


DEFAULT_CONFIG: Dict[str, Any] = {
    "common": {
        "Nx": 48,
        "Nphi": 48,
        "ax": 0.10,
        "alpha_min": 0.0,
        "alpha_max": 1.0,
        "Nalpha": 141,
        "nu": 0.0,
        "q": 1.0,
    },
    "sweep": {
        "L_grid": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        "rh_grid": [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
        "x0_annulus": 2.0,
        "x0_full": 0.0,
    },
    "diagnostics": {
        "n_mid": 4,
        "zero_dos_window_fraction": 0.05,
        "box_counting_grids": [32, 48, 64, 96, 128],
        "horizon_cutoff_x": 1.0,
        "ldos_energy_bins": 140,
        "ldos_sigma_fraction": 0.03,
        "alpha_ref_flux_response": 0.50,
        "flux_response_delta_alpha": 0.01,
        "flux_response_energy_window_fraction": 0.40,
        "ab_alpha_ref": 0.50,
        "ab_Nphi_scan": 81,
        "ab_n_levels_show": 14,
        "stiffness_delta_phi": 0.16,
        "robust_case": {"L": 12.0, "r_h": 2.0, "x0": 0.0},
        "robust_Nx_grid": [24, 32, 40, 48, 56],
        "robust_Nalpha_grid": [41, 61, 81, 101, 141],
        "robust_xmax_fixed": 4.8,
        "robust_x0_grid": [0.0, 1.0, 2.0],
    },
    "plots": {
        "use_tex": "auto",
        "tex_preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "mathtext_fontset": "cm",
        "figure_dpi": 240,
        "spectra_marker_size": 0.08,
        "scatter_alpha": 1.0,
        "scatter_marker_size": 12,
        "ab_level_cmap": "coolwarm",
        "annotate_heatmaps": False,
        "state_cases": [
            {"title": "weak curvature annulus", "L": 20.0, "r_h": 1.0, "x0": 2.0},
            {"title": "strong curvature annulus", "L": 8.0, "r_h": 1.0, "x0": 2.0},
            {"title": "small horizon full exterior", "L": 12.0, "r_h": 1.0, "x0": 0.0},
            {"title": "large horizon full exterior", "L": 12.0, "r_h": 4.0, "x0": 0.0},
        ],
        "ldos_cases": [
            {"title": "small horizon", "L": 12.0, "r_h": 1.0, "x0": 0.0},
            {"title": "large horizon", "L": 12.0, "r_h": 4.0, "x0": 0.0},
        ],
        "flux_scatter_cases": [
            {"title": "small horizon", "L": 12.0, "r_h": 1.0, "x0": 0.0},
            {"title": "large horizon", "L": 12.0, "r_h": 4.0, "x0": 0.0},
        ],
        "ab_cases": [
            {"title": "small horizon", "L": 12.0, "r_h": 1.0, "x0": 0.0},
            {"title": "large horizon", "L": 12.0, "r_h": 4.0, "x0": 0.0},
        ],
    },
    "output_names": {
        "metrics_csv": "btz_curved_hofstadter_metrics.csv",
        "summary_json": "btz_curved_hofstadter_summary.json",
        "config_json": "btz_curved_hofstadter_config_used.json",
        "state_spectra_png": "btz_state_colored_spectra.png",
        "ldos_png": "btz_ldos_maps.png",
        "flux_scatter_png": "btz_flux_radius_scatter.png",
        "ab_response_png": "btz_ab_response.png",
        "annulus_heatmaps_png": "btz_annulus_heatmaps.png",
        "full_heatmaps_png": "btz_full_heatmaps.png",
        "robustness_png": "btz_robustness.png",
    },
}


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def get_default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def resolve_config(config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = get_default_config()
    if config is not None:
        deep_update(cfg, config)
    return cfg


def alpha_grid_from_config(config: Mapping[str, Any]) -> np.ndarray:
    common = config["common"]
    return np.linspace(
        float(common["alpha_min"]),
        float(common["alpha_max"]),
        int(common["Nalpha"]),
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_config_snapshot(config: Mapping[str, Any], output_dir: Path) -> Path:
    path = output_dir / config["output_names"]["config_json"]
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config), f, indent=2)
    return path


def _numeric_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "common": _to_jsonable(config.get("common", {})),
        "sweep": _to_jsonable(config.get("sweep", {})),
        "diagnostics": _to_jsonable(config.get("diagnostics", {})),
    }


def load_cached_metrics_if_compatible(output_dir: Path, config: Mapping[str, Any]) -> pd.DataFrame | None:
    metrics_path = output_dir / config["output_names"]["metrics_csv"]
    config_path = output_dir / config["output_names"]["config_json"]
    if not metrics_path.exists() or not config_path.exists():
        return None
    try:
        cached_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if _numeric_signature(cached_cfg) != _numeric_signature(config):
        return None
    return pd.read_csv(metrics_path)


# -----------------------------------------------------------------------------
# Matplotlib helpers
# -----------------------------------------------------------------------------
def _tex_dependency_status() -> Tuple[bool, List[str]]:
    if os.environ.get("BTZ_DISABLE_LATEX", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False, ["disabled by BTZ_DISABLE_LATEX"]
    required = ["latex", "dvipng"]
    missing = [exe for exe in required if shutil.which(exe) is None]
    return len(missing) == 0, missing


def configure_matplotlib(config: MutableMapping[str, Any]) -> Dict[str, Any]:
    plots = config.get("plots", {})
    request = plots.get("use_tex", "auto")
    tex_available, missing = _tex_dependency_status()

    if isinstance(request, str):
        norm = request.strip().lower()
        requested_tex = norm not in {"false", "off", "0", "never", "mathtext"}
    else:
        requested_tex = bool(request)

    effective_use_tex = requested_tex and tex_available
    plt.rcParams.update(
        {
            "text.usetex": effective_use_tex,
            "font.family": "serif",
            "mathtext.fontset": plots.get("mathtext_fontset", "cm"),
            "axes.unicode_minus": False,
        }
    )
    if effective_use_tex:
        plt.rcParams["text.latex.preamble"] = plots.get("tex_preamble", r"\usepackage{amsmath}\usepackage{amssymb}")
    else:
        plt.rcParams["text.latex.preamble"] = ""
        if requested_tex and missing:
            warnings.warn(
                "External TeX rendering requested but unavailable ("
                + ", ".join(missing)
                + "). Falling back to matplotlib mathtext.",
                RuntimeWarning,
                stacklevel=2,
            )

    runtime = config.setdefault("_runtime", {})
    runtime["text_backend"] = "usetex" if effective_use_tex else "mathtext"
    runtime["requested_use_tex"] = request
    runtime["missing_tex_dependencies"] = missing
    return runtime


def _fmt_param(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:g}"


def _case_title(case: Mapping[str, Any]) -> str:
    raw = str(case.get("title", case.get("label", ""))).strip()
    lead = raw.splitlines()[0] if raw else ""
    return (
        lead
        + "\n"
        + rf"$L={_fmt_param(case['L'])},\; r_h={_fmt_param(case['r_h'])},\; x_0={_fmt_param(case['x0'])}$"
    )


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
def x_grid(Nx: int, ax: float, x0: float) -> np.ndarray:
    return x0 + (np.arange(Nx) + 0.5) * ax


def kappa_grid(Nphi: int, nu: float = 0.0, phi_ab: float = 0.0, q: float = 1.0) -> np.ndarray:
    """Dimensionless angular quasi-momenta.

    We use
        c_{n,j} = N_phi^{-1/2} sum_s exp(i kappa_s j) c_n(kappa_s),
        kappa_s = [2*pi*(s+nu) - q*Phi] / N_phi,
    so the Harper potential is 2*lambda_n*cos(kappa_s - theta_n).
    """
    return (2.0 * np.pi * (np.arange(Nphi) + nu) - q * phi_ab) / float(Nphi)


def tridiagonal_components(
    Nx: int,
    ax: float,
    Nphi: int,
    alpha: float,
    kappa: float,
    L: float,
    r_h: float,
    x0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if Nx < 2:
        raise ValueError("Nx must be at least 2.")

    x = x_grid(Nx, ax, x0)
    r = np.sqrt(x * x + r_h * r_h)
    lam = (Nphi / (2.0 * np.pi)) * x / (L * r)

    x_half = x0 + np.arange(1, Nx) * ax
    r_half = np.sqrt(x_half * x_half + r_h * r_h)
    offdiag = -x_half * r_half / (L * L * ax)

    phase_offset = 2.0 * np.pi * alpha * (x0 / ax)
    theta = 2.0 * np.pi * alpha * (np.arange(Nx) + 0.5)
    diag = 2.0 * lam * np.cos(kappa - phase_offset - theta)
    return diag, offdiag, x


def eigensystem_open(
    Nx: int,
    ax: float,
    Nphi: int,
    alpha: float,
    kappa: float,
    L: float,
    r_h: float,
    x0: float,
    horizon_cutoff_x: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag, offdiag, x = tridiagonal_components(Nx, ax, Nphi, alpha, kappa, L, r_h, x0)
    vals, vecs = eigh_tridiagonal(diag, offdiag)
    probs = np.abs(vecs) ** 2
    xbar = (x[:, None] * probs).sum(axis=0)
    xi2 = np.sqrt(((x[:, None] - xbar[None, :]) ** 2 * probs).sum(axis=0))
    ipr = (probs ** 2).sum(axis=0)
    wh = probs[x < horizon_cutoff_x].sum(axis=0)
    return vals, vecs, x, xbar, xi2, ipr, wh


def eigvals_open(
    Nx: int,
    ax: float,
    Nphi: int,
    alpha: float,
    kappa: float,
    L: float,
    r_h: float,
    x0: float,
) -> np.ndarray:
    diag, offdiag, _ = tridiagonal_components(Nx, ax, Nphi, alpha, kappa, L, r_h, x0)
    return eigh_tridiagonal(diag, offdiag, eigvals_only=True)


def scan_case_state_data(
    Nx: int,
    ax: float,
    Nphi: int,
    alphas: np.ndarray,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    horizon_cutoff_x: float,
) -> Dict[str, np.ndarray]:
    kappas = kappa_grid(Nphi, nu=nu, phi_ab=0.0, q=q)
    n_total = len(alphas) * len(kappas) * Nx
    A = np.empty(n_total, dtype=float)
    E = np.empty(n_total, dtype=float)
    XBAR = np.empty(n_total, dtype=float)
    WH = np.empty(n_total, dtype=float)

    cursor = 0
    for alpha in alphas:
        for kap in kappas:
            vals, _, x, xbar, _, _, wh = eigensystem_open(
                Nx=Nx,
                ax=ax,
                Nphi=Nphi,
                alpha=float(alpha),
                kappa=float(kap),
                L=L,
                r_h=r_h,
                x0=x0,
                horizon_cutoff_x=horizon_cutoff_x,
            )
            n = len(vals)
            A[cursor : cursor + n] = alpha
            E[cursor : cursor + n] = vals
            XBAR[cursor : cursor + n] = xbar
            WH[cursor : cursor + n] = wh
            cursor += n

    bw = max(abs(E.min()), abs(E.max()))
    x = x_grid(Nx, ax, x0)
    return {
        "alpha": A,
        "energy": E,
        "xbar": XBAR,
        "horizon_weight": WH,
        "bandwidth": np.array([bw]),
        "x": x,
    }


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def box_counting_dimension(alpha_pts: np.ndarray, energy_pts: np.ndarray, grids: Sequence[int]) -> float:
    bw = max(abs(np.min(energy_pts)), abs(np.max(energy_pts)))
    if bw <= 0:
        return float("nan")
    e_norm = energy_pts / bw

    counts = []
    scales = []
    for G in grids:
        ia = np.floor(alpha_pts * G).astype(int)
        ie = np.floor((e_norm + 1.0) * 0.5 * G).astype(int)
        mask = (ia >= 0) & (ia < G) & (ie >= 0) & (ie < G)
        occ = np.zeros((G, G), dtype=bool)
        occ[ie[mask], ia[mask]] = True
        counts.append(max(int(occ.sum()), 1))
        scales.append(G)
    coeff = np.polyfit(np.log(scales), np.log(counts), 1)
    return float(coeff[0])


def zero_energy_dos(energies: np.ndarray, window_fraction: float) -> float:
    bw = max(abs(energies.min()), abs(energies.max()))
    delta = max(window_fraction * bw, 1e-12)
    return float(np.mean(np.abs(energies) < 0.5 * delta) / delta)


def sorted_full_spectrum(
    Nx: int,
    ax: float,
    Nphi: int,
    alpha: float,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    phi_ab: float,
    q: float,
) -> np.ndarray:
    vals_list = []
    for kap in kappa_grid(Nphi, nu=nu, phi_ab=phi_ab, q=q):
        vals_list.append(eigvals_open(Nx, ax, Nphi, alpha, float(kap), L, r_h, x0))
    return np.sort(np.concatenate(vals_list))


def ground_state_energy_half_filling(vals_sorted: np.ndarray) -> float:
    n_occ = len(vals_sorted) // 2
    return float(np.sum(vals_sorted[:n_occ]))


def charge_stiffness_at_alpha(
    Nx: int,
    ax: float,
    Nphi: int,
    alpha_ref: float,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    delta_phi: float,
    bandwidth_norm: float,
) -> float:
    em = ground_state_energy_half_filling(
        sorted_full_spectrum(Nx, ax, Nphi, alpha_ref, L, r_h, x0, nu, -delta_phi, q)
    )
    e0 = ground_state_energy_half_filling(
        sorted_full_spectrum(Nx, ax, Nphi, alpha_ref, L, r_h, x0, nu, 0.0, q)
    )
    ep = ground_state_energy_half_filling(
        sorted_full_spectrum(Nx, ax, Nphi, alpha_ref, L, r_h, x0, nu, delta_phi, q)
    )
    return float(abs(em - 2.0 * e0 + ep) / (delta_phi * delta_phi * bandwidth_norm))


def compute_case_metrics(
    Nx: int,
    ax: float,
    Nphi: int,
    alphas: np.ndarray,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    n_mid: int,
    zero_dos_window_fraction: float,
    box_grids: Sequence[int],
    horizon_cutoff_x: float,
    alpha_ref_ab: float,
    delta_phi: float,
) -> Dict[str, float]:
    kappas = kappa_grid(Nphi, nu=nu, phi_ab=0.0, q=q)
    n_total = len(alphas) * len(kappas) * Nx
    A = np.empty(n_total, dtype=float)
    E = np.empty(n_total, dtype=float)

    xi_mid: List[float] = []
    ipr_mid: List[float] = []
    wh_mid: List[float] = []
    xbar_mid: List[float] = []
    sorted_spec_alpha: List[np.ndarray] = []

    cursor = 0
    for alpha in alphas:
        vals_alpha = np.empty(len(kappas) * Nx, dtype=float)
        vc = 0
        for kap in kappas:
            vals, _, _, xbar, xi2, ipr, wh = eigensystem_open(
                Nx=Nx,
                ax=ax,
                Nphi=Nphi,
                alpha=float(alpha),
                kappa=float(kap),
                L=L,
                r_h=r_h,
                x0=x0,
                horizon_cutoff_x=horizon_cutoff_x,
            )
            n = len(vals)
            A[cursor : cursor + n] = alpha
            E[cursor : cursor + n] = vals
            cursor += n
            vals_alpha[vc : vc + n] = vals
            vc += n

            idx = np.argsort(np.abs(vals))[: min(n_mid, len(vals))]
            xi_mid.extend(xi2[idx].tolist())
            ipr_mid.extend(ipr[idx].tolist())
            wh_mid.extend(wh[idx].tolist())
            xbar_mid.extend(xbar[idx].tolist())

        sorted_spec_alpha.append(np.sort(vals_alpha))

    sorted_spec_alpha = np.asarray(sorted_spec_alpha)
    bw = max(abs(E.min()), abs(E.max()))
    flux_sens = float(np.mean(np.std(sorted_spec_alpha, axis=0)) / bw)
    rho0 = zero_energy_dos(E, zero_dos_window_fraction)
    dbox = box_counting_dimension(A, E, box_grids)
    Dphi = charge_stiffness_at_alpha(
        Nx=Nx,
        ax=ax,
        Nphi=Nphi,
        alpha_ref=alpha_ref_ab,
        L=L,
        r_h=r_h,
        x0=x0,
        nu=nu,
        q=q,
        delta_phi=delta_phi,
        bandwidth_norm=bw,
    )

    return {
        "D_box": dbox,
        "flux_sensitivity": flux_sens,
        "avg_ipr_mid": float(np.mean(ipr_mid)),
        "avg_xi_mid": float(np.mean(xi_mid)),
        "rho_zero": rho0,
        "avg_horizon_weight_mid": float(np.mean(wh_mid)),
        "avg_xbar_mid": float(np.mean(xbar_mid)),
        "charge_stiffness": Dphi,
        "half_bandwidth": bw,
    }


def parameter_sweep(config: Mapping[str, Any]) -> pd.DataFrame:
    common = config["common"]
    sweep = config["sweep"]
    diag = config["diagnostics"]
    alphas = alpha_grid_from_config(config)

    rows: List[Dict[str, float | str]] = []
    scenarios = [("annulus", float(sweep["x0_annulus"])), ("full_exterior", float(sweep["x0_full"]))]
    for scenario, x0 in scenarios:
        for r_h in sweep["rh_grid"]:
            for L in sweep["L_grid"]:
                metrics = compute_case_metrics(
                    Nx=int(common["Nx"]),
                    ax=float(common["ax"]),
                    Nphi=int(common["Nphi"]),
                    alphas=alphas,
                    L=float(L),
                    r_h=float(r_h),
                    x0=x0,
                    nu=float(common["nu"]),
                    q=float(common["q"]),
                    n_mid=int(diag["n_mid"]),
                    zero_dos_window_fraction=float(diag["zero_dos_window_fraction"]),
                    box_grids=[int(v) for v in diag["box_counting_grids"]],
                    horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
                    alpha_ref_ab=float(diag["ab_alpha_ref"]),
                    delta_phi=float(diag["stiffness_delta_phi"]),
                )
                metrics.update(
                    {
                        "scenario": scenario,
                        "x0": x0,
                        "L": float(L),
                        "r_h": float(r_h),
                        "Nx": int(common["Nx"]),
                        "Nphi": int(common["Nphi"]),
                        "ax": float(common["ax"]),
                        "Nalpha": len(alphas),
                        "nu": float(common["nu"]),
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Additional state-resolved diagnostics
# -----------------------------------------------------------------------------
def local_dos_map(
    Nx: int,
    ax: float,
    Nphi: int,
    alphas: np.ndarray,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    horizon_cutoff_x: float,
    energy_bins: int,
    sigma_fraction: float,
) -> Dict[str, np.ndarray]:
    # First pass for bandwidth.
    scan = scan_case_state_data(
        Nx=Nx,
        ax=ax,
        Nphi=Nphi,
        alphas=alphas,
        L=L,
        r_h=r_h,
        x0=x0,
        nu=nu,
        q=q,
        horizon_cutoff_x=horizon_cutoff_x,
    )
    bw = float(scan["bandwidth"][0])
    x = scan["x"]
    e_grid = np.linspace(-1.0, 1.0, int(energy_bins))
    sigma = max(float(sigma_fraction), 1e-4)
    ldos = np.zeros((Nx, len(e_grid)), dtype=float)

    kappas = kappa_grid(Nphi, nu=nu, phi_ab=0.0, q=q)
    count = 0
    for alpha in alphas:
        for kap in kappas:
            vals, vecs, _, _, _, _, _ = eigensystem_open(
                Nx=Nx,
                ax=ax,
                Nphi=Nphi,
                alpha=float(alpha),
                kappa=float(kap),
                L=L,
                r_h=r_h,
                x0=x0,
                horizon_cutoff_x=horizon_cutoff_x,
            )
            e_norm = vals / bw
            probs = np.abs(vecs) ** 2
            gauss = np.exp(-0.5 * ((e_grid[None, :] - e_norm[:, None]) / sigma) ** 2) / (
                math.sqrt(2.0 * math.pi) * sigma
            )
            ldos += probs @ gauss
            count += 1

    ldos /= max(count, 1)
    return {"x": x, "energy_norm": e_grid, "ldos": ldos, "bandwidth": np.array([bw])}


def flux_response_vs_radius(
    Nx: int,
    ax: float,
    Nphi: int,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    alpha_ref: float,
    delta_alpha: float,
    horizon_cutoff_x: float,
    energy_window_fraction: float,
) -> Dict[str, np.ndarray]:
    kappas = kappa_grid(Nphi, nu=nu, phi_ab=0.0, q=q)

    # bandwidth at zero AB flux from alpha_ref scan only
    all_vals_ref = []
    XBAR, RESP, ENORM, WH = [], [], [], []
    for kap in kappas:
        vals0, _, _, xbar0, _, _, wh0 = eigensystem_open(
            Nx=Nx,
            ax=ax,
            Nphi=Nphi,
            alpha=alpha_ref,
            kappa=float(kap),
            L=L,
            r_h=r_h,
            x0=x0,
            horizon_cutoff_x=horizon_cutoff_x,
        )
        valsm = eigvals_open(Nx, ax, Nphi, alpha_ref - delta_alpha, float(kap), L, r_h, x0)
        valsp = eigvals_open(Nx, ax, Nphi, alpha_ref + delta_alpha, float(kap), L, r_h, x0)
        all_vals_ref.append(vals0)
        XBAR.append(xbar0)
        RESP.append(np.abs(valsp - valsm) / (2.0 * delta_alpha))
        ENORM.append(np.abs(vals0))
        WH.append(wh0)

    all_vals_ref = np.concatenate(all_vals_ref)
    bw = max(abs(all_vals_ref.min()), abs(all_vals_ref.max()))
    xbar = np.concatenate(XBAR)
    response = np.concatenate(RESP) / bw
    energy_norm = np.concatenate(ENORM) / bw
    horizon_weight = np.concatenate(WH)
    x = x_grid(Nx, ax, x0)
    x_max = float(x.max())
    xbar_norm = xbar / x_max

    mask = energy_norm <= float(energy_window_fraction)
    return {
        "xbar_norm": xbar_norm[mask],
        "response_norm": response[mask],
        "energy_norm": energy_norm[mask],
        "horizon_weight": horizon_weight[mask],
        "bandwidth": np.array([bw]),
    }


def ab_spectral_flow_and_current(
    Nx: int,
    ax: float,
    Nphi: int,
    L: float,
    r_h: float,
    x0: float,
    nu: float,
    q: float,
    alpha_ref: float,
    Nphi_scan: int,
) -> Dict[str, np.ndarray]:
    phi_grid = np.linspace(0.0, 2.0 * np.pi, int(Nphi_scan), endpoint=False)
    spectra = []
    egs = []
    for phi_ab in phi_grid:
        vals = sorted_full_spectrum(Nx, ax, Nphi, alpha_ref, L, r_h, x0, nu, float(phi_ab), q)
        spectra.append(vals)
        egs.append(ground_state_energy_half_filling(vals))
    spectra = np.asarray(spectra)
    egs = np.asarray(egs)
    bw = max(abs(spectra.min()), abs(spectra.max()))

    dphi = phi_grid[1] - phi_grid[0]
    current = -(np.roll(egs, -1) - np.roll(egs, 1)) / (2.0 * dphi * bw)
    return {
        "phi_grid": phi_grid,
        "spectra": spectra / bw,
        "egs": egs / bw,
        "current": current,
        "bandwidth": np.array([bw]),
    }


# -----------------------------------------------------------------------------
# Robustness diagnostics
# -----------------------------------------------------------------------------
def robustness_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    common = config["common"]
    diag = config["diagnostics"]
    case = diag["robust_case"]
    q = float(common["q"])
    nu = float(common["nu"])
    x_max_fixed = float(diag["robust_xmax_fixed"])

    # Nx convergence at fixed x_max and x0.
    Nx_rows = []
    for Nx in diag["robust_Nx_grid"]:
        Nx = int(Nx)
        ax = x_max_fixed / Nx
        alphas = np.linspace(float(common["alpha_min"]), float(common["alpha_max"]), int(common["Nalpha"]))
        metrics = compute_case_metrics(
            Nx=Nx,
            ax=ax,
            Nphi=Nx,
            alphas=alphas,
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=nu,
            q=q,
            n_mid=int(diag["n_mid"]),
            zero_dos_window_fraction=float(diag["zero_dos_window_fraction"]),
            box_grids=[int(v) for v in diag["box_counting_grids"]],
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
            alpha_ref_ab=float(diag["ab_alpha_ref"]),
            delta_phi=float(diag["stiffness_delta_phi"]),
        )
        Nx_rows.append({"Nx": Nx, "S": metrics["flux_sensitivity"], "rho0": metrics["rho_zero"], "Wh": metrics["avg_horizon_weight_mid"]})

    # alpha-grid convergence at fixed base lattice.
    alpha_rows = []
    for Nalpha in diag["robust_Nalpha_grid"]:
        alphas = np.linspace(float(common["alpha_min"]), float(common["alpha_max"]), int(Nalpha))
        metrics = compute_case_metrics(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            alphas=alphas,
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=nu,
            q=q,
            n_mid=int(diag["n_mid"]),
            zero_dos_window_fraction=float(diag["zero_dos_window_fraction"]),
            box_grids=[int(v) for v in diag["box_counting_grids"]],
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
            alpha_ref_ab=float(diag["ab_alpha_ref"]),
            delta_phi=float(diag["stiffness_delta_phi"]),
        )
        alpha_rows.append({"Nalpha": int(Nalpha), "S": metrics["flux_sensitivity"], "rho0": metrics["rho_zero"], "Wh": metrics["avg_horizon_weight_mid"]})

    # fixed-xmax comparison as x0 changes, with ax fixed and Nx adjusted.
    x0_rows = []
    ax0 = float(common["ax"])
    for x0 in diag["robust_x0_grid"]:
        Nx = int(round((x_max_fixed - float(x0)) / ax0))
        if Nx < 8:
            continue
        alphas = np.linspace(float(common["alpha_min"]), float(common["alpha_max"]), int(common["Nalpha"]))
        metrics = compute_case_metrics(
            Nx=Nx,
            ax=ax0,
            Nphi=int(common["Nphi"]),
            alphas=alphas,
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(x0),
            nu=nu,
            q=q,
            n_mid=int(diag["n_mid"]),
            zero_dos_window_fraction=float(diag["zero_dos_window_fraction"]),
            box_grids=[int(v) for v in diag["box_counting_grids"]],
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
            alpha_ref_ab=float(diag["ab_alpha_ref"]),
            delta_phi=float(diag["stiffness_delta_phi"]),
        )
        x0_rows.append({"x0": float(x0), "Nx": Nx, "S": metrics["flux_sensitivity"], "rho0": metrics["rho_zero"], "Wh": metrics["avg_horizon_weight_mid"]})

    return {
        "Nx": pd.DataFrame(Nx_rows),
        "Nalpha": pd.DataFrame(alpha_rows),
        "x0": pd.DataFrame(x0_rows),
        "case": case,
        "xmax_fixed": x_max_fixed,
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _heatmap(
    ax: plt.Axes,
    array: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    annotate: bool = False,
) -> None:
    im = ax.imshow(array, origin="lower", aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$r_h$")
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ax.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    plt.colorbar(im, ax=ax, shrink=0.84)


def _pivot_metric(df: pd.DataFrame, scenario: str, col: str, L_grid: Sequence[float], rh_grid: Sequence[float]) -> np.ndarray:
    table = (
        df[df["scenario"] == scenario]
        .pivot(index="r_h", columns="L", values=col)
        .reindex(index=rh_grid, columns=L_grid)
    )
    return table.to_numpy()


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    xc, yc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (x >= a) & (x < b) if b < edges[-1] else (x >= a) & (x <= b)
        if np.any(mask):
            xc.append(0.5 * (a + b))
            yc.append(np.mean(y[mask]))
    return np.asarray(xc), np.asarray(yc)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_state_colored_spectra(output_dir: Path, config: Mapping[str, Any]) -> Path:
    common = config["common"]
    diag = config["diagnostics"]
    plots = config["plots"]
    alphas = alpha_grid_from_config(config)
    cases = plots["state_cases"]
    dpi = int(plots["figure_dpi"])
    marker_size = float(plots["spectra_marker_size"])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    last_scatter = None
    for ax, case in zip(axes, cases):
        scan = scan_case_state_data(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            alphas=alphas,
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=float(common["nu"]),
            q=float(common["q"]),
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
        )
        x = scan["x"]
        x_max = float(x.max())
        last_scatter = ax.scatter(
            scan["alpha"],
            scan["energy"] / float(scan["bandwidth"][0]),
            c=scan["xbar"] / x_max,
            s=marker_size,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        ax.set_title(_case_title(case), fontsize=10)
        ax.set_xlabel(r"$\alpha_B$")
        ax.set_ylabel(r"$E/E_{\mathrm{bw}}$")
        ax.set_xlim(float(common["alpha_min"]), float(common["alpha_max"]))
        ax.set_ylim(-1.0, 1.0)
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist(), shrink=0.92)
        cbar.set_label(r"$\bar{x}/x_{\max}$")
    path = output_dir / config["output_names"]["state_spectra_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_ldos_maps(output_dir: Path, config: Mapping[str, Any]) -> Path:
    common = config["common"]
    diag = config["diagnostics"]
    plots = config["plots"]
    alphas = alpha_grid_from_config(config)
    cases = plots["ldos_cases"]
    dpi = int(plots["figure_dpi"])

    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, case in zip(axes, cases):
        data = local_dos_map(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            alphas=alphas,
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=float(common["nu"]),
            q=float(common["q"]),
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
            energy_bins=int(diag["ldos_energy_bins"]),
            sigma_fraction=float(diag["ldos_sigma_fraction"]),
        )
        im = ax.imshow(
            data["ldos"].T,
            origin="lower",
            aspect="auto",
            extent=[float(data["x"].min()), float(data["x"].max()), -1.0, 1.0],
        )
        ax.axvline(float(diag["horizon_cutoff_x"]), color="w", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$E/E_{\mathrm{bw}}$")
        ax.set_title(_case_title(case), fontsize=10)
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.set_label(r"$\rho(x,E)$")
    path = output_dir / config["output_names"]["ldos_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_flux_response_scatter(output_dir: Path, config: Mapping[str, Any]) -> Path:
    common = config["common"]
    diag = config["diagnostics"]
    plots = config["plots"]
    cases = plots["flux_scatter_cases"]
    dpi = int(plots["figure_dpi"])
    marker_size = float(plots.get("scatter_marker_size", 12))
    marker_alpha = float(plots.get("scatter_alpha", 1.0))

    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    last = None
    for ax, case in zip(axes, cases):
        data = flux_response_vs_radius(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=float(common["nu"]),
            q=float(common["q"]),
            alpha_ref=float(diag["alpha_ref_flux_response"]),
            delta_alpha=float(diag["flux_response_delta_alpha"]),
            horizon_cutoff_x=float(diag["horizon_cutoff_x"]),
            energy_window_fraction=float(diag["flux_response_energy_window_fraction"]),
        )

        order = np.argsort(data["horizon_weight"])
        xbar_norm = data["xbar_norm"][order]
        response_norm = data["response_norm"][order]
        horizon_weight = data["horizon_weight"][order]

        last = ax.scatter(
            xbar_norm,
            response_norm,
            c=horizon_weight,
            cmap="plasma",
            s=marker_size,
            alpha=marker_alpha,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.0,
            edgecolors="none",
            rasterized=True,
        )
        xc, yc = _binned_mean(xbar_norm, response_norm, bins=12)
        ax.plot(xc, yc, lw=2.0, color="black")
        ax.set_xlabel(r"$\bar{x}/x_{\max}$")
        ax.set_ylabel(r"$\mathcal{F}_m$")
        ax.set_title(_case_title(case) + "\n" + rf"$\alpha_*= {float(diag['alpha_ref_flux_response']):.2f}$", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
    if last is not None:
        cbar = fig.colorbar(last, ax=axes.tolist(), shrink=0.9)
        cbar.set_label(r"near-horizon weight $W_h$")
    path = output_dir / config["output_names"]["flux_scatter_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_ab_response(output_dir: Path, config: Mapping[str, Any]) -> Path:
    common = config["common"]
    diag = config["diagnostics"]
    plots = config["plots"]
    cases = plots["ab_cases"]
    dpi = int(plots["figure_dpi"])
    nshow = int(diag["ab_n_levels_show"])

    fig, axes = plt.subplots(2, len(cases), figsize=(6.4 * len(cases), 8.0), constrained_layout=True)
    level_norm = None
    level_cmap = plt.get_cmap(str(plots.get("ab_level_cmap", "coolwarm")))
    for col, case in enumerate(cases):
        data_R = ab_spectral_flow_and_current(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=0.0,
            q=float(common["q"]),
            alpha_ref=float(diag["ab_alpha_ref"]),
            Nphi_scan=int(diag["ab_Nphi_scan"]),
        )
        data_NS = ab_spectral_flow_and_current(
            Nx=int(common["Nx"]),
            ax=float(common["ax"]),
            Nphi=int(common["Nphi"]),
            L=float(case["L"]),
            r_h=float(case["r_h"]),
            x0=float(case["x0"]),
            nu=0.5,
            q=float(common["q"]),
            alpha_ref=float(diag["ab_alpha_ref"]),
            Nphi_scan=int(diag["ab_Nphi_scan"]),
        )

        spec = data_R["spectra"]
        mid = spec.shape[1] // 2
        sl = slice(max(mid - nshow // 2, 0), min(mid + nshow // 2, spec.shape[1]))
        offsets = np.arange(sl.start, sl.stop) - mid
        vmax = max(1, int(np.max(np.abs(offsets))))
        level_norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        ax_top = axes[0, col]
        for idx in range(sl.start, sl.stop):
            delta_m = idx - mid
            ax_top.plot(
                data_R["phi_grid"] / (2.0 * np.pi),
                spec[:, idx],
                lw=1.15,
                color=level_cmap(level_norm(delta_m)),
            )
        ax_top.axhline(0.0, color="black", lw=0.6, alpha=0.25)
        ax_top.set_xlabel(r"$\Phi/(2\pi)$")
        ax_top.set_ylabel(r"$E/E_{\mathrm{bw}}$")
        ax_top.set_title(_case_title(case) + "\n" + r"R spectral flow", fontsize=10)

        ax_bottom = axes[1, col]
        ax_bottom.plot(data_R["phi_grid"] / (2.0 * np.pi), data_R["current"], label=r"R ($\nu=0$)")
        ax_bottom.plot(data_NS["phi_grid"] / (2.0 * np.pi), data_NS["current"], label=r"NS ($\nu=1/2$)")
        ax_bottom.set_xlabel(r"$\Phi/(2\pi)$")
        ax_bottom.set_ylabel(r"$I(\Phi)/E_{\mathrm{bw}}$")
        ax_bottom.set_title(_case_title(case) + "\n" + r"persistent current", fontsize=10)
        ax_bottom.legend(frameon=False, fontsize=8)

    if level_norm is not None:
        sm = plt.cm.ScalarMappable(norm=level_norm, cmap=level_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0, :].tolist(), shrink=0.88, pad=0.02)
        cbar.set_label(r"sorted central-level index $\delta m$")

    path = output_dir / config["output_names"]["ab_response_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_annulus_heatmaps(df: pd.DataFrame, output_dir: Path, config: Mapping[str, Any]) -> Path:
    L_grid = [float(v) for v in config["sweep"]["L_grid"]]
    rh_grid = [float(v) for v in config["sweep"]["rh_grid"]]
    dpi = int(config["plots"]["figure_dpi"])
    metrics = [
        ("D_box", r"Box dimension $D_{\mathrm{box}}$"),
        ("flux_sensitivity", r"Flux sensitivity $S$"),
        ("rho_zero", r"Near-zero DOS $\rho_0$"),
        ("avg_ipr_mid", r"Near-zero IPR"),
        ("avg_xi_mid", r"Near-zero length $\xi_2$"),
        ("charge_stiffness", r"AB stiffness $D_\Phi/E_{\mathrm{bw}}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, metrics):
        arr = _pivot_metric(df, "annulus", col, L_grid, rh_grid)
        _heatmap(
            ax,
            arr,
            [str(v) for v in L_grid],
            [str(v) for v in rh_grid],
            title,
            annotate=bool(config["plots"].get("annotate_heatmaps", False)),
        )
    path = output_dir / config["output_names"]["annulus_heatmaps_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_full_heatmaps(df: pd.DataFrame, output_dir: Path, config: Mapping[str, Any]) -> Path:
    L_grid = [float(v) for v in config["sweep"]["L_grid"]]
    rh_grid = [float(v) for v in config["sweep"]["rh_grid"]]
    dpi = int(config["plots"]["figure_dpi"])
    metrics = [
        ("flux_sensitivity", r"Flux sensitivity $S$"),
        ("rho_zero", r"Near-zero DOS $\rho_0$"),
        ("avg_horizon_weight_mid", r"Near-zero horizon weight $W_h$"),
        ("avg_ipr_mid", r"Near-zero IPR"),
        ("avg_xi_mid", r"Near-zero length $\xi_2$"),
        ("charge_stiffness", r"AB stiffness $D_\Phi/E_{\mathrm{bw}}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, metrics):
        arr = _pivot_metric(df, "full_exterior", col, L_grid, rh_grid)
        _heatmap(
            ax,
            arr,
            [str(v) for v in L_grid],
            [str(v) for v in rh_grid],
            title,
            annotate=bool(config["plots"].get("annotate_heatmaps", False)),
        )
    path = output_dir / config["output_names"]["full_heatmaps_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_robustness(output_dir: Path, config: Mapping[str, Any]) -> Path:
    dpi = int(config["plots"]["figure_dpi"])
    rb = robustness_data(config)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)

    def _norm(series: pd.Series) -> np.ndarray:
        s0 = float(series.iloc[0]) if len(series) else 1.0
        if abs(s0) < 1e-15:
            return np.asarray(series, dtype=float)
        return np.asarray(series, dtype=float) / s0

    dfNx = rb["Nx"]
    axes[0].plot(dfNx["Nx"], _norm(dfNx["S"]), marker="o", label=r"$S$")
    axes[0].plot(dfNx["Nx"], _norm(dfNx["rho0"]), marker="s", label=r"$\rho_0$")
    axes[0].plot(dfNx["Nx"], _norm(dfNx["Wh"]), marker="^", label=r"$W_h$")
    axes[0].set_xlabel(r"$N_x = N_\phi$ at fixed $x_{\max}$")
    axes[0].set_ylabel("relative to first point")
    axes[0].set_title("finite-size convergence", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    dfA = rb["Nalpha"]
    axes[1].plot(dfA["Nalpha"], _norm(dfA["S"]), marker="o", label=r"$S$")
    axes[1].plot(dfA["Nalpha"], _norm(dfA["rho0"]), marker="s", label=r"$\rho_0$")
    axes[1].plot(dfA["Nalpha"], _norm(dfA["Wh"]), marker="^", label=r"$W_h$")
    axes[1].set_xlabel(r"$N_\alpha$")
    axes[1].set_ylabel("relative to first point")
    axes[1].set_title("flux-grid convergence", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)

    dfx0 = rb["x0"]
    axes[2].plot(dfx0["x0"], _norm(dfx0["S"]), marker="o", label=r"$S$")
    axes[2].plot(dfx0["x0"], _norm(dfx0["rho0"]), marker="s", label=r"$\rho_0$")
    axes[2].plot(dfx0["x0"], _norm(dfx0["Wh"]), marker="^", label=r"$W_h$")
    axes[2].set_xlabel(r"$x_0$ at fixed $x_{\max}$")
    axes[2].set_ylabel("relative to first point")
    axes[2].set_title("fixed-window horizon comparison", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)

    case = rb["case"]
    sup_title = (
        rf"robustness checks for $L={_fmt_param(case['L'])}$, $r_h={_fmt_param(case['r_h'])}$, "
        + rf"$x_{{\max}}={_fmt_param(rb['xmax_fixed'])}$"
    )
    fig.suptitle(sup_title, fontsize=12)

    path = output_dir / config["output_names"]["robustness_png"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def write_summary(df: pd.DataFrame, output_dir: Path, config: Mapping[str, Any]) -> Path:
    ann = df[df["scenario"] == "annulus"]
    full = df[df["scenario"] == "full_exterior"]
    summary = {
        "config_used": _to_jsonable(config),
        "annulus_fixed_rh_1": ann[np.isclose(ann["r_h"], 1.0)]
        .sort_values("L")[["L", "D_box", "flux_sensitivity", "rho_zero", "charge_stiffness"]]
        .to_dict(orient="records"),
        "full_fixed_L_12": full[np.isclose(full["L"], 12.0)]
        .sort_values("r_h")[["r_h", "flux_sensitivity", "avg_horizon_weight_mid", "rho_zero", "charge_stiffness"]]
        .to_dict(orient="records"),
    }
    path = output_dir / config["output_names"]["summary_json"]
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    output_dir: str | Path = ".",
    config: Mapping[str, Any] | None = None,
    return_dataframe: bool = False,
) -> Dict[str, str] | Tuple[Dict[str, str], pd.DataFrame, Dict[str, Any]]:
    config = resolve_config(config)
    configure_matplotlib(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cached_metrics_if_compatible(output_dir, config)
    metrics_csv = output_dir / config["output_names"]["metrics_csv"]
    if df is None:
        df = parameter_sweep(config)
        df.to_csv(metrics_csv, index=False)

    config_json = save_config_snapshot(config, output_dir)
    artifacts = {
        "metrics_csv": str(metrics_csv),
        "config_json": str(config_json),
        "state_spectra_png": str(plot_state_colored_spectra(output_dir, config)),
        "ldos_png": str(plot_ldos_maps(output_dir, config)),
        "flux_scatter_png": str(plot_flux_response_scatter(output_dir, config)),
        "ab_response_png": str(plot_ab_response(output_dir, config)),
        "annulus_heatmaps_png": str(plot_annulus_heatmaps(df, output_dir, config)),
        "full_heatmaps_png": str(plot_full_heatmaps(df, output_dir, config)),
        "robustness_png": str(plot_robustness(output_dir, config)),
        "summary_json": str(write_summary(df, output_dir, config)),
    }

    if return_dataframe:
        return artifacts, df, config
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BTZ-derived curved Harper analysis.")
    parser.add_argument("--output-dir", default=".", help="Directory where figures and tables will be written.")
    args = parser.parse_args()
    artifacts = main(output_dir=args.output_dir)
    print(json.dumps(artifacts, indent=2))
