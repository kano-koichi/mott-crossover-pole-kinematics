"""Regenerate the controlled self-energy pole family used in the manuscript.

The family keeps the interaction path and total pole-residue evolution fixed and
changes only the pole-energy trajectory through ``alpha``.  No precomputed
spectra, observables, or marker positions are consumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import completion_curve, completion_marker, make_grid, prepare_readout_weights, retarded_green_function, scalar_readouts, spectral_function
from common import make_band
from common import write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "pole_kinematics"
DETAIL = RESULTS / "pole_kinematics_detail"


def _sigma_w(omega: np.ndarray, interaction: float, pole_energy: float, eta: float) -> np.ndarray:
    z = np.asarray(omega, dtype=float) + 1j * float(eta)
    residue = float(interaction) ** 2 / 4.0
    return residue * z / (z * z - float(pole_energy) ** 2)


def _readouts_for_sigma(grid, band, weights, sigma_w: np.ndarray, lattice_eta: float) -> tuple[dict[str, float], float]:
    sigma = np.broadcast_to(sigma_w, grid.kx.shape + (grid.omega.size,))
    spectral = spectral_function(retarded_green_function(grid, band, sigma, eta=float(lattice_eta)))
    values = scalar_readouts(grid, spectral, weights)
    local_dos = np.mean(spectral, axis=(0, 1))
    zero_index = int(np.argmin(np.abs(grid.omega)))
    return values, float(local_dos[zero_index])


def _family_state(lam: float, alpha: float, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    u_over_uc = float(cfg["u_max_ratio"]) * float(lam)
    interaction = float(cfg["critical_interaction"]) * u_over_uc
    z_formula = max(0.0, 1.0 - u_over_uc**2)
    omega_dmft = float(cfg["pole_energy_prefactor"]) * np.sqrt(z_formula)
    pole_energy = float(alpha) * omega_dmft
    residue = interaction**2 / 4.0
    return interaction, z_formula, pole_energy, residue


def _evaluate_family(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["controlled_pole_family"]
    band_data = next(item for item in config["bands"] if item["id"] == cfg["band_id"])
    band = make_band(band_data)
    grid = make_grid(int(cfg["nk"]), int(cfg["nw"]), float(cfg["wmax"]), float(config["grid"]["temperature"]))
    readout_cfg = dict(config["readout"])
    readout_cfg["hubbard_min"] = float(cfg["high_energy_window"][0])
    readout_cfg["hubbard_max"] = float(cfg["high_energy_window"][1])
    weights = prepare_readout_weights(grid, band, readout_cfg, readout_temperature=float(cfg["readout_temperature"]))
    lambdas = np.linspace(0.0, 1.0, int(cfg["n_lambda"]))
    alphas = np.asarray(cfg["alphas"], dtype=float)
    readout_names = [
        "transport_quadratic", "transport_linear", "low_energy_spectral",
        "antinodal_spectral", "nodal_spectral", "high_energy_spectral", "d_wave_spectral",
    ]
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for lam in lambdas:
            interaction, z_formula, pole_energy, residue = _family_state(float(lam), float(alpha), cfg)
            sigma_w = _sigma_w(grid.omega, interaction, pole_energy, float(cfg["self_energy_broadening"]))
            values, a0 = _readouts_for_sigma(grid, band, weights, sigma_w, float(cfg["lattice_broadening"]))
            rows.append({
                "alpha": float(alpha), "lambda": float(lam),
                "u_over_uc": float(cfg["u_max_ratio"]) * float(lam),
                "R_total": float(residue), "pole_energy": float(pole_energy),
                "Z_dmft_formula": float(z_formula), "A0": float(a0),
                **{name: float(values[name]) for name in readout_names},
            })
    curves = pd.DataFrame(rows)
    fractions = np.asarray(config["path_discrimination"]["dense_fractions"], dtype=float)
    marker_rows: list[dict[str, Any]] = []
    for alpha, group in curves.groupby("alpha"):
        group = group.sort_values("lambda")
        x = group["lambda"].to_numpy(float)
        t = group["transport_quadratic"].to_numpy(float)
        lo = group["low_energy_spectral"].to_numpy(float)
        hi = group["high_energy_spectral"].to_numpy(float)
        for fraction in fractions:
            mt = completion_marker(t, x, float(fraction))
            ml = completion_marker(lo, x, float(fraction))
            mh = completion_marker(hi, x, float(fraction))
            marker_rows.append({
                "alpha": float(alpha), "fraction": float(fraction),
                "transport_quadratic": float(mt), "low_energy_spectral": float(ml),
                "high_energy_spectral": float(mh), "high_minus_low": float(mh - ml),
            })
    return curves, pd.DataFrame(marker_rows)


def _pinning_gate(curves: pd.DataFrame) -> pd.DataFrame:
    moving = curves[np.isclose(curves["alpha"], 1.0)].sort_values("lambda").copy()
    pinned = curves[np.isclose(curves["alpha"], 0.0)].sort_values("lambda").copy()
    out = pd.DataFrame({
        "lambda": moving["lambda"].to_numpy(float),
        "u_over_uc": moving["u_over_uc"].to_numpy(float),
        "Z": moving["Z_dmft_formula"].to_numpy(float),
        "R_total": moving["R_total"].to_numpy(float),
        "pole_energy": moving["pole_energy"].to_numpy(float),
        "A0_dmft": moving["A0"].to_numpy(float),
        "A0_pinned": pinned["A0"].to_numpy(float),
        "HE_completion": completion_curve(moving["high_energy_spectral"].to_numpy(float)),
        "LE_completion": completion_curve(moving["low_energy_spectral"].to_numpy(float)),
        "HE_value": moving["high_energy_spectral"].to_numpy(float),
        "LE_value": moving["low_energy_spectral"].to_numpy(float),
    })
    a0m = float(out["A0_dmft"].iloc[0]); a0p = float(out["A0_pinned"].iloc[0])
    out["A0_dmft_rel"] = out["A0_dmft"] / a0m
    out["A0_pinned_rel"] = out["A0_pinned"] / a0p
    return out


def _competition(config: dict[str, Any]) -> pd.DataFrame:
    """Independent progression-law test with fixed endpoints.

    Residue and pole-motion progress are assigned independent powers while the
    initial/final residue and pole energies remain fixed.  This is a bounded SM
    diagnostic, not a second headline model.
    """
    cfg = config["controlled_pole_family"]
    comp_cfg = cfg["residue_motion_competition"]
    band_data = next(item for item in config["bands"] if item["id"] == cfg["band_id"])
    band = make_band(band_data)
    grid = make_grid(int(comp_cfg["nk"]), int(comp_cfg["nw"]), float(cfg["wmax"]), float(config["grid"]["temperature"]))
    readout_cfg = dict(config["readout"])
    readout_cfg["hubbard_min"] = float(cfg["high_energy_window"][0])
    readout_cfg["hubbard_max"] = float(cfg["high_energy_window"][1])
    weights = prepare_readout_weights(grid, band, readout_cfg, readout_temperature=float(cfg["readout_temperature"]))
    lambdas = np.linspace(0.0, 1.0, int(comp_cfg["n_lambda"]))
    u_final = float(cfg["critical_interaction"]) * float(cfg["u_max_ratio"])
    r_final = u_final**2 / 4.0
    omega_initial = float(cfg["pole_energy_prefactor"])
    rows: list[dict[str, Any]] = []
    for p_r in comp_cfg["residue_exponents"]:
        for p_m in comp_cfg["motion_exponents"]:
            low: list[float] = []; high: list[float] = []
            for lam in lambdas:
                residue = r_final * float(lam) ** float(p_r)
                interaction = 2.0 * np.sqrt(max(residue, 0.0))
                pole_energy = omega_initial * (1.0 - float(lam) ** float(p_m))
                sigma_w = _sigma_w(grid.omega, interaction, pole_energy, float(cfg["self_energy_broadening"]))
                values, _ = _readouts_for_sigma(grid, band, weights, sigma_w, float(cfg["lattice_broadening"]))
                low.append(float(values["low_energy_spectral"])); high.append(float(values["high_energy_spectral"]))
            for fraction in config["markers"]["fractions"]:
                ml = completion_marker(np.asarray(low), lambdas, float(fraction))
                mh = completion_marker(np.asarray(high), lambdas, float(fraction))
                rows.append({"p_residue": float(p_r), "p_motion": float(p_m), "fraction": float(fraction), "low": ml, "high": mh, "high_minus_low": mh - ml})
    return pd.DataFrame(rows)


def generate(config: dict[str, Any]) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True); DETAIL.mkdir(parents=True, exist_ok=True)
    curves, markers = _evaluate_family(config)
    pinning = _pinning_gate(curves)
    competition = _competition(config)
    summary_rows = []
    for alpha, group in markers.groupby("alpha"):
        d = group.sort_values("fraction")["high_minus_low"].to_numpy(float)
        def _at(q: float) -> float:
            row = group[np.isclose(group["fraction"], q)]
            return float(row.iloc[0]["high_minus_low"]) if len(row) else float("nan")
        summary_rows.append({
            "alpha": float(alpha), "min_delta": float(np.min(d)), "max_delta": float(np.max(d)),
            "all_positive": bool(np.all(d > 0)), "all_negative": bool(np.all(d < 0)),
            "neg_fraction": float(np.mean(d < 0)), "delta_q25": _at(0.25), "delta_q50": _at(0.50), "delta_q75": _at(0.75),
        })
    summary = pd.DataFrame(summary_rows)
    outputs = [
        OUT / "exact_pole_family_markers.csv", OUT / "pinning_gate.csv",
        DETAIL / "exact_pole_family_curves.csv", DETAIL / "exact_pole_family_summary.csv",
        DETAIL / "residue_motion_competition.csv", DETAIL / "pole_family_generation.json",
    ]
    markers.to_csv(outputs[0], index=False); pinning.to_csv(outputs[1], index=False)
    curves.drop(columns=["A0"]).to_csv(outputs[2], index=False); summary.to_csv(outputs[3], index=False)
    competition.to_csv(outputs[4], index=False)
    cfg = config["controlled_pole_family"]
    write_json(outputs[5], {
        "generator": "src/pole_family.py", "formula": "Sigma(z)=R z/(z^2-Omega^2), R=U^2/4, Omega=alpha*6*sqrt(max(0,1-(U/Uc)^2))",
        "critical_interaction": float(cfg["critical_interaction"]), "u_max_ratio": float(cfg["u_max_ratio"]),
        "alpha_count": int(len(cfg["alphas"])), "lambda_count": int(cfg["n_lambda"]),
        "marker_fraction_count": int(len(config["path_discrimination"]["dense_fractions"])),
        "inputs_are_precomputed_results": False,
    })
    return outputs
