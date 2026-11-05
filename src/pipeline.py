"""Current numerical pipeline for self-energy-pole crossover ordering.

The active code contains only the controlled comparison, two-site-DMFT
realization, and matched self-energy-evolution comparison used by the manuscript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from common import (  # noqa: E402
    completion_curve,
    completion_marker,
    make_grid,
    marker_diagnostics,
    physical_diagnostics,
    prepare_readout_weights,
    retarded_green_function,
    scalar_readouts,
    spectral_function,
    spectral_function_raw,
)
from common import (  # noqa: E402
    MODEL_FUNCTIONS,
    band_second_moment,
    dispersion_square,
    make_band,
    solve_two_site_dmft,
    two_site_dmft_self_energy_on_grid,
)
from common import release_memory, write_csv, write_json  # noqa: E402

RESULTS = ROOT / "results"
TWO_SITE_DMFT = RESULTS / "two_site_dmft"
PATH_DISCRIMINATION = RESULTS / "path_discrimination"

def _controlled_path_readouts(
    grid,
    band,
    family: str,
    params: dict[str, Any],
    path_name: str,
    readouts: list[str],
    lambdas: np.ndarray,
    fractions: list[float],
    config: dict[str, Any],
    readout_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate direct observable curves and first-passage markers on a controlled path."""
    active_readout_config = config["readout"] if readout_config is None else readout_config
    weights = prepare_readout_weights(grid, band, active_readout_config)
    direct_values = {name: [] for name in readouts}
    curve_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []

    for lam in lambdas:
        sigma = MODEL_FUNCTIONS[family](
            grid.omega, grid.kx, grid.ky, band, float(lam), params
        )
        spectral = spectral_function(retarded_green_function(grid, band, sigma))
        values = scalar_readouts(grid, spectral, weights)
        for name in readouts:
            direct_values[name].append(float(values[name]))

    for name in readouts:
        direct = np.asarray(direct_values[name], dtype=float)
        direct_completion = completion_curve(direct)
        diagnostics = marker_diagnostics(
            direct,
            lambdas,
            float(config["markers"]["minimum_relative_change"]),
            float(config["markers"].get("minimum_monotonicity_score", 0.85)),
            float(config["markers"].get("monotonicity_tolerance", 1e-10)),
        )
        for index, lam in enumerate(lambdas):
            curve_rows.append(
                {
                    "benchmark": path_name,
                    "family": family,
                    "dispersion": band.band_id,
                    "readout": name,
                    "lambda": float(lam),
                    "direct_value": float(direct[index]),
                    "direct_completion": float(direct_completion[index]),
                }
            )
        for fraction in fractions:
            marker_rows.append(
                {
                    "benchmark": path_name,
                    "family": family,
                    "dispersion": band.band_id,
                    "readout": name,
                    "fraction": float(fraction),
                    "direct_marker": completion_marker(direct, lambdas, float(fraction)),
                    **diagnostics,
                }
            )
    return curve_rows, marker_rows



def _two_site_dmft_readout_config(
    config: dict[str, Any],
    dmft_cfg: dict[str, Any],
    grid,
    band,
) -> dict[str, Any]:
    readout_cfg = dict(config["readout"])
    epsilon = dispersion_square(grid.kx, grid.ky, band)
    band_edge = float(np.max(np.abs(epsilon - np.mean(epsilon))))
    readout_cfg["hubbard_min"] = (
        float(dmft_cfg["high_energy_min_band_edge_factor"]) * band_edge
    )
    readout_cfg["hubbard_max"] = min(
        float(dmft_cfg["high_energy_max_band_edge_factor"]) * band_edge,
        0.95 * float(dmft_cfg["wmax"]),
    )
    if readout_cfg["hubbard_max"] <= readout_cfg["hubbard_min"]:
        raise ValueError("two-site DMFT high-energy observable window is empty")
    return readout_cfg


def _run_two_site_dmft_model_path(
    config: dict[str, Any],
    model_data: dict[str, Any],
    nk: int,
    nw: int,
    n_lambda: int,
) -> dict[str, list[dict[str, Any]]]:
    dmft_cfg = config["two_site_dmft"]
    grid = make_grid(
        int(nk),
        int(nw),
        float(dmft_cfg["wmax"]),
        float(dmft_cfg["readout_temperature"]),
    )
    band = make_band(model_data)
    second_moment = band_second_moment(
        band, int(dmft_cfg["moment_nk"])
    )
    critical_interaction = 6.0 * np.sqrt(second_moment)
    lambdas = np.linspace(0.0, 1.0, int(n_lambda))
    u_ratios = (
        float(dmft_cfg["u_min_ratio"])
        + lambdas
        * (
            float(dmft_cfg["u_max_ratio"])
            - float(dmft_cfg["u_min_ratio"])
        )
    )
    interactions = critical_interaction * u_ratios
    readout_cfg = _two_site_dmft_readout_config(
        config, dmft_cfg, grid, band
    )
    weights = prepare_readout_weights(grid, band, readout_cfg)
    readouts = list(dmft_cfg["readouts"])
    fractions = [float(value) for value in config["markers"]["fractions"]]

    solver_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    direct_values = {name: [] for name in readouts}

    for index, (lam, interaction) in enumerate(zip(lambdas, interactions)):
        print(
            f"  two-site DMFT {band.band_id} [{index + 1}/{len(lambdas)}] "
            f"U/Uc={interaction / critical_interaction:.3f}",
            flush=True,
        )
        solution = solve_two_site_dmft(
            float(interaction), second_moment, dmft_cfg
        )
        sigma_w = two_site_dmft_self_energy_on_grid(
            grid.omega, solution, dmft_cfg
        )
        sigma_kw = np.broadcast_to(
            sigma_w[None, None, :],
            grid.kx.shape + (grid.omega.size,),
        )
        green = retarded_green_function(
            grid,
            band,
            sigma_kw,
            eta=float(dmft_cfg["lattice_broadening"]),
        )
        raw = spectral_function_raw(green)
        spectral = np.maximum(raw, 0.0)
        diagnostics = physical_diagnostics(
            grid, spectral, raw, sigma_kw
        )
        values = scalar_readouts(grid, spectral, weights)
        for name in readouts:
            direct_values[name].append(float(values[name]))

        solver_rows.append(
            {
                "model": band.band_id,
                "lambda": float(lam),
                "u_over_uc": float(interaction / critical_interaction),
                **{
                    key: value
                    for key, value in solution.items()
                    if key != "impurity_data"
                },
                "high_energy_window_min": float(readout_cfg["hubbard_min"]),
                "high_energy_window_max": float(readout_cfg["hubbard_max"]),
                "impurity_inverse_temperature": float(dmft_cfg["impurity_inverse_temperature"]),
                "readout_temperature": float(dmft_cfg["readout_temperature"]),
                "temperature_convention": "ground-state-proxy impurity spectrum plus finite-width observable-window kernel",
                **diagnostics,
            }
        )


    for name in readouts:
        direct = np.asarray(direct_values[name], dtype=float)
        completion_direct = completion_curve(direct)
        diagnostics = marker_diagnostics(
            direct,
            lambdas,
            float(config["markers"]["minimum_relative_change"]),
            float(config["markers"].get("minimum_monotonicity_score", 0.85)),
            float(config["markers"].get("monotonicity_tolerance", 1e-10)),
        )
        for index, lam in enumerate(lambdas):
            curve_rows.append(
                {
                    "model": band.band_id,
                    "readout": name,
                    "lambda": float(lam),
                    "u_over_uc": float(u_ratios[index]),
                    "direct_value": float(direct[index]),
                    "direct_completion": float(completion_direct[index]),
                }
            )
        for fraction in fractions:
            direct_marker = completion_marker(
                direct, lambdas, fraction
            )
            marker_rows.append(
                {
                    "model": band.band_id,
                    "readout": name,
                    "fraction": fraction,
                    "direct_marker": direct_marker,
                    **diagnostics,
                }
            )

    release_memory()
    return {
        "solver": solver_rows,
        "curves": curve_rows,
        "markers": marker_rows,
    }


def run_two_site_dmft_spectral_snapshots(config: dict[str, Any]) -> Path:
    """Save representative local spectra for the two-site-DMFT Main figure."""
    dmft_cfg = config["two_site_dmft"]
    model_data = next(item for item in dmft_cfg["models"] if item["id"] == "two_site_square")
    grid = make_grid(
        int(dmft_cfg["nk"]), int(dmft_cfg["nw"]), float(dmft_cfg["wmax"]),
        float(dmft_cfg["readout_temperature"]),
    )
    band = make_band(model_data)
    second_moment = band_second_moment(band, int(dmft_cfg["moment_nk"]))
    critical_interaction = 6.0 * np.sqrt(second_moment)
    ratios = [0.0, 0.55, 0.85, 1.02]
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        solution = solve_two_site_dmft(float(ratio * critical_interaction), second_moment, dmft_cfg)
        sigma_w = two_site_dmft_self_energy_on_grid(grid.omega, solution, dmft_cfg)
        sigma_kw = np.broadcast_to(sigma_w[None, None, :], grid.kx.shape + (grid.omega.size,))
        green = retarded_green_function(
            grid, band, sigma_kw, eta=float(dmft_cfg["lattice_broadening"])
        )
        local_dos = np.mean(spectral_function(green), axis=(0, 1))
        for omega, value in zip(grid.omega, local_dos):
            rows.append({"model": "two_site_square", "u_over_uc": ratio,
                         "omega": float(omega), "local_dos": float(value)})
    path = TWO_SITE_DMFT / "two_site_dmft_spectral_snapshots.csv"
    write_csv(path, rows)
    release_memory()
    return path


def summarize_two_site_dmft(config: dict[str, Any], generated: list[Path]) -> Path:
    dmft_cfg = config["two_site_dmft"]
    solver = pd.read_csv(TWO_SITE_DMFT / "two_site_dmft_solver.csv")
    markers = pd.read_csv(TWO_SITE_DMFT / "two_site_dmft_markers.csv")
    convergence = pd.read_csv(TWO_SITE_DMFT / "two_site_dmft_convergence.csv")
    broadening = pd.read_csv(
        TWO_SITE_DMFT / "two_site_dmft_broadening_robustness.csv"
    )
    high_energy_windows = pd.read_csv(
        TWO_SITE_DMFT / "two_site_dmft_high_energy_window_robustness.csv"
    )

    pivot = markers.pivot_table(
        index=["model", "fraction"],
        columns="readout",
        values="direct_marker",
    ).reset_index()
    readouts = list(dmft_cfg["readouts"])
    pivot["marker_span"] = pivot[readouts].max(axis=1) - pivot[readouts].min(axis=1)
    pivot["ordering"] = pivot.apply(
        lambda row: " < ".join(
            sorted(readouts, key=lambda name: float(row[name]))
        ),
        axis=1,
    )
    ordering_rows = pivot[
        ["model", "fraction", "marker_span", "ordering"]
    ].to_dict(orient="records")

    branch_counts = (
        solver.groupby(["model", "branch"]).size().unstack(fill_value=0)
    )
    both_branches = True
    for model in solver["model"].unique():
        branches = set(
            solver.loc[solver["model"] == model, "branch"].astype(str)
        )
        both_branches = both_branches and {
            "metallic",
            "insulating",
        }.issubset(branches)

    summary = {
        "stage": "two-site DMFT validation",
        "scope": "minimal Hamiltonian-based local-correlation test using ground-state-proxy two-site DMFT",
        "temperature_convention": "ground-state-proxy impurity spectrum plus a separate finite-width observable-window kernel",
        "impurity_inverse_temperature": float(dmft_cfg["impurity_inverse_temperature"]),
        "readout_temperature": float(dmft_cfg["readout_temperature"]),
        "models": sorted(solver["model"].unique().tolist()),
        "maximum_analytic_z_error": float(
            solver["analytic_z_error"].abs().max()
        ),
        "maximum_self_consistency_residual": float(
            solver["self_consistency_residual"].abs().max()
        ),
        "maximum_impurity_occupancy_error": float(
            (solver["impurity_occupancy"] - 1.0).abs().max()
        ),
        "maximum_spectral_sum_rule_error": float(
            solver["sum_rule_abs_error_max"].max()
        ),
        "maximum_density_error": float(
            (solver["density"] - 1.0).abs().max()
        ),
        "maximum_im_sigma": float(solver["im_sigma_max"].max()),
        "maximum_convergence_marker_spread": float(
            convergence["marker_spread"].max()
        ),
        "broadening_ordering_stable": bool(
            broadening["ordering_stable"].all()
        ),
        "high_energy_window_downstream_order_stable": bool(
            high_energy_windows["downstream_order_stable_across_windows"].all()
            and high_energy_windows["high_energy_before_low_energy"].all()
        ),
        "high_energy_window_full_three_stage_order_stable": bool(
            high_energy_windows["full_order_stable_across_windows"].all()
            and high_energy_windows["full_three_stage_order_pass"].all()
        ),
        "minimum_high_energy_to_low_energy_margin": float(
            high_energy_windows["high_energy_to_low_energy_margin"].min()
        ),
        "minimum_quadratic_to_high_energy_margin": float(
            high_energy_windows["quadratic_to_high_energy_margin"].min()
        ),
        "quadratic_before_high_energy_case_fraction": float(
            high_energy_windows["quadratic_before_high_energy"].mean()
        ),
        "high_energy_window_count": int(high_energy_windows["window_id"].nunique()),
        "both_metallic_and_insulating_branches_present": bool(both_branches),
        "nontrivial_marker_separation_all_models_and_fractions": bool(
            (
                pivot["marker_span"]
                >= float(dmft_cfg["minimum_marker_separation"])
            ).all()
        ),
        "quadratic_transport_first_baseline_window_all_models_and_fractions": bool(
            (
                pivot["transport_quadratic"]
                < pivot[[name for name in readouts if name != "transport_quadratic"]].min(axis=1)
            ).all()
        ),
        "marker_orderings": ordering_rows,
        "branch_counts": {
            str(index): {
                str(column): int(branch_counts.loc[index, column])
                for column in branch_counts.columns
            }
            for index in branch_counts.index
        },
        "manuscript_figure_architecture": "two-site DMFT supports Main Figures 1-4 and Supplementary Figures S2-S4",
    }
    summary["analytic_solver_anchor_pass"] = bool(
        summary["maximum_analytic_z_error"]
        <= float(dmft_cfg["pass_analytic_z_error"])
    )
    summary["self_consistency_pass"] = bool(
        summary["maximum_self_consistency_residual"]
        <= float(dmft_cfg["pass_self_consistency_residual"])
    )
    summary["impurity_occupancy_pass"] = bool(
        summary["maximum_impurity_occupancy_error"]
        <= float(dmft_cfg["pass_impurity_occupancy_error"])
    )
    summary["spectral_sum_rule_pass"] = bool(
        summary["maximum_spectral_sum_rule_error"]
        <= float(dmft_cfg["pass_sum_rule_abs_error"])
    )
    summary["density_pass"] = bool(
        summary["maximum_density_error"]
        <= float(dmft_cfg["pass_density_error"])
    )
    summary["causality_pass"] = bool(
        summary["maximum_im_sigma"]
        <= float(dmft_cfg["pass_causality_im_sigma"])
    )
    summary["convergence_pass"] = bool(
        convergence["pass_two_site_dmft_convergence"].all()
    )
    summary["validation_pass"] = bool(
        summary["analytic_solver_anchor_pass"]
        and summary["self_consistency_pass"]
        and summary["impurity_occupancy_pass"]
        and summary["spectral_sum_rule_pass"]
        and summary["density_pass"]
        and summary["causality_pass"]
        and summary["convergence_pass"]
        and summary["broadening_ordering_stable"]
        and summary["high_energy_window_downstream_order_stable"]
        and summary["both_metallic_and_insulating_branches_present"]
        and summary["nontrivial_marker_separation_all_models_and_fractions"]
    )
    path = TWO_SITE_DMFT / "two_site_dmft_summary.json"
    write_json(path, summary)
    generated.append(path)
    return path

def _dense_marker_rows(
    curves: pd.DataFrame,
    setting_columns: list[str],
    layer: str,
    fractions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped_columns = setting_columns + ["readout"]
    for keys, group in curves.groupby(grouped_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        setting_values = keys[:-1]
        readout = str(keys[-1])
        group = group.sort_values("lambda")
        lambdas = group["lambda"].to_numpy(float)
        values = group["direct_value"].to_numpy(float)
        base = {"layer": layer, "readout": readout}
        base.update(dict(zip(setting_columns, setting_values)))
        for fraction in fractions:
            rows.append({
                **base,
                "fraction": float(fraction),
                "direct_marker": completion_marker(values, lambdas, float(fraction)),
            })
    return rows


def _run_matched_candidate_comparison(
    config: dict[str, Any], dense_fractions: np.ndarray,
) -> list[Path]:
    """Build the matched observable comparison used by Main Figure 4.

    The controlled energy-transfer path is evaluated on the same bare square
    dispersion as ``two_site_square``.  The low- and high-energy observable
    definitions use the baseline two-site-DMFT window construction, so the two
    evolutions differ in their self-energy trajectory rather than in the bare band or
    spectral observable window.
    """
    two_site_dmft_cfg = config["two_site_dmft"]
    path_discrimination_cfg = config["path_discrimination"]
    controlled_grid = path_discrimination_cfg["controlled_grid"]
    match_cfg = path_discrimination_cfg.get("matched_candidate_comparison", {})
    controlled_band_id = str(match_cfg.get("controlled_band_id", "square_reference"))
    two_site_model_id = str(match_cfg.get("two_site_model_id", "two_site_square"))

    controlled_data = next(
        item for item in config["bands"] if str(item["id"]) == controlled_band_id
    )
    dmft_data = next(
        item for item in two_site_dmft_cfg["models"] if str(item["id"]) == two_site_model_id
    )
    controlled_band = make_band(controlled_data)
    dmft_band = make_band(dmft_data)
    grid = make_grid(
        int(controlled_grid["nk"]),
        int(controlled_grid["nw"]),
        float(controlled_grid["wmax"]),
        float(config["grid"]["temperature"]),
    )
    if not np.isclose(float(grid.temperature), float(two_site_dmft_cfg["readout_temperature"])):
        raise ValueError("Matched evolution comparison requires the same observable-window temperature")

    eps_control = dispersion_square(grid.kx, grid.ky, controlled_band)
    eps_dmft = dispersion_square(grid.kx, grid.ky, dmft_band)
    max_bare_dispersion_difference = float(np.max(np.abs(eps_control - eps_dmft)))
    tolerance = float(match_cfg.get("bare_dispersion_tolerance", 1e-12))
    if max_bare_dispersion_difference > tolerance:
        raise ValueError(
            "Controlled and DMFT evolutions do not share the required bare dispersion"
        )

    readout_cfg = _two_site_dmft_readout_config(config, two_site_dmft_cfg, grid, controlled_band)
    readouts = list(two_site_dmft_cfg["readouts"])
    lambdas = np.linspace(0.0, 1.0, int(controlled_grid["n_lambda"]))
    controlled_curves, controlled_markers = _controlled_path_readouts(
        grid,
        controlled_band,
        "controlled_mott_pole",
        dict(config["controlled_mott_pole"]),
        "matched_energy_transfer",
        readouts,
        lambdas,
        [float(value) for value in dense_fractions],
        config,
        readout_config=readout_cfg,
    )
    ccurves = pd.DataFrame(controlled_curves)
    ccurves["layer"] = "controlled_matched"
    ccurves["setting"] = "matched_controlled_square"
    ccurves["candidate"] = "controlled"
    ccurves["high_energy_window_min"] = float(readout_cfg["hubbard_min"])
    ccurves["high_energy_window_max"] = float(readout_cfg["hubbard_max"])
    ccurves["readout_temperature"] = float(grid.temperature)

    cmarkers = pd.DataFrame(controlled_markers)
    cmarkers["layer"] = "controlled_matched"
    cmarkers["setting"] = "matched_controlled_square"
    cmarkers["candidate"] = "controlled"
    cmarkers["high_energy_window_min"] = float(readout_cfg["hubbard_min"])
    cmarkers["high_energy_window_max"] = float(readout_cfg["hubbard_max"])
    cmarkers["readout_temperature"] = float(grid.temperature)

    dmft_curves = pd.read_csv(TWO_SITE_DMFT / "two_site_dmft_curves.csv")
    dmft_curves = dmft_curves[dmft_curves["model"] == two_site_model_id].copy()
    dmft_curves["layer"] = "two_site_microscopic"
    dmft_curves["setting"] = two_site_model_id
    dmft_curves["candidate"] = "DMFT"
    dmft_curves["high_energy_window_min"] = float(readout_cfg["hubbard_min"])
    dmft_curves["high_energy_window_max"] = float(readout_cfg["hubbard_max"])
    dmft_curves["readout_temperature"] = float(two_site_dmft_cfg["readout_temperature"])

    dmft_marker_rows = _dense_marker_rows(
        dmft_curves,
        ["model"],
        "two_site_microscopic",
        dense_fractions,
    )
    dmarkers = pd.DataFrame(dmft_marker_rows)
    dmarkers["setting"] = two_site_model_id
    dmarkers["candidate"] = "DMFT"
    dmarkers["high_energy_window_min"] = float(readout_cfg["hubbard_min"])
    dmarkers["high_energy_window_max"] = float(readout_cfg["hubbard_max"])
    dmarkers["readout_temperature"] = float(two_site_dmft_cfg["readout_temperature"])

    curve_columns = [
        "layer", "setting", "candidate", "readout", "lambda", "direct_value",
        "direct_completion", "high_energy_window_min",
        "high_energy_window_max", "readout_temperature",
    ]
    curves = pd.concat(
        [ccurves[curve_columns], dmft_curves[curve_columns]], ignore_index=True
    )
    marker_columns = [
        "layer", "setting", "candidate", "readout", "fraction", "direct_marker",
        "high_energy_window_min", "high_energy_window_max", "readout_temperature",
    ]
    markers = pd.concat(
        [cmarkers[marker_columns], dmarkers[marker_columns]], ignore_index=True
    )

    curve_path = PATH_DISCRIMINATION / "matched_candidate_curves.csv"
    marker_path = PATH_DISCRIMINATION / "matched_candidate_markers_dense.csv"
    write_csv(curve_path, curves.to_dict("records"))
    write_csv(marker_path, markers.to_dict("records"))

    pivot = markers.pivot_table(
        index=["candidate", "fraction"], columns="readout", values="direct_marker"
    ).reset_index()
    pivot["high_minus_low"] = (
        pivot["high_energy_spectral"] - pivot["low_energy_spectral"]
    )
    control_delta = pivot[pivot["candidate"] == "controlled"]["high_minus_low"]
    dmft_delta = pivot[pivot["candidate"] == "DMFT"]["high_minus_low"]
    main_fractions = [float(value) for value in config["markers"]["fractions"]]
    main_rows = pivot[pivot["fraction"].isin(main_fractions)].copy()

    summary = {
        "comparison_role": "matched comparison of two self-energy evolutions for Main Figure 4",
        "controlled_band_id": controlled_band_id,
        "two_site_model_id": two_site_model_id,
        "maximum_bare_dispersion_difference": max_bare_dispersion_difference,
        "bare_dispersion_match_tolerance": tolerance,
        "bare_dispersion_matched": bool(max_bare_dispersion_difference <= tolerance),
        "readout_temperature": float(grid.temperature),
        "high_energy_window_min": float(readout_cfg["hubbard_min"]),
        "high_energy_window_max": float(readout_cfg["hubbard_max"]),
        "dense_fraction_count": int(len(dense_fractions)),
        "controlled_high_minus_low_positive_all_dense_fractions": bool((control_delta > 0.0).all()),
        "dmft_high_minus_low_negative_all_dense_fractions": bool((dmft_delta < 0.0).all()),
        "opposite_sign_all_dense_fractions": bool((control_delta > 0.0).all() and (dmft_delta < 0.0).all()),
        "controlled_high_minus_low_min_dense": float(control_delta.min()),
        "controlled_high_minus_low_max_dense": float(control_delta.max()),
        "dmft_high_minus_low_min_dense": float(dmft_delta.min()),
        "dmft_high_minus_low_max_dense": float(dmft_delta.max()),
        "main_fraction_deltas": [
            {
                "candidate": str(row.candidate),
                "fraction": float(row.fraction),
                "high_minus_low": float(row.high_minus_low),
            }
            for row in main_rows.itertuples()
        ],
        "claim_boundary": "opposite relative ordering under matched observable definitions; agreement is not unique microscopic reconstruction",
    }
    summary_path = PATH_DISCRIMINATION / "matched_candidate_summary.json"
    write_json(summary_path, summary)
    return [curve_path, marker_path, summary_path]


TWO_SITE_DMFT_STEPWISE = TWO_SITE_DMFT / "_stepwise"


def _two_site_dmft_step_path(group: str, case_id: str, kind: str) -> Path:
    safe = str(case_id).replace(".", "p").replace("-", "m")
    return TWO_SITE_DMFT_STEPWISE / f"{group}__{safe}__{kind}.csv"


def _remove_two_site_dmft_step_group(group: str) -> None:
    if not TWO_SITE_DMFT_STEPWISE.exists():
        return
    for path in TWO_SITE_DMFT_STEPWISE.glob(f"{group}__*.csv"):
        path.unlink()
    try:
        TWO_SITE_DMFT_STEPWISE.rmdir()
    except OSError:
        pass


def run_two_site_dmft_case(config: dict[str, Any], model_id: str) -> list[Path]:
    dmft_cfg = config["two_site_dmft"]
    model_data = next(item for item in dmft_cfg["models"] if item["id"] == model_id)
    result = _run_two_site_dmft_model_path(
        config,
        model_data,
        int(dmft_cfg["nk"]),
        int(dmft_cfg["nw"]),
        int(dmft_cfg["n_lambda"]),
    )
    paths = []
    for kind in ("solver", "curves", "markers"):
        path = _two_site_dmft_step_path("model", model_id, kind)
        write_csv(path, result[kind])
        paths.append(path)
    return paths


def merge_two_site_dmft(config: dict[str, Any]) -> list[Path]:
    combined = {kind: [] for kind in ("solver", "curves", "markers")}
    for model_data in config["two_site_dmft"]["models"]:
        model_id = str(model_data["id"])
        for kind in combined:
            path = _two_site_dmft_step_path("model", model_id, kind)
            if not path.is_file():
                raise FileNotFoundError(f"Missing two-site-DMFT model checkpoint: {path}")
            combined[kind].extend(pd.read_csv(path).to_dict(orient="records"))
    paths = {
        "solver": TWO_SITE_DMFT / "two_site_dmft_solver.csv",
        "curves": TWO_SITE_DMFT / "two_site_dmft_curves.csv",
        "markers": TWO_SITE_DMFT / "two_site_dmft_markers.csv",
    }
    for kind, path in paths.items():
        write_csv(path, combined[kind])
    snapshot_path = run_two_site_dmft_spectral_snapshots(config)
    _remove_two_site_dmft_step_group("model")
    return list(paths.values()) + [snapshot_path]


def run_two_site_dmft_convergence_case(config: dict[str, Any], case_index: int) -> Path:
    dmft_cfg = config["two_site_dmft"]
    audit_cfg = dmft_cfg["convergence"]
    grid_cfg = audit_cfg["grids"][int(case_index)]
    model_data = next(
        item for item in dmft_cfg["models"] if item["id"] == audit_cfg["model_id"]
    )
    print(f"  two-site DMFT convergence nk={grid_cfg['nk']} nw={grid_cfg['nw']}", flush=True)
    result = _run_two_site_dmft_model_path(
        config,
        model_data,
        int(grid_cfg["nk"]),
        int(grid_cfg["nw"]),
        int(audit_cfg["n_lambda"]),
    )
    rows = [
        {**row, "nk": int(grid_cfg["nk"]), "nw": int(grid_cfg["nw"])}
        for row in result["markers"]
    ]
    case_id = f"nk{grid_cfg['nk']}_nw{grid_cfg['nw']}"
    path = _two_site_dmft_step_path("convergence", case_id, "markers")
    write_csv(path, rows)
    return path


def merge_two_site_dmft_convergence(config: dict[str, Any]) -> Path:
    audit_cfg = config["two_site_dmft"]["convergence"]
    frames = []
    for grid_cfg in audit_cfg["grids"]:
        case_id = f"nk{grid_cfg['nk']}_nw{grid_cfg['nw']}"
        path = _two_site_dmft_step_path("convergence", case_id, "markers")
        if not path.is_file():
            raise FileNotFoundError(f"Missing two-site-DMFT convergence checkpoint: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    spread = (
        frame.groupby(["model", "readout", "fraction"])["direct_marker"]
        .agg(lambda values: float(values.max() - values.min()))
        .rename("marker_spread")
        .reset_index()
    )
    frame = frame.merge(spread, on=["model", "readout", "fraction"], how="left")
    frame["pass_two_site_dmft_convergence"] = (
        frame["marker_spread"] <= float(audit_cfg["pass_marker_spread"])
    )
    path = TWO_SITE_DMFT / "two_site_dmft_convergence.csv"
    frame.to_csv(path, index=False)
    _remove_two_site_dmft_step_group("convergence")
    return path


def run_two_site_dmft_broadening_case(config: dict[str, Any], case_index: int) -> Path:
    dmft_cfg = config["two_site_dmft"]
    audit_cfg = dmft_cfg["broadening_robustness"]
    broadening = float(audit_cfg["values"][int(case_index)])
    model_data = next(
        item for item in dmft_cfg["models"] if item["id"] == audit_cfg["model_id"]
    )
    print(f"  two-site DMFT broadening eta={broadening}", flush=True)
    local_config = dict(config)
    local_gate = dict(dmft_cfg)
    local_gate["lattice_broadening"] = broadening
    local_config["two_site_dmft"] = local_gate
    result = _run_two_site_dmft_model_path(
        local_config,
        model_data,
        int(audit_cfg["nk"]),
        int(audit_cfg["nw"]),
        int(audit_cfg["n_lambda"]),
    )
    rows = [{**row, "lattice_broadening": broadening} for row in result["markers"]]
    path = _two_site_dmft_step_path("broadening", f"eta{broadening:.6f}", "markers")
    write_csv(path, rows)
    return path


def merge_two_site_dmft_broadening(config: dict[str, Any]) -> Path:
    dmft_cfg = config["two_site_dmft"]
    audit_cfg = dmft_cfg["broadening_robustness"]
    frames = []
    for broadening in audit_cfg["values"]:
        path = _two_site_dmft_step_path("broadening", f"eta{float(broadening):.6f}", "markers")
        if not path.is_file():
            raise FileNotFoundError(f"Missing two-site-DMFT broadening checkpoint: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    readouts = list(dmft_cfg["readouts"])
    pivot = frame.pivot_table(
        index=["model", "fraction", "lattice_broadening"],
        columns="readout",
        values="direct_marker",
    ).reset_index()
    pivot["ordering"] = pivot.apply(
        lambda row: " < ".join(sorted(readouts, key=lambda name: float(row[name]))),
        axis=1,
    )
    pivot["ordering_stable"] = (
        pivot.groupby(["model", "fraction"])["ordering"].transform("nunique") == 1
    )
    path = TWO_SITE_DMFT / "two_site_dmft_broadening_robustness.csv"
    pivot.to_csv(path, index=False)
    _remove_two_site_dmft_step_group("broadening")
    return path


def run_two_site_dmft_high_energy_window_case(config: dict[str, Any], model_id: str) -> Path:
    dmft_cfg = config["two_site_dmft"]
    audit_cfg = dmft_cfg["high_energy_window_robustness"]
    model_data = next(item for item in dmft_cfg["models"] if item["id"] == model_id)
    fractions = [float(value) for value in config["markers"]["fractions"]]
    readouts = list(dmft_cfg["readouts"])
    rows: list[dict[str, Any]] = []

    grid = make_grid(
        int(audit_cfg["nk"]),
        int(audit_cfg["nw"]),
        float(dmft_cfg["wmax"]),
        float(dmft_cfg["readout_temperature"]),
    )
    band = make_band(model_data)
    second_moment = band_second_moment(band, int(dmft_cfg["moment_nk"]))
    critical_interaction = 6.0 * np.sqrt(second_moment)
    lambdas = np.linspace(0.0, 1.0, int(audit_cfg["n_lambda"]))
    u_ratios = float(dmft_cfg["u_min_ratio"]) + lambdas * (
        float(dmft_cfg["u_max_ratio"]) - float(dmft_cfg["u_min_ratio"])
    )
    interactions = critical_interaction * u_ratios
    epsilon = dispersion_square(grid.kx, grid.ky, band)
    band_edge = float(np.max(np.abs(epsilon - np.mean(epsilon))))

    window_specs: list[dict[str, Any]] = []
    for window in audit_cfg["windows"]:
        readout_cfg = dict(config["readout"])
        window_min = float(window["min_factor"]) * band_edge
        window_max = min(
            float(window["max_factor"]) * band_edge,
            0.95 * float(dmft_cfg["wmax"]),
        )
        if window_max <= window_min:
            raise ValueError(f"Invalid two-site-DMFT high-energy audit window: {window}")
        readout_cfg["hubbard_min"] = window_min
        readout_cfg["hubbard_max"] = window_max
        window_specs.append(
            {
                "window_id": str(window["id"]),
                "min_factor": float(window["min_factor"]),
                "max_factor": float(window["max_factor"]),
                "window_min": window_min,
                "window_max": window_max,
                "weights": prepare_readout_weights(grid, band, readout_cfg),
                "values": {name: [] for name in readouts},
            }
        )

    for index, interaction in enumerate(interactions):
        print(
            f"  two-site DMFT high-energy-window {band.band_id} "
            f"[{index + 1}/{len(interactions)}]",
            flush=True,
        )
        solution = solve_two_site_dmft(float(interaction), second_moment, dmft_cfg)
        sigma_w = two_site_dmft_self_energy_on_grid(grid.omega, solution, dmft_cfg)
        sigma_kw = np.broadcast_to(
            sigma_w[None, None, :], grid.kx.shape + (grid.omega.size,)
        )
        green = retarded_green_function(
            grid, band, sigma_kw, eta=float(dmft_cfg["lattice_broadening"])
        )
        spectral = np.maximum(spectral_function_raw(green), 0.0)
        for spec in window_specs:
            values = scalar_readouts(grid, spectral, spec["weights"])
            for readout in readouts:
                spec["values"][readout].append(float(values[readout]))

    for spec in window_specs:
        for readout in readouts:
            values = np.asarray(spec["values"][readout], dtype=float)
            diagnostics = marker_diagnostics(
                values,
                lambdas,
                float(config["markers"]["minimum_relative_change"]),
                float(config["markers"].get("minimum_monotonicity_score", 0.85)),
                float(config["markers"].get("monotonicity_tolerance", 1e-10)),
            )
            for fraction in fractions:
                rows.append(
                    {
                        "model": band.band_id,
                        "window_id": spec["window_id"],
                        "min_band_edge_factor": spec["min_factor"],
                        "max_band_edge_factor": spec["max_factor"],
                        "window_min": spec["window_min"],
                        "window_max": spec["window_max"],
                        "readout": readout,
                        "fraction": fraction,
                        "marker": float(completion_marker(values, lambdas, fraction)),
                        **diagnostics,
                    }
                )
    path = _two_site_dmft_step_path("high_energy", model_id, "markers")
    write_csv(path, rows)
    release_memory()
    return path


def merge_two_site_dmft_high_energy_window(config: dict[str, Any]) -> Path:
    dmft_cfg = config["two_site_dmft"]
    frames = []
    for model_data in dmft_cfg["models"]:
        path = _two_site_dmft_step_path("high_energy", str(model_data["id"]), "markers")
        if not path.is_file():
            raise FileNotFoundError(f"Missing two-site-DMFT high-energy checkpoint: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    readouts = list(dmft_cfg["readouts"])
    pivot = frame.pivot_table(
        index=[
            "model",
            "window_id",
            "min_band_edge_factor",
            "max_band_edge_factor",
            "fraction",
        ],
        columns="readout",
        values="marker",
    ).reset_index()
    baseline_order = "transport_quadratic < high_energy_spectral < low_energy_spectral"
    pivot["ordering"] = pivot.apply(
        lambda row: " < ".join(sorted(readouts, key=lambda name: float(row[name]))),
        axis=1,
    )
    pivot["baseline_full_order"] = baseline_order
    pivot["full_three_stage_order_pass"] = pivot["ordering"] == baseline_order
    pivot["quadratic_before_high_energy"] = (
        pivot["transport_quadratic"] < pivot["high_energy_spectral"]
    )
    pivot["high_energy_before_low_energy"] = (
        pivot["high_energy_spectral"] < pivot["low_energy_spectral"]
    )
    pivot["quadratic_to_high_energy_margin"] = (
        pivot["high_energy_spectral"] - pivot["transport_quadratic"]
    )
    pivot["high_energy_to_low_energy_margin"] = (
        pivot["low_energy_spectral"] - pivot["high_energy_spectral"]
    )
    pivot["downstream_order_stable_across_windows"] = (
        pivot.groupby(["model", "fraction"])["high_energy_before_low_energy"]
        .transform("min")
        .astype(bool)
    )
    pivot["full_order_stable_across_windows"] = (
        pivot.groupby(["model", "fraction"])["full_three_stage_order_pass"]
        .transform("min")
        .astype(bool)
    )
    path = TWO_SITE_DMFT / "two_site_dmft_high_energy_window_robustness.csv"
    write_csv(path, pivot.to_dict(orient="records"))
    _remove_two_site_dmft_step_group("high_energy")
    return path

def run_path_discrimination(config: dict[str, Any]) -> list[Path]:
    """Generate the matched controlled-vs-DMFT self-energy-evolution comparison outputs.

    The comparison stage is intentionally restricted to the two self-energy evolutions used by the current manuscript: a controlled pole trajectory and the two-site-DMFT evolution under matched bare dispersion and observable definitions.
    """
    path_cfg = config.get("path_discrimination", {})
    if not path_cfg.get("enabled", False):
        return []
    dense_fractions = np.asarray(path_cfg["dense_fractions"], dtype=float)
    matched_paths = _run_matched_candidate_comparison(config, dense_fractions)

    markers = pd.read_csv(PATH_DISCRIMINATION / "matched_candidate_markers_dense.csv")
    curves = pd.read_csv(PATH_DISCRIMINATION / "matched_candidate_curves.csv")

    layer_map = {"controlled": "controlled", "DMFT": "two_site_microscopic"}
    markers["layer"] = markers["candidate"].map(layer_map)
    curves["layer"] = curves["candidate"].map(layer_map)
    marker_path = PATH_DISCRIMINATION / "cross_model_markers_dense.csv"
    write_csv(marker_path, markers.to_dict("records"))

    contrast_rows = []
    for candidate, group in markers.groupby("candidate"):
        pivot = group.pivot_table(index="fraction", columns="readout", values="direct_marker")
        for fraction, row in pivot.iterrows():
            available = [(name, float(row[name])) for name in pivot.columns if np.isfinite(row[name])]
            ordering = " < ".join(name for name, _ in sorted(available, key=lambda item: item[1]))
            contrast_rows.append({
                "candidate": str(candidate),
                "fraction": float(fraction),
                "ordering": ordering,
                "high_minus_low": float(row["high_energy_spectral"] - row["low_energy_spectral"]),
            })
    contrast_path = PATH_DISCRIMINATION / "downstream_order_contrast.csv"
    write_csv(contrast_path, contrast_rows)

    summary_matched = json.loads((PATH_DISCRIMINATION / "matched_candidate_summary.json").read_text())
    summary = {
        "stage": "self-energy-evolution comparison",
        "scope": "matched controlled-vs-two-site-DMFT self-energy-evolution comparison",
        "candidate_paths": ["controlled", "DMFT"],
        "dense_fraction_count": int(len(dense_fractions)),
        "bare_dispersion_matched": bool(summary_matched["bare_dispersion_matched"]),
        "opposite_sign_all_dense_fractions": bool(summary_matched["opposite_sign_all_dense_fractions"]),
        "claim_boundary": "Relative ordering can reject an evolution that predicts the opposite sign under matched observable definitions; agreement does not uniquely reconstruct a microscopic path.",
        "discrimination_pass": bool(summary_matched["bare_dispersion_matched"] and summary_matched["opposite_sign_all_dense_fractions"]),
    }
    summary_path = PATH_DISCRIMINATION / "path_discrimination_summary.json"
    write_json(summary_path, summary)
    return [*matched_paths, marker_path, contrast_path, summary_path]




def run_phase(phase: str, config: dict[str, Any]) -> None:
    if phase == "generate_pole_kinematics":
        from pole_family import generate as generate_pole_family
        generate_pole_family(config)
    elif phase == "generate_five_bath_ed_dmft":
        from five_bath_ed import generate as generate_five_bath
        generate_five_bath(config)
    elif phase == "two_site_square":
        run_two_site_dmft_case(config, "two_site_square")
    elif phase == "two_site_anisotropic":
        run_two_site_dmft_case(config, "two_site_anisotropic")
    elif phase == "two_site_merge":
        merge_two_site_dmft(config)
    elif phase == "two_site_convergence_0":
        run_two_site_dmft_convergence_case(config, 0)
    elif phase == "two_site_convergence_1":
        run_two_site_dmft_convergence_case(config, 1)
    elif phase == "two_site_convergence_merge":
        merge_two_site_dmft_convergence(config)
    elif phase == "two_site_broadening_0":
        run_two_site_dmft_broadening_case(config, 0)
    elif phase == "two_site_broadening_1":
        run_two_site_dmft_broadening_case(config, 1)
    elif phase == "two_site_broadening_2":
        run_two_site_dmft_broadening_case(config, 2)
    elif phase == "two_site_broadening_merge":
        merge_two_site_dmft_broadening(config)
    elif phase == "two_site_high_energy_square":
        run_two_site_dmft_high_energy_window_case(config, "two_site_square")
    elif phase == "two_site_high_energy_anisotropic":
        run_two_site_dmft_high_energy_window_case(config, "two_site_anisotropic")
    elif phase == "two_site_high_energy_merge":
        merge_two_site_dmft_high_energy_window(config)
    elif phase == "path_discrimination":
        run_path_discrimination(config)
    elif phase == "summaries":
        summarize_two_site_dmft(config, [])
        if (TWO_SITE_DMFT / "two_site_dmft_curves.csv").exists():
            run_path_discrimination(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
