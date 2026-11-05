"""Shared numerical primitives for the Paper9 reproduction pipeline."""
from __future__ import annotations

# ---- I/O and reproducibility helpers ----
"""Configuration, output, manifest, and figure-QA utilities."""
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def clear_files(directory: Path, suffixes: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in sorted(files):
        entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"files": entries}


def release_memory() -> None:
    """Return large temporary NumPy allocations to the OS between model paths."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# ---- lattice and impurity models ----
"""Band structure, controlled self-energy, and two-site-DMFT models."""
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BandConfig:
    t: float
    tp: float
    tpp: float
    mu: float
    band_id: str = "band"
    tx: float | None = None
    ty: float | None = None


def dispersion_square(kx: np.ndarray, ky: np.ndarray, band: BandConfig) -> np.ndarray:
    tx = band.t if band.tx is None else band.tx
    ty = band.t if band.ty is None else band.ty
    return (
        -2.0 * (tx * np.cos(kx) + ty * np.cos(ky))
        - 4.0 * band.tp * np.cos(kx) * np.cos(ky)
        - 2.0 * band.tpp * (np.cos(2.0 * kx) + np.cos(2.0 * ky))
    )


def velocity2_square(kx: np.ndarray, ky: np.ndarray, band: BandConfig) -> np.ndarray:
    tx = band.t if band.tx is None else band.tx
    ty = band.t if band.ty is None else band.ty
    vx = (
        2.0 * tx * np.sin(kx)
        + 4.0 * band.tp * np.sin(kx) * np.cos(ky)
        + 4.0 * band.tpp * np.sin(2.0 * kx)
    )
    vy = (
        2.0 * ty * np.sin(ky)
        + 4.0 * band.tp * np.cos(kx) * np.sin(ky)
        + 4.0 * band.tpp * np.sin(2.0 * ky)
    )
    return vx * vx + vy * vy


def d_wave_form_factor(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return 0.5 * (np.cos(kx) - np.cos(ky))


def _broadcast_sigma(sigma_w: np.ndarray, shape_k: tuple[int, int]) -> np.ndarray:
    return np.broadcast_to(sigma_w, shape_k + (sigma_w.shape[-1],))


def sigma_controlled_mott_pole(
    omega: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    band: BandConfig,
    lam: float,
    params: dict[str, Any],
) -> np.ndarray:
    del band
    delta = float(params["delta_max"]) * float(lam)
    gamma_m = float(params["gamma_m"])
    gamma_bg = float(params["gamma_bg"])
    bath_width = float(params["bath_width"])
    w = omega[None, None, :]
    background = gamma_bg * bath_width / (w + 1j * bath_width)
    sigma_w = delta**2 / (w + 1j * gamma_m) + background
    return _broadcast_sigma(sigma_w, kx.shape)


MODEL_FUNCTIONS = {
    "controlled_mott_pole": sigma_controlled_mott_pole,
}


def make_band(data: dict[str, Any]) -> BandConfig:
    return BandConfig(
        t=float(data["t"]),
        tp=float(data["tp"]),
        tpp=float(data["tpp"]),
        mu=float(data["mu"]),
        band_id=str(data.get("id", "band")),
        tx=(None if data.get("tx") is None else float(data["tx"])),
        ty=(None if data.get("ty") is None else float(data["ty"])),
    )



# -----------------------------------------------------------------------------
# two-site DMFT: self-consistent two-site DMFT for the half-filled Hubbard model
# -----------------------------------------------------------------------------

_TWO_SITE_N_ORBITALS = 4
_TWO_SITE_DIMENSION = 1 << _TWO_SITE_N_ORBITALS


def _fermion_annihilation_matrix(orbital: int) -> np.ndarray:
    matrix = np.zeros((_TWO_SITE_DIMENSION, _TWO_SITE_DIMENSION), dtype=complex)
    for state in range(_TWO_SITE_DIMENSION):
        if (state >> orbital) & 1:
            occupied_below = (state & ((1 << orbital) - 1)).bit_count()
            sign = -1.0 if occupied_below % 2 else 1.0
            matrix[state ^ (1 << orbital), state] = sign
    return matrix


_TWO_SITE_ANNIHILATION = tuple(
    _fermion_annihilation_matrix(index)
    for index in range(_TWO_SITE_N_ORBITALS)
)
_TWO_SITE_CREATION = tuple(operator.conj().T for operator in _TWO_SITE_ANNIHILATION)
_TWO_SITE_NUMBER = tuple(
    _TWO_SITE_CREATION[index] @ _TWO_SITE_ANNIHILATION[index]
    for index in range(_TWO_SITE_N_ORBITALS)
)
_TWO_SITE_IDENTITY = np.eye(_TWO_SITE_DIMENSION, dtype=complex)


def _two_site_impurity_data(
    interaction: float,
    hybridization: float,
    beta: float,
) -> dict[str, Any]:
    """Diagonalize the particle-hole-symmetric two-site Anderson impurity.

    Orbital order is impurity up/down followed by bath up/down.  The
    interaction is written as U(n_up-1/2)(n_down-1/2), so the half-filled
    Hartree contribution is removed and the bath level is zero.
    """
    interaction = float(interaction)
    hybridization = float(hybridization)
    beta = float(beta)
    shifted_up = _TWO_SITE_NUMBER[0] - 0.5 * _TWO_SITE_IDENTITY
    shifted_down = _TWO_SITE_NUMBER[1] - 0.5 * _TWO_SITE_IDENTITY
    hamiltonian = interaction * (shifted_up @ shifted_down)
    for impurity, bath in ((0, 2), (1, 3)):
        hamiltonian = hamiltonian + hybridization * (
            _TWO_SITE_CREATION[impurity] @ _TWO_SITE_ANNIHILATION[bath]
            + _TWO_SITE_CREATION[bath] @ _TWO_SITE_ANNIHILATION[impurity]
        )

    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    boltzmann = np.exp(-beta * (energies - energies[0]))
    boltzmann /= np.sum(boltzmann)

    impurity_occupancy = 0.0
    for orbital in (0, 1):
        operator_eigenbasis = (
            eigenvectors.conj().T @ _TWO_SITE_NUMBER[orbital] @ eigenvectors
        )
        impurity_occupancy += float(
            np.sum(boltzmann * np.real(np.diag(operator_eigenbasis)))
        )
    double_operator = _TWO_SITE_NUMBER[0] @ _TWO_SITE_NUMBER[1]
    double_eigenbasis = eigenvectors.conj().T @ double_operator @ eigenvectors
    double_occupancy = float(
        np.sum(boltzmann * np.real(np.diag(double_eigenbasis)))
    )

    transition_weight = np.zeros(
        (_TWO_SITE_DIMENSION, _TWO_SITE_DIMENSION), dtype=float
    )
    for orbital in (0, 1):
        annihilation_eigenbasis = (
            eigenvectors.conj().T
            @ _TWO_SITE_ANNIHILATION[orbital]
            @ eigenvectors
        )
        transition_weight += 0.5 * (
            boltzmann[:, None] + boltzmann[None, :]
        ) * np.abs(annihilation_eigenbasis) ** 2

    return {
        "energies": energies,
        "transition_weight": transition_weight,
        "energy_difference": energies[:, None] - energies[None, :],
        "impurity_occupancy": impurity_occupancy,
        "double_occupancy": double_occupancy,
        "ground_state_energy": float(energies[0]),
    }


def _two_site_impurity_green(
    complex_frequency: np.ndarray,
    impurity_data: dict[str, Any],
) -> np.ndarray:
    z = np.atleast_1d(np.asarray(complex_frequency, dtype=complex))
    weight = np.asarray(impurity_data["transition_weight"], dtype=float)
    energy_difference = np.asarray(
        impurity_data["energy_difference"], dtype=float
    )
    return np.sum(
        weight[:, :, None]
        / (z[None, None, :] + energy_difference[:, :, None]),
        axis=(0, 1),
    )


def two_site_impurity_self_energy(
    complex_frequency: np.ndarray,
    interaction: float,
    hybridization: float,
    beta: float,
    impurity_data: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact two-site impurity self-energy at half filling."""
    z = np.atleast_1d(np.asarray(complex_frequency, dtype=complex))
    data = impurity_data or _two_site_impurity_data(
        interaction, hybridization, beta
    )
    impurity_green = _two_site_impurity_green(z, data)
    noninteracting_inverse = z - float(hybridization) ** 2 / z
    self_energy = noninteracting_inverse - 1.0 / impurity_green
    return self_energy, data


def two_site_quasiparticle_weight(
    interaction: float,
    hybridization: float,
    beta: float,
    imaginary_frequencies: np.ndarray,
) -> tuple[float, float]:
    """Estimate z from the low-frequency imaginary-axis self-energy."""
    frequencies = np.asarray(imaginary_frequencies, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("imaginary_frequencies must be positive")
    sigma, _ = two_site_impurity_self_energy(
        1j * frequencies,
        interaction,
        hybridization,
        beta,
    )
    slope = float(
        np.sum(frequencies * np.imag(sigma))
        / np.sum(frequencies * frequencies)
    )
    z_value = 1.0 / (1.0 - slope)
    return float(np.clip(z_value, 0.0, 1.0)), slope


def solve_two_site_dmft(
    interaction: float,
    second_moment: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Solve the ground-state-proxy half-filled two-site DMFT fixed point.

    The finite ``impurity_inverse_temperature`` is used only to project the
    exactly diagonalized impurity onto its ground-state limit. It is not
    identified with the separate observable-window temperature used later.

    For the particle-hole-symmetric one-bath-site problem, exact
    diagonalization gives the metallic fixed point
        z = 1 - U^2/(36 M2),   V^2 = z M2,
    with the insulating solution z=V=0 above Uc=6 sqrt(M2).  The ED
    low-frequency derivative is evaluated independently as a solver audit
    and is not fed back into the root selection near the critical point.
    """
    interaction = float(interaction)
    second_moment = float(second_moment)
    beta = float(params["impurity_inverse_temperature"])
    frequencies = np.asarray(params["z_imaginary_frequencies"], dtype=float)
    if interaction < 0.0 or second_moment <= 0.0:
        raise ValueError("interaction and second_moment must be physical")

    critical_interaction = 6.0 * math.sqrt(second_moment)
    self_consistent_z = max(
        0.0, 1.0 - (interaction / critical_interaction) ** 2
    )
    hybridization = math.sqrt(second_moment * self_consistent_z)
    branch = "metallic" if self_consistent_z > 0.0 else "insulating"
    data = _two_site_impurity_data(interaction, hybridization, beta)

    if branch == "metallic":
        numerical_z, estimator_slope = two_site_quasiparticle_weight(
            interaction, hybridization, beta, frequencies
        )
        numerical_error = numerical_z - self_consistent_z
    else:
        numerical_z = float("nan")
        estimator_slope = float("nan")
        numerical_error = float("nan")

    return {
        "interaction": interaction,
        "second_moment": second_moment,
        "critical_interaction": critical_interaction,
        "hybridization": float(hybridization),
        "quasiparticle_weight": float(self_consistent_z),
        "analytic_quasiparticle_weight": float(self_consistent_z),
        "numerical_quasiparticle_weight": float(numerical_z),
        "analytic_z_error": float(numerical_error),
        "self_consistency_residual": float(
            hybridization**2 - second_moment * self_consistent_z
        ),
        "iterations": 0,
        "converged": True,
        "branch": branch,
        "impurity_occupancy": float(data["impurity_occupancy"]),
        "double_occupancy": float(data["double_occupancy"]),
        "ground_state_energy": float(data["ground_state_energy"]),
        "z_estimator_slope": float(estimator_slope),
        "impurity_data": data,
    }

def two_site_dmft_self_energy_on_grid(
    omega: np.ndarray,
    solution: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    """Evaluate the converged two-site-DMFT self-energy on a real grid."""
    eta = float(params["spectral_broadening"])
    sigma, _ = two_site_impurity_self_energy(
        np.asarray(omega, dtype=float) + 1j * eta,
        float(solution["interaction"]),
        float(solution["hybridization"]),
        float(params["impurity_inverse_temperature"]),
        impurity_data=solution["impurity_data"],
    )
    return sigma


def band_second_moment(band: BandConfig, nk: int = 512) -> float:
    """Numerically evaluate the second moment of the bare dispersion."""
    k = np.linspace(-np.pi, np.pi, int(nk), endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    epsilon = dispersion_square(kx, ky, band)
    centered = epsilon - float(np.mean(epsilon))
    return float(np.mean(centered**2))





# ---- Green-function/observable numerics ----
"""Numerical grid, observable, marker, and physical-audit routines."""
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

@dataclass(frozen=True)
class Grid:
    kx: np.ndarray
    ky: np.ndarray
    omega: np.ndarray
    dk_weight: float
    dw: float
    temperature: float


@dataclass(frozen=True)
class ReadoutWeights:
    fermi_derivative: np.ndarray
    velocity2: np.ndarray
    nodal: np.ndarray
    antinodal: np.ndarray
    d_wave: np.ndarray
    hubbard_window: np.ndarray


READOUTS = [
    "transport_quadratic",
    "transport_linear",
    "low_energy_spectral",
    "antinodal_spectral",
    "nodal_spectral",
    "high_energy_spectral",
    "d_wave_spectral",
]


def make_grid(nk: int, nw: int, wmax: float, temperature: float) -> Grid:
    if nk < 2 or nw < 2 or wmax <= 0.0 or temperature <= 0.0:
        raise ValueError("Invalid grid parameters")
    k = np.linspace(-np.pi, np.pi, int(nk), endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    omega = np.linspace(-float(wmax), float(wmax), int(nw))
    return Grid(
        kx=kx,
        ky=ky,
        omega=omega,
        dk_weight=1.0 / float(nk * nk),
        dw=float(omega[1] - omega[0]),
        temperature=float(temperature),
    )


def fermi_function(omega: np.ndarray, temperature: float) -> np.ndarray:
    x = np.clip(omega / max(float(temperature), 1e-12), -80.0, 80.0)
    return 1.0 / (np.exp(x) + 1.0)


def minus_dfermi_domega(omega: np.ndarray, temperature: float) -> np.ndarray:
    x = np.clip(omega / (2.0 * max(float(temperature), 1e-12)), -80.0, 80.0)
    return 1.0 / (4.0 * temperature * np.cosh(x) ** 2)


def retarded_green_function(
    grid: Grid,
    band: BandConfig,
    sigma_kw: np.ndarray,
    eta: float = 1e-10,
) -> np.ndarray:
    epsilon = dispersion_square(grid.kx, grid.ky, band)
    return 1.0 / (
        grid.omega[None, None, :]
        + band.mu
        - epsilon[:, :, None]
        - sigma_kw
        + 1j * eta
    )


def spectral_function_raw(green_kw: np.ndarray) -> np.ndarray:
    return -np.imag(green_kw) / np.pi


def spectral_function(green_kw: np.ndarray) -> np.ndarray:
    raw = spectral_function_raw(green_kw)
    return np.maximum(raw, 0.0)


def _smooth_step(x: np.ndarray, x0: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - x0) / max(float(width), 1e-12)))


def prepare_readout_weights(
    grid: Grid,
    band: BandConfig,
    config: dict[str, Any],
    readout_temperature: float | None = None,
) -> ReadoutWeights:
    fd = d_wave_form_factor(grid.kx, grid.ky)
    fd_abs = np.abs(fd)
    nodal_width = float(config["nodal_width"])
    antinodal_center = float(config["antinodal_center"])
    omega_abs = np.abs(grid.omega)
    kernel_temperature = (
        grid.temperature if readout_temperature is None else float(readout_temperature)
    )
    if kernel_temperature <= 0.0:
        raise ValueError("readout_temperature must be positive")
    return ReadoutWeights(
        fermi_derivative=minus_dfermi_domega(grid.omega, kernel_temperature),
        velocity2=velocity2_square(grid.kx, grid.ky, band),
        nodal=1.0 / (1.0 + (fd_abs / nodal_width) ** 4),
        antinodal=_smooth_step(fd_abs, antinodal_center, 0.06),
        d_wave=fd**2,
        hubbard_window=(omega_abs >= float(config["hubbard_min"]))
        & (omega_abs <= float(config["hubbard_max"])),
    )


def scalar_readouts(
    grid: Grid,
    spectral_kw: np.ndarray,
    weights: ReadoutWeights,
) -> dict[str, float]:
    """Evaluate all scalar observables without allocating full 3D weight products."""
    prefactor = grid.dk_weight * grid.dw
    low_energy_map = np.einsum(
        "ijw,w->ij", spectral_kw, weights.fermi_derivative, optimize=False
    )
    quadratic_map = np.einsum(
        "ijw,ijw,w->ij",
        spectral_kw,
        spectral_kw,
        weights.fermi_derivative,
        optimize=False,
    )
    high_energy_map = np.einsum(
        "ijw,w->ij",
        spectral_kw,
        weights.hubbard_window.astype(float),
        optimize=False,
    )

    return {
        "transport_quadratic": float(
            np.sum(weights.velocity2 * quadratic_map) * prefactor
        ),
        "transport_linear": float(
            np.sum(weights.velocity2 * low_energy_map) * prefactor
        ),
        "low_energy_spectral": float(np.sum(low_energy_map) * prefactor),
        "antinodal_spectral": float(
            np.sum(weights.antinodal * low_energy_map) * prefactor
        ),
        "nodal_spectral": float(
            np.sum(weights.nodal * low_energy_map) * prefactor
        ),
        "high_energy_spectral": float(np.sum(high_energy_map) * prefactor),
        "d_wave_spectral": float(
            np.sum(weights.d_wave * low_energy_map) * prefactor
        ),
    }


def particle_density(grid: Grid, spectral_kw: np.ndarray) -> float:
    occupation = fermi_function(grid.omega, grid.temperature)[None, None, :]
    return float(2.0 * np.sum(occupation * spectral_kw) * grid.dk_weight * grid.dw)


def finite_window_sum_rule(grid: Grid, spectral_kw: np.ndarray) -> dict[str, float]:
    integrated = np.trapz(spectral_kw, grid.omega, axis=2)
    error = integrated - 1.0
    return {
        "sum_rule_mean": float(np.mean(integrated)),
        "sum_rule_min": float(np.min(integrated)),
        "sum_rule_max": float(np.max(integrated)),
        "sum_rule_abs_error_mean": float(np.mean(np.abs(error))),
        "sum_rule_abs_error_max": float(np.max(np.abs(error))),
    }


def physical_diagnostics(
    grid: Grid,
    spectral_kw: np.ndarray,
    raw_spectral_kw: np.ndarray,
    sigma_kw: np.ndarray,
) -> dict[str, float]:
    diagnostics = finite_window_sum_rule(grid, spectral_kw)
    diagnostics.update(
        {
            "density": particle_density(grid, spectral_kw),
            "A_raw_min": float(np.min(raw_spectral_kw)),
            "A_raw_negative_fraction": float(np.mean(raw_spectral_kw < -1e-12)),
            "im_sigma_max": float(np.max(np.imag(sigma_kw))),
            "im_sigma_positive_fraction": float(
                np.mean(np.imag(sigma_kw) > 1e-12)
            ),
        }
    )
    return diagnostics


def relative_change(series: np.ndarray) -> float:
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return float("nan")
    scale = max(abs(float(y[0])), float(np.max(np.abs(y))), 1e-15)
    return float((y[-1] - y[0]) / scale)


def completion_curve(series: np.ndarray) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if y.size < 2 or not np.all(np.isfinite(y)):
        return np.full_like(y, np.nan, dtype=float)
    denominator = float(y[-1] - y[0])
    if abs(denominator) < 1e-30:
        return np.full_like(y, np.nan, dtype=float)
    return (y - y[0]) / denominator


def completion_marker(
    series: np.ndarray,
    lambdas: np.ndarray,
    fraction: float,
) -> float:
    """Return the first-passage coordinate of an endpoint-normalized curve.

    For a nondecreasing completion curve this is the usual generalized inverse.
    For an audited weakly nonmonotone curve it is explicitly a first-passage
    coordinate; cumulative-dominance statements are not inferred from that case.
    Plateaus use the first coordinate at which the target is reached.
    """
    x = np.asarray(lambdas, dtype=float)
    c = completion_curve(series)
    finite = np.isfinite(x) & np.isfinite(c)
    if np.sum(finite) < 2:
        return float("nan")
    x = x[finite]
    c = c[finite]
    target = float(fraction)
    for index in range(len(x) - 1):
        c0, c1 = float(c[index]), float(c[index + 1])
        if c0 == target:
            return float(x[index])
        # The curve starts at zero and ends at one.  The first upward crossing
        # therefore implements the first-passage convention even when later
        # reversals occur.
        if c0 < target <= c1 and c1 != c0:
            return float(
                x[index]
                + (target - c0) / (c1 - c0) * (x[index + 1] - x[index])
            )
    return float("nan")


def marker_diagnostics(
    series: np.ndarray,
    lambdas: np.ndarray,
    minimum_relative_change: float,
    minimum_monotonicity_score: float = 0.85,
    monotonicity_tolerance: float = 1e-10,
) -> dict[str, Any]:
    y = np.asarray(series, dtype=float)
    x = np.asarray(lambdas, dtype=float)
    finite = np.isfinite(y) & np.isfinite(x)
    if np.sum(finite) < 2:
        return {
            "relative_change": float("nan"),
            "monotonicity_score": float("nan"),
            "reverse_step_count": 0,
            "maximum_reverse_completion_step": float("nan"),
            "monotonicity_class": "undefined",
            "marker_coordinate_type": "undefined",
            "validity": "excluded",
            "direction": "undefined",
        }
    y = y[finite]
    rel = relative_change(y)
    steps = np.diff(y)
    net = abs(float(y[-1] - y[0]))
    total = float(np.sum(np.abs(steps)))
    monotonicity = net / total if total > 1e-30 else 1.0
    completion = completion_curve(y)
    completion_steps = np.diff(completion)
    reverse = completion_steps < -abs(float(monotonicity_tolerance))
    reverse_step_count = int(np.sum(reverse))
    maximum_reverse_step = (
        float(np.max(-completion_steps[reverse])) if reverse_step_count else 0.0
    )
    is_nondecreasing = bool(reverse_step_count == 0)
    if y[-1] > y[0]:
        direction = "increasing"
    elif y[-1] < y[0]:
        direction = "decreasing"
    else:
        direction = "flat"
    if not np.isfinite(rel) or abs(rel) < minimum_relative_change:
        validity = "excluded"
    elif monotonicity < minimum_monotonicity_score:
        validity = "diagnostic"
    else:
        validity = "primary"
    return {
        "relative_change": float(rel),
        "monotonicity_score": float(monotonicity),
        "reverse_step_count": reverse_step_count,
        "maximum_reverse_completion_step": maximum_reverse_step,
        "monotonicity_class": "nondecreasing" if is_nondecreasing else "nonmonotone",
        "marker_coordinate_type": (
            "generalized_inverse" if is_nondecreasing else "first_passage"
        ),
        "validity": validity,
        "direction": direction,
    }


def solve_mu_for_density(
    target_density: float,
    mu_center: float,
    mu_half_width: float,
    steps: int,
    density_at_mu: Callable[[float], float],
) -> tuple[float, float]:
    lower = float(mu_center - mu_half_width)
    upper = float(mu_center + mu_half_width)
    density_lower = density_at_mu(lower)
    density_upper = density_at_mu(upper)
    if not density_lower <= target_density <= density_upper:
        raise RuntimeError(
            "Target density is not bracketed by the requested chemical-potential interval"
        )
    midpoint = mu_center
    density_mid = density_at_mu(midpoint)
    for _ in range(int(steps)):
        midpoint = 0.5 * (lower + upper)
        density_mid = density_at_mu(midpoint)
        if density_mid < target_density:
            lower = midpoint
        else:
            upper = midpoint
    return float(midpoint), float(density_mid)

