# Self-energy pole evolution and crossover ordering

Reproducibility code for **“Self-energy pole evolution shapes the ordering of low- and high-energy spectral changes near the Mott transition.”** The repository regenerates the numerical results and Main/Supplementary figures from the declared equations, configuration, and the finite-bath primitive input deck.

Repository URL: `https://github.com/kano-koichi/mott-crossover-pole-kinematics`. The repository is intended to remain private during peer review and to become public upon acceptance.

## Requirements

Python 3.12 or newer with:

```bash
python -m pip install -r requirements.txt
```

The validated environment used Python 3.13 with NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, SciPy 1.17.0, and Numba 0.65.1.

## Reproduce the results

From the repository root, the canonical command is:

```bash
python run.py --fresh
```

This clears the generated `results/` tree, runs all 18 checkpoint-aware numerical phases, verifies the result manifest, and then rebuilds all publication figures.

Useful partial commands are:

```bash
python run.py analysis --fresh       # numerical pipeline only
python run.py figures                # figures from existing results
python run.py analysis --continue-run
python run.py analysis --step        # one verified numerical phase
```

No precomputed spectrum, self-energy, observable, or crossover-marker table is used as a calculation input. The added final phase regenerates the self-consistent physical-path derivative decomposition and 5-bath alternative readout checks before publication figures are built.

## Primitive-input boundary

The controlled pole family, two-site-DMFT calculations, robustness checks, and matched electronic-evolution comparison calculations are regenerated from equations and `inputs/config.json`.

`inputs/five_bath_ed_dmft/bath_fit_checkpoint.csv` is the declared primitive numerical input for the five-bath validation. It contains nine particle-hole-symmetric bath parameter sets for `U/t = 0,...,8` plus fit/self-consistency diagnostics. All rows satisfy the stored bath-parameter convergence flag; this flag refers to convergence of the finite-bath parameter iteration and does not imply a vanishing DMFT self-consistency residual. The residuals are nonzero (~0.0382–0.0402 in the stored input diagnostic), explicitly reported in Supplementary Fig. S5. The deck was not newly refitted to zero DMFT self-consistency residual as part of this revision. The code reconstructs the finite Anderson impurity Hamiltonian and regenerates the Green function, self-energy, lattice spectrum, observables, markers, and robustness outputs from this deck.

No downloaded experimental or third-party numerical dataset is required.

## Repository layout

- `run.py` — single canonical execution entrypoint and checkpoint controller.
- `inputs/config.json` — numerical and observable configuration.
- `inputs/five_bath_ed_dmft/bath_fit_checkpoint.csv` — declared five-bath primitive input.
- `src/common.py` — shared model, Green-function, observable, and I/O numerics.
- `src/pole_family.py` — controlled self-energy pole-family calculation.
- `src/five_bath_ed.py` — five-bath Hamiltonian/ED reconstruction.
- `src/pipeline.py` — two-site DMFT, robustness, summaries, and the matched electronic-evolution comparison.
- `src/figures.py` — Main Figs. 1–4 and Supplementary Figs. S1–S7.
- `results/` — generated machine-readable outputs, receipt, and manifest when present.
- `figures/` — generated publication PDFs when present.

The clean/submission snapshot intentionally omits regenerable `results/` and `figures/`; the full local snapshot retains the validated generated outputs.

## Receipt and result-manifest boundary

`results/manifest.json` hashes the immutable generated CSV/JSON result artifacts, excluding `results/run_receipt.json` and itself. The run receipt is finalized after result-manifest verification, so it is validated separately by its completion status and phase records. This prevents a self-referential receipt/manifest hash mismatch.

## Verification

A valid fresh run ends with `status = completed`, all 18 phases completed, no failed phase, `cache_reused = false`, `fallback_used = false`, and `manifest_verified = true` in `results/run_receipt.json`.

## License

Original code is released under the MIT License; see `LICENSE`.
