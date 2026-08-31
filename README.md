# Stochastic intrusion/extrusion, hysteresis, and nucleation in ZIF-71 (RHO) and ZIF-67 (SOD)

## Overview

This deposit contains Python simulation code, processed numerical results, analysis notebooks, and publication-ready figures for stochastic wetting/dewetting and nucleation in two zeolitic imidazolate framework (ZIF) topologies:

- **ZIF-71**, represented by the **RHO** topology;
- **ZIF-67**, represented by the **SOD** topology.

The repository is divided into two complementary studies:

1. **Hysteresis cycles** — stochastic intrusion/extrusion simulations under cyclic pressure and analysis of filling curves and hysteresis metrics.
2. **Nucleation (Sear analysis)** — pressure sweeps of first-nucleation times fitted with a stretched-exponential survival model.

## Repository contents

```text
ZENODO_repository/
├── README.md
├── Hysteresis_cycles/
│   ├── ZIF-71(RHO)/
│   │   └── RHO_sim.py
│   ├── ZIF-67(SOD)/
│   │   └── SOD_sim.py
│   └── Analysis/
│       ├── analyse_RHO_SOD_results.ipynb
│       ├── RHO/                         # replicate-level RHO filling data
│       ├── SOD/                         # replicate-level SOD filling data
│       ├── RHO_analysis_replicates.csv
│       ├── RHO_analysis_summary.csv
│       ├── SOD_analysis_replicates.csv
│       ├── SOD_analysis_summary.csv
│       └── figures/                     # hysteresis-loop and metric figures
└── nucleation/
    ├── ZIF-71(RHO)/
    │   └── zif71_sear.py
    ├── ZIF-67(SOD)/
    │   └── zif67_sear.py
    └── Analysis/
        ├── RHO/                         # ZIF-71 pressure-sweep results
        ├── SOD/                         # ZIF-67 pressure-sweep results
        ├── sear_pressure_sweep_analysis.ipynb
        ├── sear_pressure_sweep_enriched.csv
        ├── sear_pressure_trend_statistics.csv
        ├── sear_regime_counts.csv
        ├── sear_flagged_results.csv
        └── *.png                        # pressure, fit, and survival figures
```

## Software requirements

The code is written in Python and uses:

- Python 3.9 or newer;
- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- Jupyter Notebook or JupyterLab for the analysis notebooks.

Install the required packages in an isolated environment, for example:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install numpy pandas scipy matplotlib jupyter
```

The simulation scripts can be computationally demanding. Runtime and memory use depend on the topology, network size, pressure-grid resolution, number of replicates, and available CPU cores.

## Reproducing the hysteresis simulations

Run commands from the `ZENODO_repository` directory so that relative output paths are easy to locate.

### ZIF-71 (RHO)

```bash
python "Hysteresis_cycles/ZIF-71(RHO)/RHO_sim.py" \
  --seed 42 \
  --n-runs 10 \
  --output-dir "Hysteresis_cycles/Analysis/RHO_reproduced"
```

`--n-runs` accepts values from 1 to 26. The default base seed is 42. A one-at-a-time sensitivity case can be run with:

```bash
python "Hysteresis_cycles/ZIF-71(RHO)/RHO_sim.py" \
  --sensitivity \
  --case-id 0 \
  --seed 42 \
  --output-dir "Hysteresis_cycles/Analysis/RHO_sensitivity"
```

### ZIF-67 (SOD)

```bash
python "Hysteresis_cycles/ZIF-67(SOD)/SOD_sim.py" \
  --seed 42 \
  --output-dir "Hysteresis_cycles/Analysis/SOD_reproduced"
```

A sensitivity case can be selected with `--sensitivity --case-id N`. The number of baseline simulations and growth levels are configured in the `num_simulations` and `growth_steps_list` variables near the main block of `SOD_sim.py`.

### Hysteresis analysis

Open and run:

```text
Hysteresis_cycles/Analysis/analyse_RHO_SOD_results.ipynb
```

The notebook reads the replicate CSV files, calculates summary metrics, and generates the figures stored under `Hysteresis_cycles/Analysis/figures/`.

## Reproducing the nucleation calculations

The nucleation scripts perform multiple stochastic realizations at each pressure and fit the survival probability with the stretched-exponential model. It is based on the method of Sear (2013) (dx.doi.org/10.1021/cg301849f | Cryst. Growth Des. 2013, 13, 1329−1333).

```text
S(t) = exp[-(t / tau)^beta],
```

where `tau` is the characteristic first-nucleation time and `beta` is the shape parameter. 

Run the simulations with:

```bash
python "nucleation/ZIF-71(RHO)/zif71_sear.py"
python "nucleation/ZIF-67(SOD)/zif67_sear.py"
```

The pressure ranges and other numerical settings are defined in the main block near the end of each script. The deposited processed sweeps contain:

- ZIF-71 (RHO): 26–31 MPa;
- ZIF-67 (SOD): 24–29 MPa.

The simulations use multiprocessing and may take substantial time. Each nucleation script currently constructs seeds from the execution time; consequently, a fresh run is statistically comparable but is not guaranteed to reproduce every deposited value bit-for-bit.

### Nucleation analysis

Open and run:

```text
nucleation/Analysis/sear_pressure_sweep_analysis.ipynb
```

The notebook automatically locates the two deposited sweep CSV files, validates their schema and regime labels, calculates derived Weibull quantities, summarizes pressure trends, and exports:

- `sear_pressure_sweep_enriched.csv`;
- `sear_pressure_trend_statistics.csv`;
- `sear_regime_counts.csv`;
- `sear_flagged_results.csv`;
- `sear_pressure_sweep_overview.png`;
- `sear_fit_diagnostics.png`;
- `sear_reconstructed_survival.png`.

The survival curves in the analysis figure are reconstructed from the fitted `beta` and `tau` values.

## Data conventions

- **Pressure:** MPa.
- **Filling/state variables:** dimensionless fractions unless stated otherwise in a table.
- **`nucleation_rate`:** fraction of stochastic runs that nucleated within the simulation horizon; it is not an event frequency per unit time.
- **`n_nodes`:** number of nodes in the simulated network.
- **`n_nucleated`:** number of realizations that nucleated.
- **`r2` and `rmse`:** goodness-of-fit diagnostics for the stretched-exponential survival fit.
- **Random seeds:** hysteresis outputs record or derive seeds from a user-selectable base seed; the nucleation scripts use time-derived seeds.

CSV files are comma-separated, include a header row, and can be read with standard scientific-computing or spreadsheet software.

## Scope and interpretation

The deposited RHO and SOD simulations use different topology definitions and network sizes. Direct differences between ZIF-71 and ZIF-67 should therefore not be attributed solely to topology without accounting for model parameters and system size.

Each nucleation pressure has one aggregated fitted result. Trend statistics in the notebook are descriptive; uncertainty in the fitted parameters cannot be reconstructed from the aggregate tables alone.

## Citation

If you use this code or data, please, cite the publication: > [Johnson et al. ([2026]). How Crystallite Size and Topology Control Intrusion–Extrusion Hysteresis in Metal–Organic Frameworks. [ACS Applied Materials & Interfaces]. [DOI]]

**Associated publication**

> [Liam J. W. Johnson, Daniel Moreno-Rodríguez, Eder Amayuelas, Luis Bartolomé, Francisco Bonilla, German Gómez, Juan-Miguel López del Amo, Gabriel A. López, Alberto Giacomello, and Yaroslav Grosu]. ([2026]). [How Crystallite Size and Topology Control Intrusion–Extrusion Hysteresis in Metal–Organic Frameworks]. *[ACS Applied Materials & Interfaces]*. [DOI]

## Contact

For questions about the simulations or data, contact:

> [Daniel Moreno-Rodriguez] — [daniel.morenorodriguez@uniroma1.it]



