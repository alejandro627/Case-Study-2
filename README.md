# Case Study 2 – Resilience-oriented design of the Transfer Line CPPS

Python implementation of Case Study 2 of the doctoral thesis "Sustainability-oriented optimization of modular and reconfigurable cyber-physical production systems: A Complexity-engineering-based framework for resilience". The case study applies a
complexity-engineering-based design framework to the Montrac Transfer Line 3.0 of the Smart Mini Factory (SMF)
laboratory: a reconfigurable cyber-physical production system (CPPS) with a shuttle transport system, a shared UR10
robot, a UR3e assembly robot, a FANUC assembly cell and two manual workstations.

The repository contains the models, configuration data, results and tests of the four steps of the framework and of
the test with the physical demonstrator:

| Step | Content | Code |
|---|---|---|
| 1. AD complexity framework | functional decomposition (documented in the thesis); defines the design variables | – |
| 2. Uncertain system modelling | discrete-event simulation (DES v10.0), six-discipline MDA (MDA 1.5), deterministic sensitivity analysis and MDF optimisation, uncertainty model UDES 2.1 | `des/`, `disciplines/`, `mda/`, `sensitivity/`, `optimisation/`, `uncertainty/`, `udes_2_0/` |
| 3. UMDA | uncertainty propagation, acceptance of the resilience objective, sample sizes, design screening, reference-configuration study | `umda_2_1/`, `umda11/` |
| 4. UMDO | multi-fidelity NSGA-II with chance constraints, verification and high-fidelity evaluation | `umdo_2_1/` |
| Physical validation | replay of the measured test logs (three test phases) in the DES | `validation_physical/` |

## Model overview

* **Production process.** One shuttle cycle produces one batch of 9 products:
  WS6 (UR10 loads two trays) → WS9 (workpiece setup) → WS4 (UR3e assembly) → WS6 (UR10 transfers the FANUC tray) →
  WS8 (inspection, UR10 dispatch) → WS6. The FANUC branch runs in parallel. Process and transport times are means of
  20 measured repetitions (`cs2_mdo/config/baseline.json`).
* **Disciplines.** D1 transportation, D2 material handling, D3 assembly, D4 quality and dispatch, D5 workpiece setup,
  D6 coordination, coupled through event-based interfaces in one SimPy environment.
* **Design variables (131,220 configurations).** DV1 number of shuttles; DV2–DV5 and DV7 manual, collaborative or
  automated alternatives of the loading, setup, assembly (WS4 and FANUC) and dispatch operations; DV6 quality-control
  assistance; DV8 inspection policy; DV9 energy-management policy; DV10 maintenance policy. Design identifiers are written
  as `Z` followed by the ten levels, e.g. `Z3111331241`.
* **Uncertainty.** Restoration times of the UR10, the shuttle, the UR3e and the FANUC cell after a breakdown
  (truncated normal distributions scaled by design-dependent factors K_j(z)); 16 disruption scenarios DS0–DS15.
* **Adaptation mechanisms.** Manual fallback, ramp-up after restart and backlog catch-up by a temporary speed-up.
* **Objectives.** Maximise E[TH_d] (throughput under disruption), minimise E[E_d] (energy per 8-h shift) and maximise
  E[R_phi(t_f)] (time-dependent resilience), subject to chance constraints on recovery, throughput and energy.

## Repository structure

```
cs2_mdo/
  config/                 model inputs (process, energy, design variables, calibration, uncertainty model)
  des/                    DES core; transfer_line_des_v10_0.py is a single-file version of the DES
  disciplines/            D1-D6
  mda/                    coordinator, design variables, alternatives, couplings, constraints C1-C6, responses
  sensitivity/            OFAT, full factorial, Sobol and interaction indices
  optimisation/           exact Pareto front and NSGA-II (deterministic MDF)
  uncertainty/            breakdown model, restoration-time sampling, scenarios (UDES 1.1)
  udes_2_0/               DV10, adaptation mechanisms, R_phi(t_f) (UDES 2.0 / 2.1)
  umda11/                 trajectory store and time-dependent resilience metrics
  umda_2_1/               UMDA (Step 3)
  umdo_2_1/               UMDO (Step 4)
  validation_physical/    physical validation
  tests/                  test suite (run_tests.py)
  run_*.py                command-line entry points
  results/                results of all steps (CSV, JSON, NPZ, figures) and the raw logs of the physical test
```

## Installation

Python 3.10 or later (developed with Python 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

All commands are run from the repository root. Parallel steps use `--workers` processes (default 10).

## Reproducing the results

The `results/` folder already contains the outputs of every step, so each step can be inspected or continued
without re-running the previous ones. Running a step again overwrites its outputs.

### Step 2 – deterministic model, sensitivity analysis and MDF

```bash
python -B cs2_mdo/des/transfer_line_des_v10_0.py --shuttles 1 --no-write   # single-file DES, baseline
python -B cs2_mdo/run_mda.py                                                # MDA, baseline design, 8 h
python -B cs2_mdo/run_mda.py --dv DV1=2 --dv DV5=1 --hours 4               # other design and horizon
python -B cs2_mdo/run_gantt.py                                              # Gantt charts, 1-3 shuttles

python -B cs2_mdo/run_sensitivity.py ofat                                   # one-factor-at-a-time
python -B cs2_mdo/run_sensitivity.py factorial --workers 10                 # all 26,244 nine-DV designs (~30 min)
python -B cs2_mdo/run_sensitivity.py indices                                # Sobol and interaction indices
python -B cs2_mdo/run_sensitivity.py plots

python -B cs2_mdo/run_mdf.py exact                                          # exact Pareto front (factorial data)
python -B cs2_mdo/run_mdf.py nsga2 --seeds 1 2 3 4 5                        # NSGA-II with direct DES evaluation
python -B cs2_mdo/run_mdf.py compare
python -B cs2_mdo/run_mdf.py representatives
python -B cs2_mdo/run_mdf.py plots
```

### Step 3 – UMDA

```bash
python -B cs2_mdo/run_umda21.py plan         # number of runs per stage
python -B cs2_mdo/run_umda21.py run          # stages A, C and S, 32,370 DES runs (~1 h with 10 workers, resumable)
python -B cs2_mdo/run_umda21.py reference    # reference configurations, DS1-DS4 and DS15, N = 500 (11,600 runs)
python -B cs2_mdo/run_umda21.py report       # tables (CSV) and figures
```

### Step 4 – UMDO

```bash
python -B cs2_mdo/run_umdo21.py search --pop 60 --gen 40 --seed 1   # NSGA-II, N = 50 per scenario (~8 h, resumable)
python -B cs2_mdo/run_umdo21.py verify                              # archive front at N = 100 (independent samples)
python -B cs2_mdo/run_umdo21.py hifi                                # representatives at N = 500
python -B cs2_mdo/run_umdo21.py report                              # tables (CSV) and figures
python -B -m cs2_mdo.umdo_2_1.front_stats                           # statistics of the verified front by DV level
python -B -m cs2_mdo.umdo_2_1.not_improved_all                      # representatives with K = 1 (not improved)
```

### Physical validation

```bash
python -B -m cs2_mdo.validation_physical.analysis
```

Inputs: the task logs and event notes of the three test phases in
`cs2_mdo/results/Validation_physical_demonstrator/` (Phase 1 nominal operation, Phase 2 disruptions without resilience
actions, Phase 3 disruptions with resilience actions). Outputs: `.../Validation_physical_demonstrator/analysis/`.

### Tests

```bash
python -B -m cs2_mdo.tests.run_tests
```

The suite checks the production process, the couplings and constraints, the energy model, the design-variable table
and alternatives, the sensitivity analysis, the optimisation, the uncertainty model and the adaptation mechanisms;
the results are written to `cs2_mdo/results/T_test_results.csv`.

## Main result files

| File | Content |
|---|---|
| `results/sensitivity/factorial_results.csv` | deterministic responses of all 26,244 nine-DV designs |
| `results/optimisation/exact_pareto_designs.csv` | exact deterministic Pareto front |
| `results/umda_2_1/UMDA2_1_runs.csv` | all UMDA runs (one row per design, scenario and sample) |
| `results/umda_2_1/UMDA2_1_ref_table_*.csv` | uncertainty propagation of the reference configurations |
| `results/umdo_2_1/UMDO2_1_search_archive.csv` | every design evaluated by the UMDO search |
| `results/umdo_2_1/UMDO2_1_verified_front.csv` | verified Pareto front (N = 100) |
| `results/umdo_2_1/UMDO2_1_representatives.csv` | representative designs (max throughput, min energy, max resilience, balanced) |
| `results/umdo_2_1/UMDO2_1_hifi_runs.csv` | high-fidelity evaluation of the representatives (N = 500) |
| `results/Validation_physical_demonstrator/analysis/V_table_*.csv` | physical validation tables |

## Notes

* Measured process and transport times are deterministic inputs; the restoration-time distributions, the
  design-dependent restoration factors and the parameters of the adaptation mechanisms are engineering assumptions.
* `UDES2_CONFIG=udes_2_0_config.json` selects the UDES 2.0 settings (catch-up ends on the product count, R_phi at
  restoration + 60 min); the default is UDES 2.1.
* The DES is deterministic for a given design, scenario and restoration-time sample; LHS samples are generated with
  fixed seeds, so all runs are reproducible.

## License and citation

The code is released under the MIT License (see `LICENSE`).

If you use this code, please cite: Humberto Alejandro Barrero-Arciniegas, "Sustainability-oriented optimization of modular and reconfigurable cyber-physical production systems: A Complexity-engineering-based framework for resilience", doctoral thesis, Free University of Bozen-Bolzano, 2026.
