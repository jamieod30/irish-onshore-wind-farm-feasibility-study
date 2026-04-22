# Irish Onshore Wind Farm Feasibility Study

A reproducible renewable energy engineering project that demonstrates an early-stage onshore wind farm development workflow for Ireland, from national wind resource screening through parcel-level layout optimisation and wake-based performance comparison.

## Project Summary

This repository presents a preliminary feasibility study for a grid-connected onshore wind project in Ireland. The project is designed to reflect the staged workflow used by renewable energy developers during early project definition.

The study begins with regional wind resource screening using ERA5 reanalysis data, progresses through site selection and baseline layout development, applies GIS-based environmental and planning constraints, and then refines candidate wind farm layouts within screened developable parcels using PyWake-based wake modelling.

The project is structured as a reproducible engineering workflow with modular scripts, staged pipeline runners, tracked assumptions, and report-ready outputs.

## Project Objectives

The project was developed to demonstrate how an early-stage wind project can be assessed using a practical developer-style workflow. The key objectives are to:

- identify promising onshore wind development regions in Ireland using ERA5 data
- compare candidate sites using engineering-based wind resource metrics
- select a preferred regional site
- define and validate a baseline wind farm concept
- prepare wake model inputs and turbine performance definitions
- estimate baseline gross and net annual energy production
- screen the preferred site area using GIS-based environmental and planning constraints
- identify suitable land parcels for constrained development
- generate and compare parcel-based candidate layouts
- evaluate wake-loss and energy-yield trade-offs between candidate layouts
- compare selected constrained layouts against the original benchmark case
- provide outputs suitable for a professional engineering portfolio and public GitHub repository

## Workflow Overview

The repository is organised into staged pipeline sequences.

### Sequence 1 – Resource Assessment and Wake Inputs
This stage processes ERA5 data and prepares the wind resource basis for later modelling.

Scripts:
- `scripts/run_resource_assessment.py`
- `scripts/run_site_selection.py`
- `scripts/run_wake_inputs.py`
- `scripts/run_generate_turbine_curves.py`

Outputs include:
- processed ERA5 time series
- Weibull parameters
- directional frequency distributions
- site ranking tables
- wake input files
- turbine power and thrust curves
- report-ready resource figures and tables

### Sequence 2 – Baseline Benchmark Case
This stage runs the original unconstrained benchmark layout through PyWake and stores a fixed reference case for later comparison.

Scripts:
- `scripts/run_pywake_baseline.py`
- `scripts/run_build_baseline_performance_reference.py`

Outputs include:
- baseline farm summary
- turbine summary
- directional wake summary
- wake map
- baseline performance reference metadata

### Sequence 3 – Layout Quality Control and Candidate Spacing
This stage performs geometry and spacing checks on layouts.

Scripts:
- `scripts/run_layout_qc.py`
- `scripts/run_layout_spacing.py`

Outputs include:
- layout QC summary and failures
- nearest-neighbour spacing tables
- pairwise spacing tables
- candidate layout spacing summary

### Sequence 4 – Candidate Layout Wake Assessment and Selection
This stage runs all parcel candidate layouts for each case through PyWake and promotes the preferred layout based on the project selection logic.

Scripts:
- `scripts/run_case_candidate_pywake.py`
- `scripts/run_select_best_candidate_layout.py`

Outputs include:
- candidate wake summaries for each case
- per-layout PyWake results
- selected aligned layout for each case
- layout metadata

### Sequence 5 – Selected Layout Reassessment and Benchmark Comparison
This stage reruns the promoted selected layouts and compares them against the original benchmark.

Scripts:
- `scripts/run_selected_case_pywake.py`
- `scripts/run_selected_case_performance_comparison.py`
- `scripts/run_refined_layout_comparison.py`

Outputs include:
- selected-case wake summaries
- baseline versus selected performance comparisons
- layout change summaries
- case ranking tables
- metadata bundles

### Full Pipeline
The full sequence can be run with:

- `pipelines/run_full_pipeline.py`

This runs sequences 1 to 5 in order.

## Current Study Logic

The project uses two different layout concepts:

### Baseline Benchmark Layout
A 12-turbine unconstrained benchmark layout was first developed around the selected ERA5 screening location using nominal 5D × 7D spacing assumptions. This baseline serves as a fixed reference case for wake loss and annual energy production.

### Constrained Parcel Layouts
Following GIS screening, a preferred developable parcel (`fid_7`) was selected. Candidate layouts are then generated within this parcel and compared against the benchmark using spacing, wake loss, and net AEP metrics.

This distinction is important:
- the baseline is a resource-led benchmark
- the refined layouts are parcel-led feasible development cases

## Repository Structure

### `data/`
Stores raw, processed, and derived project inputs.

Key subfolders:
- `data/era5/` – ERA5 wind data and derived wind resource tables
- `data/constraints/` – GIS constraint layers and metadata
- `data/wake_inputs/` – prepared wind climate inputs for PyWake
- `data/turbines/` – turbine metadata and generated power/CT curves
- `data/optimisation/` – baseline, parcel candidate, selected layout, and comparison data

### `src/`
Reusable Python modules implementing the project logic.

Key modules:
- `src/resource_assessment/`
- `src/layout/`
- `src/wake/`
- `src/constraints/`

### `scripts/`
Thin stage runners for individual tasks.

### `pipelines/`
Stage-sequence runners that execute grouped parts of the workflow in order.

### `outputs/`
Generated project results including:
- figures
- report tables
- wake modelling outputs
- optimisation comparison outputs

### `report/`
Contains the main written feasibility study.

## Core Tools and Methods

The project uses a hybrid engineering toolchain:

- **Python** for data processing, automation, QA, wake modelling, and comparison logic
- **PyWake** for wake modelling and energy yield estimation
- **QGIS** for constraint mapping, parcel identification, and candidate layout development
- **ERA5** for preliminary wind resource assessment
- **Copernicus GLO-30** and public Irish spatial datasets for desk-study constraints

## Key Modelling Assumptions

This is a feasibility-stage project and includes several simplifying assumptions:

- wind resource based on ERA5 reanalysis rather than mast or LiDAR measurements
- no long-term correction or MCP procedure
- feasibility-stage turbine performance curves rather than OEM-certified curves
- NOJ engineering wake model
- constant turbulence intensity in the current wake setup
- desk-study GIS constraints rather than field-verified planning and environmental assessment
- no detailed geotechnical, peat, access-road, or landownership assessment

The outputs should therefore be interpreted as structured early-stage engineering estimates rather than development-ready project values.

## Selected Turbine

Current baseline turbine assumption:

- **Model:** Vestas V136-4.2 MW
- **Rated power:** 4.2 MW
- **Rotor diameter:** 136 m
- **Hub height:** 112 m

Turbine assumptions are stored centrally in:

- `data/turbines/vestas_v136_4p2mw/turbine_metadata.json`

## GIS Constraint Screening

The GIS stage identifies screened suitable land by excluding areas affected by:

- building setback constraints
- road setback constraints
- water setback constraints
- SAC and SPA ecological constraints
- slope exclusion

These layers are organised under:

- `data/constraints/`

with supporting metadata in:
- `data/constraints/metadata/constraint_assumptions.csv`
- `data/constraints/metadata/layer_catalogue.csv`

## Optimisation Cases

The current optimisation framework is designed to support:
- a benchmark baseline case
- one or more selected parcel cases
- multiple candidate layouts per parcel
- automated candidate wake assessment
- automated promotion of a preferred layout
- selected-case comparison back to the benchmark

The first constrained parcel case currently under development is:

- `fid_7`

## How to Run

### Run full project pipeline
python pipelines/run_full_pipeline.py

### Run individual sequences
python pipelines/run_pipeline_sequence_1.py
python pipelines/run_pipeline_sequence_2.py
python pipelines/run_pipeline_sequence_3.py
python pipelines/run_pipeline_sequence_4.py
python pipelines/run_pipeline_sequence_5.py

### Run individual scripts

python scripts/run_resource_assessment.py
python scripts/run_pywake_baseline.py
python scripts/run_layout_spacing.py
python scripts/run_case_candidate_pywake.py --all
python scripts/run_selected_case_pywake.py --all

### Output Highlights
outputs/wake/baseline_noj/
outputs/wake/optimisation/
outputs/wake/selected_cases/
outputs/tables/resource/
outputs/tables/layouts/
outputs/tables/optimisation/

### Why This Project Exists

This repository is intended as a portfolio-quality engineering project for renewable energy and wind industry roles. The emphasis is on:

- practical development workflow
- reproducible analysis
- clean project structure
- transparent assumptions
- technically defensible comparisons
- outputs suitable for engineering review and recruiter assessment

### Status

The project currently includes:

wind resource screening
site selection
baseline wake modelling
GIS constraint screening
parcel-based candidate layout optimisation framework
selected-layout comparison framework

### Further stages to include:

additional parcel development cases
electrical / collector system concepts
storage and grid integration analysis
techno-economic assessment
report refinement and portfolio visualisation