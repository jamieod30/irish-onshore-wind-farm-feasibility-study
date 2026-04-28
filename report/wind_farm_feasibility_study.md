# Wind Farm Feasibility Assessment for a Proposed Onshore Development in Ireland

## Executive Summary

This report presents a feasibility-stage assessment for a proposed onshore wind farm in Ireland using an independently developed geospatial and wake-modelling workflow. The project was designed to replicate key elements of a professional early-stage development process where access to commercial software was limited. Public datasets, GIS screening, Python automation, and wake-loss modelling were integrated to identify a preferred region, screen developable land, generate candidate layouts, and compare turbine configurations.

Five candidate regions were first ranked using ERA5-derived wind resource indicators. Galway emerged as the strongest overall site based on mean wind speed, power density, winter performance, directional consistency, and turbulence proxy metrics. A detailed GIS constraints assessment was then undertaken within the selected region to identify feasible land parcels.

Following constraint screening, 214 individual parcels remained. Parcels were classified by developable area and prioritised using proximity to roads and separation from buildings. Parcel fid_7 was identified as the preferred primary candidate. Layout testing showed that a constrained 10-turbine arrangement within fid_7 provided the best single-parcel balance of energy yield and wake performance. Additional nearby parcels were then assessed to recover the original 12-turbine benchmark scale. A distributed 12-turbine multi-parcel layout (candidate d) achieved the highest net annual energy production among tested expansion options.

The study demonstrates not only engineering feasibility methods, but also the ability to design a defensible analytical pipeline, adapt methodology to tool constraints, and translate technical outputs into practical development decisions.

## Project Scope

The project currently covers:

* wind resource screening using ERA5 reanalysis data
* multi-criteria ranking of candidate regions
* GIS-based land suitability and constraints filtering
* parcel prioritisation and developable area assessment
* turbine layout generation and refinement
* wake-loss and annual energy production modelling using PyWake
* comparative feasibility assessment of shortlisted layouts

Planned future extensions include grid integration screening, storage options assessment, and techno-economic evaluation.

## 1. Introduction

Early-stage wind farm development requires the integration of resource assessment, land constraints analysis, engineering judgement, and commercial awareness. In practice, many of these tasks are performed using specialist commercial platforms. This study explores how a robust feasibility workflow can be developed using accessible tools while maintaining transparent engineering logic.

The objective was to identify a credible onshore wind development opportunity in Ireland and refine it into a realistic turbine layout supported by quantitative evidence.

## 2. Methodology

## 2.1 Regional Wind Resource Screening

Five candidate Irish regions were screened using ERA5-based wind metrics:

* Galway
  n- Donegal
* Mayo
* Midlands
* Kerry

Metrics included mean wind speed at 100 m, power density, Weibull parameters, winter wind speed, directional consistency, shear exponent, and turbulence proxy. Weighted scoring was used to rank candidate regions.

## 2.2 GIS Constraint Screening

The preferred region was subjected to geospatial filtering using exclusion layers representing development constraints. These included proximity to buildings, transport corridors, environmental limitations, water features, and terrain-related restrictions where applicable.

Remaining suitable land was dissolved into individual parcels for further screening.

## 2.3 Parcel Prioritisation

A two-stage parcel ranking system was applied:

**Viability class by area**

* Class A: >2.5 km²
* Class B: >1.0 km²
* Class C: ≤1.0 km²

**Priority score (0–1 nominal scale)**

* greater distance from buildings improved score
* closer proximity to roads improved score

## 2.4 Layout Development and Wake Modelling

Candidate turbine layouts were developed using the selected parcel geometry and tested using the PyWake framework with a representative Vestas V136-4.2 MW turbine model.

The initial benchmark concept assumed 12 turbines. Constrained layouts containing between 5 and 12 turbines were compared to identify the most effective feasible arrangement. Where the preferred parcel could not efficiently host 12 turbines, nearby parcels were assessed for expansion options.

## 3. Results

## 3.1 Regional Site Selection

| Rank | Region   | Weighted Score |
| ---- | -------- | -------------- |
| 1    | Galway   | 3.836          |
| 2    | Donegal  | 2.797          |
| 3    | Mayo     | 2.774          |
| 4    | Midlands | 2.700          |
| 5    | Kerry    | 2.291          |

Galway ranked first and was selected for detailed development screening.

## 3.2 Parcel Screening Results

After constraints filtering, 214 parcels remained.

Top-ranked parcels included:

| Parcel | Area (km²) | Class | Priority Score |
| ------ | ---------- | ----- | -------------- |
| fid_7  | 3.545      | A     | 0.921          |
| fid_63 | 2.683      | A     | 0.627          |
| fid_81 | 3.906      | A     | 0.592          |
| fid_18 | 1.527      | B     | 0.659          |
| fid_28 | 1.674      | B     | 0.537          |

Although some smaller parcels scored highly on access/separation metrics, fid_7 offered the strongest combined balance of scale, contiguity, and development practicality.

## 3.3 Primary Parcel Layout Optimisation

Testing of constrained single-parcel layouts found that a 10-turbine configuration delivered the best outcome within fid_7. Adding further turbines within the same parcel materially increased wake losses and reduced overall layout efficiency.

Selected 10-turbine case:

* Installed capacity: 42.0 MW
* Net AEP: 109.74 GWh
* Wake loss: 5.22%
* Net capacity factor: 29.83%

## 3.4 Multi-Parcel Expansion to 12 Turbines

To maintain comparison with the original 12-turbine benchmark, adjacent smaller parcels around fid_7 were screened for two additional turbine positions. Five feasible 12-turbine combinations (A–E) were tested.

Best result: **candidate d**

* Installed capacity: 50.4 MW
* Net AEP: 131.61 GWh
* Wake loss: 5.27%
* Net capacity factor: 29.81%

This was the strongest-performing constrained 12-turbine option.

## 3.5 Benchmark Comparison

Compared with the original unconstrained 12-turbine benchmark:

* Net AEP change: -0.65 GWh (-0.49%)
* Wake loss increase: +0.46 percentage points
* Installed capacity: unchanged at 50.4 MW

This indicates that the refined constrained layout preserved almost all benchmark energy yield while adapting to realistic land constraints.

## 4. Discussion

Several project outcomes are notable.

First, wind resource quality alone did not determine the final development concept. Land geometry and constraints materially affected feasible turbine count and placement.

Second, the highest-performing practical solution was not simply the largest parcel filled to maximum density. The preferred result combined an efficient primary parcel layout with selective expansion onto nearby land.

Third, transparent custom workflows can provide strong feasibility-stage insights when commercial tools are unavailable, provided assumptions and limitations are clearly stated.

## 5. Limitations

This study is intended as a feasibility-stage desk assessment and does not replace full development studies.

Key limitations include:

* ERA5 reanalysis data used in place of on-site mast or LiDAR measurements
* representative turbine power/thrust curves used for modelling
* simplified optimisation relative to dedicated commercial layout software
* no detailed electrical design or grid connection study
* no geotechnical, peat, ecological, or planning consent investigations
* no financial model or LCOE assessment at current stage

## 6. Conclusions

The developed workflow successfully progressed from national-scale screening to a defensible preferred layout concept.

Key conclusions are:

* Galway was the strongest candidate region among those assessed
* GIS screening reduced the search area to 214 feasible parcels
* fid_7 was the best primary parcel for development
* the most effective single-parcel layout contained 10 turbines
* a distributed 12-turbine expansion (candidate d) best matched benchmark project scale
* the final constrained concept retained nearly all benchmark net energy yield

The project demonstrates competence in geospatial analysis, data-driven engineering judgement, Python workflow development, wake modelling, and communication of practical feasibility outcomes.

## 7. Future Work

Planned next steps include:

* grid connection and network constraint screening
  n- battery storage integration options
* capital cost and operating cost estimation
* revenue and sensitivity modelling
* techno-economic comparison of final development pathways
* environmental and consenting risk review
