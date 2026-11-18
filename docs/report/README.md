# Final report - structure, sources and status

The report follows plan section 13.4. Chapters are drafted here in Markdown (version-controlled, numbers
traceable to `results/master.csv` through Findings.md) and converted to the college's Word format at the end.
Every chapter must be rewritten in the team's own words before submission, and AI assistance declared as the
college requires (plan decision D7).

| Ch. | Title | Source | Status |
|---|---|---|---|
| 1 | Introduction | paper sections 1.1-1.5; replace 1.5 as in `03_paper_updates.md` | to adapt from the paper |
| 2 | Literature review | paper sections 2-3 (25 studies, Tables I-IV) | to adapt from the paper |
| 3 | Problem statement and objectives | paper sections 4-6 with the corrections in `03_paper_updates.md` (radius formula, R1-R4 + s floor, K, evaluation plan, D1-D7) | corrections drafted |
| 4 | Methodology | `04_methodology.md` | draft done |
| 5 | Experimental setup | `05_experimental_setup.md` | draft done; status column to finalise |
| 6 | Results and discussion | `06_results.md` (from Findings, `results/figures/`, `results/*/summary.csv`) | 6.1-6.5 and 6.10 drafted; FedGuard sections 6.6-6.9 pending |
| 7 | Limitations and future work | `07_limitations.md` + paper section 7 | draft done; breaking points pending |
| App. | Reproducibility | README (setup, queue, plots), configs, GitHub link, `docs/demo.md` | - |

## Chapter 6 outline (results), mapped to the paper's objectives

| Objective | Evidence | Figure / table |
|---|---|---|
| O1 baselines | B0 ceiling (F22), B1 vs alpha (F29), comparison with ref [1] (F13) | `b0_centralised_*`, `b1_fedavg_*` |
| O2 the conflict, measured | B2 cost of privacy (F38); AutoGM lambda dilemma (F39); E4 HRR heatmaps; mixed privacy (F47) | `b2_dpfedavg_*`, `e4_conflict.png`, `e5_mixed_privacy.png` |
| O3 FedGuard design | E0 radius calibration (F35), synthetic logic check (F01, F02, F31, F48), unit tests, validation choice (F50), trust weights of one real run | `e0_calibration.png`, `synthetic_scenario_*.png`, dashboard image |
| O4 validation under DP + non-IID + attacks | E5 main table with paired differences, E6 ablation, E7 breaking points | `e5_attacks.png`, `e6_ablation.png`, `e7_stress.png` |
| O5 honest reporting | leakage protocol (F07-F13), metrics incl. HRR and per-class recall, where FedGuard lost, F46 (invalidated runs) | per-class recall tables |

Rule for every table: state seeds, epsilon/delta, alpha, K, attacker share, caps and dataset version (Appendix E).
