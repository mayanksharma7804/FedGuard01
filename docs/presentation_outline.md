# Presentation outline (12-15 slides, plan 13.5)

> DRAFT (19 Nov 2026), final results included (F50-F58). Keep each slide to one idea,
> one figure, at most ~25 words of text. Talk time ~12 minutes + demo 5 minutes.

| # | Title | On the slide | Figure / visual | Speaker notes (who says what) |
|---|---|---|---|---|
| 1 | FedGuard | Privacy-preserving and Byzantine-robust federated intrusion detection; team, guide, AKGEC, date | title | Names; one sentence: "we make two defences that fight each other work together". |
| 2 | One attack, two victims | Hospital Monday, bank Thursday, same ransomware kit; neither can name it alone; they cannot share logs | two-building sketch | The paper's story (section 1.1). Logs reveal patients, transactions, network maps - law forbids sharing. |
| 3 | Federated learning | Share the lesson, not the data: global model -> local training -> updates -> average | FL loop diagram | Recipe analogy (plan 2.2). One round = everyone trains locally, server averages. |
| 4 | Leaks and liars | Updates leak records (gradient inversion) -> differential privacy. Members can lie (poisoning) -> robust aggregation | two icons | DP = clip each record's influence + calibrated noise (epsilon). Robust = trust what is close to the crowd (AutoGM). |
| 5 | The conflict | Honest DP noise looks like an attack | plan Figure 2.1 style 2-D sketch; F39 table as a callout | No AutoGM lambda works under DP: small lambda follows one client (HRR 0.90), large lambda lets attackers keep 33% (F39). |
| 6 | The FedGuard idea | Clients declare (sigma, C, B, T, lr); server predicts noise radius r = lr sigma C sqrt(Td)/B; forgive r, judge the rest | formula + noise-length / 90-degree facts | High-dimensional noise: almost fixed length, almost orthogonal -> Pythagoras (quadrature rule). |
| 7 | Architecture | Client tier (CNN-LSTM + DP-SGD) / aggregation tier (FedGuard R1-R4 + s floor) / distribution | three-tier diagram (paper Fig. 5, updated) | R1 centre noise, R2 quadrature, R3 robust allowance, R4 noise wobble, s floor (F48). Pure NumPy; same code in our engine and in Flower. |
| 8 | Data without leakage | 2.39M flows -> 95.5% duplicates -> 104,022 unique; identifiers dropped (IPs, ports, DNS_QUERY_ID) | `eda_duplicates.png` | Why our scores are lower than published 91-93% - and honest (F07-F13). A test caught a float32 leak (F11). |
| 9 | Can the server predict the noise? | measured / predicted = 1.000-1.005 over 150 pairs; noise orthogonal to signal | `e0_calibration.png` | E0 also caught a 5-20% error in our first formula - Opacus' real batch size (F34). Foundation of the method. |
| 10 | Baselines and the cost of privacy | centralised 0.597; FedAvg 0.603 / 0.537 / 0.435 (alpha 100 / 0.5 / 0.1); DP eps 8 / 3 / 1: 0.25 / 0.22 / 0.18 | `b1_fedavg_by_alpha.png` + `b2_dpfedavg_by_epsilon.png` | Federation costs nothing with IID data; non-IID loses rare classes; DP roughly halves macro-F1 - rare attacks vanish (F38). |
| 11 | Who gets silenced? (O2, O3) | AutoGM: every eps-1 client weight 0 (HRR 0.40). FedGuard: every honest client 0.10 (HRR 0.004), same macro-F1 | `weights_mixed_privacy_autogm.png` vs `_fedguard.png`, `e5_mixed_privacy.png` | The paper's hospital story, measured over 3 seeds (F55). Across non-IID levels AutoGM sidelines 27-40%, FedGuard <= 10% (F56, `e4_conflict.png`). |
| 12 | Main attack table (O4) | Shared eps 3: FedGuard = AutoGM on every attack; gaussian stopped, sign / label flip not | `e5_attacks.png` | Paired differences -0.018..+0.011 (F54). Say why: the attack is buried in DP noise of length ~13 (F39). With mixed privacy + sign flip both robust aggregators fall below FedAvg (F55). |
| 13 | What matters and where it breaks | Refinements buy fairness, not accuracy; FedGuard limits attackers more than AutoGM; all break at 40% | `e6_ablation.png`, `e7_stress.png` | No R4 / no s floor: more honest clients sidelined (F57). Eps 8 + sign flip: FedGuard 0.170 vs AutoGM 0.110 (F58). |
| 14 | Live demo | 5 organisations on Flower; attacker red, eps-1 hospital blue; AutoGM then FedGuard | dashboard (`docs/demo.md`) | Fallback: replay `scripts/demo_dashboard.py --dir results/demo_fedguard`, or the recorded video. |
| 15 | Limitations and future work | honest declarations assumed; DP utility with small clients; one dataset; next: verify declared noise, client-level DP, CICIoT2023 | bullet list | Be the first to say the weaknesses (Chapter 7). Thank you + questions. |

## Backup slides (for questions)
- B1: per-class recall vs alpha (`b1_fedavg_per_class_recall.png`) - rare classes lost under non-IID (F29).
- B2: per-class recall vs epsilon (`b2_dpfedavg_per_class_recall.png`).
- Synthetic logic check (`synthetic_scenario_a/b.png`) - presented as a logic check only, never as a result.
- Engine parity with Flower (6.9e-17) and the bug it found (F24, F26).
- What went wrong and how we caught it: F24, F34, F46, F51 (shows scientific process).

## Rehearsal checklist
- Every member presents at least two slides and can answer any viva-pack question (`docs/viva_pack.md`).
- Demo rehearsed twice (`docs/demo.md`); backup video recorded; firewall rule and IPs written down.
