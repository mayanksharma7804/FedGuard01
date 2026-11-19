# Chapter 6 - Results and discussion

> DRAFT (19 Nov 2026) - all experiments finished (144 / 144 queue runs, Findings F07-F58). All numbers are test-split values, mean +/- standard
> deviation over seeds 42, 43, 44, unless stated otherwise; each traces to `results/master.csv` or to the
> summary tables named in the text.

## 6.1 The dataset after leakage control (O5)

Removing identifier columns turned 95.5% of the 2.39 million NF-UNSW-NB15-v2 flows into exact duplicates of
other flows (F07); one of the "features", `DNS_QUERY_ID`, was itself an identifier that hid ~131k duplicate
benign flows (F08), and every attack flow came from 4 of the 40 source addresses (F09). After de-duplication
104,022 unique flows remain, with tiny rare classes (Worms: 36 training rows) (F12). A leakage test caught one
test row that had collided with a training row through float32 rounding, which was fixed before any model was
trained (F11). **Consequence:** our numbers come from a protocol in which a model cannot score by memorising
hosts or duplicated rows; they are lower than, and not directly comparable with, results reported on the raw
dataset (F13). Figures: `eda_duplicates.png`, `eda_class_counts.png`.

## 6.2 Centralised ceiling and the cost of federation (O1)

| Setting | Macro-F1 | Accuracy | Binary F1 | Benign FPR |
|---|---|---|---|---|
| B0 centralised | **0.597 +/- 0.004** | 0.898 | 0.912 | 0.068 |
| B1 FedAvg, K = 5, alpha = 100 (IID) | **0.603 +/- 0.005** | 0.896 | 0.911 | 0.074 |
| B1 FedAvg, alpha = 0.5 | **0.537 +/- 0.076** | 0.888 | 0.906 | 0.057 |
| B1 FedAvg, alpha = 0.1 | **0.435 +/- 0.028** | 0.873 | 0.894 | 0.067 |

(F22, F29; `results/b0_centralised/summary.csv`, `results/b1_fedavg/summary_by_alpha.csv`; figures
`b0_centralised_*.png`, `b1_fedavg_*.png`.)

- **Federation itself costs nothing** with IID clients: FedAvg reaches the centralised ceiling.
- **Non-IID data costs up to 0.16 macro-F1**, but almost nothing in binary attack-vs-benign quality (0.894 vs
  0.912). What is lost is *which* attack it is: per-class recall falls most for rare classes held by few
  clients (Worms 0.30 -> 0.03, Backdoor 0.36 -> 0.13 from alpha 100 to 0.1); in one partition a small client
  holding 241 of the 247 Reconnaissance rows has 6.6% of the FedAvg weight, and Reconnaissance recall is 0 (F29).
- The accuracy of ~0.90 is close to the 91-93% of ref [1]; macro-F1 shows the rare-class weakness accuracy hides.
- The most dangerous error is attack-as-benign: 47% of Analysis and 21% of Backdoor flows are missed (F23).
- The spread between partitions at alpha = 0.5 (+/- 0.076) is large, which is why every method is always compared
  on the same partitions and per-seed differences are reported.

## 6.3 The cost of privacy (O2, foundation)

| epsilon (per client, delta = 1/N) | Macro-F1 | Binary F1 | Benign FPR | Attack detection |
|---|---|---|---|---|
| no DP (same 40-round budget) | **0.513 +/- 0.056** | 0.904 | 0.058 | 0.919 |
| 8 | **0.253 +/- 0.036** | 0.851 | 0.095 | 0.880 |
| 3 | **0.219 +/- 0.029** | 0.853 | 0.114 | 0.911 |
| 1 | **0.183 +/- 0.028** | 0.800 | 0.110 | 0.812 |

(B2: DP-FedAvg, K = 5, alpha 0.5, clip C = 2, F38; `results/b2_dpfedavg/summary_by_epsilon.csv`, figures
`b2_dpfedavg_*.png`.)

Record-level DP roughly halves macro-F1 even at epsilon 8. Under DP the model keeps Benign (~0.89 recall),
Exploits (0.86-0.94) and partly Fuzzers, while Analysis, Backdoor, DoS, Generic, Reconnaissance, Shellcode and
Worms fall to ~0 recall; attack-vs-benign detection mostly survives. The reason is a signal-to-noise limit: with
1-19k rows per client and 40k parameters, each round's DP noise (length ~13 at epsilon 3) is about ten times the
clean update (F33, F35), and rare classes have only tens of rows per client, so their signal drowns first. The
paper's suggestion that noise can act as a regulariser (its section 3.3) was not observed: stronger privacy was
never better. **Consequence for the robustness experiments:** aggregators are compared on models that mainly
separate Benign / Exploits / Fuzzers, so binary F1 and benign FPR are reported next to macro-F1.

## 6.4 The noise radius can be predicted from declared settings (O3 foundation, E0)

FedGuard's premise is that the server can predict each client's DP noise from what the client declares.
Experiment E0 trained the same client twice from the same model with the same batches and dropout - once with
noise multiplier sigma and once with sigma = 0 - and compared the measured noise ||delta_noisy - delta_clean||
with the declared radius r = lr * sigma * C * sqrt(T * d) / B.

| Start model | sigma | Predicted r | Measured | Ratio (mean +/- std) | Pairs |
|---|---|---|---|---|---|
| initial | 0.5 / 1 / 2 | 2.00 / 3.99 / 7.99 | 2.01 / 4.02 / 8.02 | 1.004-1.005 +/- 0.005 | 75 |
| after 5 FedAvg rounds | 0.5 / 1 / 2 | 2.00 / 3.99 / 7.99 | 2.00 / 3.99 / 7.99 | 1.000 +/- 0.004-0.005 | 75 |

(F35; `results/e0_calibration/e0_table.csv`, figure `e0_calibration.png`.)

The prediction is correct to within about 1%, independent of sigma and of the number of local steps (40-375),
with a spread close to the theoretical wobble of a Gaussian noise length (1/sqrt(2d) = 0.0035); no correction
factor is needed (kappa = 1). The cosine between the noise and the clean update is -0.002 to -0.008, i.e. the
noise is orthogonal to the signal, which justifies FedGuard's quadrature rule. **E0 was not a formality:** its
first run found that the declared radius was 5-20% too small for small clients, because Opacus samples batches
with rate 1/ceil(n/B) and divides by an integer expected batch size, not by B (F34). An honest small client would
have looked suspicious to FedGuard; the fix was made and regression-tested before any DP baseline was run.

## 6.5 The conflict, measured: attacks and AutoGM under DP (O2, O4)

**Attacks break the plain average.** K = 10 clients, alpha 0.5, 30% attackers (who also run DP-SGD and declare an
honest-looking sigma), 40-round budget, test macro-F1:

| Aggregator | Privacy | none | label flip | sign flip | gaussian |
|---|---|---|---|---|---|
| FedAvg | no DP | 0.370 +/- 0.015 | 0.303 +/- 0.019 | 0.264 +/- 0.036 | 0.338 +/- 0.019 |
| FedAvg | eps 3 | 0.218 +/- 0.006 | 0.173 +/- 0.036 | 0.163 +/- 0.015 | 0.145 +/- 0.012 |
| AutoGM (lam_scale 4) | eps 3 | 0.210 +/- 0.022 | 0.123 +/- 0.039 | 0.152 +/- 0.002 | 0.206 +/- 0.009 |
| FedGuard (frozen, F50) | eps 3 | 0.203 +/- 0.013 | 0.126 +/- 0.041 | 0.150 +/- 0.001 | 0.212 +/- 0.005 |

(`results/e5_attacks/summary.csv`, figure `e5_attacks.png`; discussed in 6.7.) Without DP every attack costs FedAvg
0.03-0.11 macro-F1; with DP the model is already weak and gaussian attackers hurt most (0.218 -> 0.145).

**AutoGM has no good setting under DP (F39).** Tuned on the validation split (epsilon 3, alpha 0.5, seed 42):

| lambda_scale | HRR, no attack | Attackers' weight, 30% sign flip | Mean validation macro-F1 |
|---|---|---|---|
| 0.5 / 1 | **0.90** (weights one-hot: the model follows one client) | 0.00 / 0.015 | 0.048 |
| 2 | 0.10 | **0.365** | 0.128 |
| **4 (chosen)** | 0.00 | **0.325** | **0.178** |
| 8 | 0.00 | 0.310 | 0.174 |

With a small lambda AutoGM collapses onto a single client, whose full DP noise becomes the global update; with a
large lambda it keeps every honest client but the sign-flip attackers keep about a third of the weight - more than
their fair share of 0.30. A sign-flipped DP update is barely further from the centre than an honest one: its extra
displacement (~1-1.5) is small next to the noise radius (~13) and the +/-10% spread of radii between honest clients.
This is the paper's Gap 1 measured on real IDS training: no lambda both keeps the honest private clients and
excludes the attackers. In the main table, AutoGM gives sign-flip and label-flip attackers 0.33 of the weight,
while FedAvg gives them 0.21 (their share of the data).

**Under mixed privacy the most private clients are silenced (F47).** With epsilon 1, 3 and 8 assigned in turn to
the ten clients, AutoGM gave each of the four epsilon-1 clients weight 0 in every round, and the epsilon-8 clients
0.23-0.24 each - the hospital-and-bank story of the paper's introduction; confirmed on the 40-round runs with three
seeds in 6.7.

## 6.6 Choosing FedGuard's settings on validation (O3)

FedGuard's options were chosen exactly like AutoGM's lambda: on the **validation** split, seed 42, epsilon 3,
alpha 0.5, by the mean validation macro-F1 over no attack and 30% sign flip - a rule fixed before any result was
seen (F50; `results/p7_fedguard_tune/selection_val.csv`).

| Setting (one change at a time) | Mean val macro-F1 | Honest rejection, no attack | Attackers' weight, sign flip |
|---|---|---|---|
| **quadrature + R4 + s floor, lambda_scale 4 (chosen)** | **0.1780** | **0.005** | 0.289 |
| linear, lambda_scale 1 (the paper's linear rule) | 0.1779 | 0.200 | 0.261 |
| linear + R4 + s floor, lambda_scale 4 | 0.1752 | 0.005 | 0.278 |
| quadrature, lambda_scale 1 (the guide's default) | 0.1742 | 0.115 | 0.287 |
| quadrature + R4, lambda_scale 1 | 0.1697 | 0.020 | 0.271 |
| (three more variants) | 0.1676-0.1749 | 0.005-0.020 | 0.266-0.283 |

The choice rests on a near-tie in macro-F1 (0.1780 vs 0.1779) and a large difference in fairness: without the s
floor (F48) FedGuard sidelines 11.5-20% of honest clients on clean data, with it 0.5%. These settings were frozen
(`configs/fedguard_frozen.yaml`) for every result below and for the live demo.

## 6.7 Main comparison: FedGuard vs AutoGM vs FedAvg (O3, O4)

**One shared epsilon of 3 (E5, table in 6.5).** FedGuard and AutoGM are statistically tied on every attack: the
paired per-seed differences (same partition, seed and attackers) are -0.018 to +0.011 macro-F1
(`results/e5_attacks/paired_fedguard_vs_autogm.csv`). Both give gaussian attackers weight 0 and recover from that
attack (0.21 vs 0.145 for FedAvg). Against sign and label flip both give the attackers ~0.30-0.32 of the weight and
neither beats FedAvg - the attack's extra displacement (~1-1.5) is buried in DP noise of length ~13, as F39 predicted.
With one shared epsilon every client declares about the same noise (F36), so FedGuard has nothing extra to work with:
here it is a safe drop-in for AutoGM, not an improvement (F54).

**Mixed privacy - the paper's scenario (E5m, F55).** Epsilon 1, 3 and 8 assigned in turn to the ten clients:

| Aggregator | macro-F1 (no attack) | Honest rejection | Weight of an honest client at eps 1 / 3 / 8 | macro-F1 (30% sign flip) | Attackers' weight |
|---|---|---|---|---|---|
| FedAvg | 0.190 +/- 0.012 | 0.333 | 0.087 / 0.125 / 0.116 | **0.153 +/- 0.002** | 0.208 |
| AutoGM | 0.219 +/- 0.033 | **0.400** | **0.000** / 0.091 / 0.238 | 0.113 +/- 0.037 | 0.395 |
| **FedGuard** | 0.214 +/- 0.038 | **0.004** | **0.102 / 0.102 / 0.101** | 0.118 +/- 0.033 | 0.276 |

(`results/e5_mixed_privacy/`, figures `e5_mixed_privacy.png`, `weights_mixed_privacy_autogm.png`,
`weights_mixed_privacy_fedguard.png`.)

- **This is the central result.** AutoGM gives every epsilon-1 client weight 0 in every round of every seed and
  over-trusts the low-noise epsilon-8 clients (0.24 each): the most privacy-careful organisations are silenced. FedGuard
  gives every honest client almost exactly the fair share 1/K = 0.10, whatever its privacy level, with the same
  macro-F1 (0.214 vs 0.219; per seed 0.201 / 0.256 / 0.183 vs 0.217 / 0.253 / 0.186). Objectives O2 and O3 are met.
- **Where it stops:** with 30% sign-flip attackers under mixed privacy both robust aggregators do worse than plain
  FedAvg (0.118, 0.113 vs 0.153). Most of the gap is one seed (44), in which both collapse to predicting only Benign;
  in the other two seeds FedGuard scores 0.138 / 0.136 against FedAvg's 0.154 / 0.153 and AutoGM's 0.105 / 0.153.
  FedGuard gives the attackers less weight than AutoGM (0.28 vs 0.39) but more than FedAvg's data share (0.21).

## 6.8 Honest rejection across privacy and heterogeneity (O2, E4)

| Honest rejection (AutoGM / FedGuard) | no DP | eps 3 | eps 1 | mixed 1-3-8 |
|---|---|---|---|---|
| alpha 0.1 | 0.36 / 0.05 | 0.10 / 0.00 | 0.09 / 0.00 | 0.40 / 0.01 |
| alpha 0.5 | 0.36 / 0.06 | 0.00 / 0.00 | 0.00 / 0.00 | 0.40 / 0.01 |
| alpha 100 | 0.27 / 0.10 | 0.00 / 0.00 | 0.00 / 0.00 | 0.40 / 0.00 |

(no attackers, seeds 42 and 43; `results/e4_conflict/`, figure `e4_conflict.png`; F56.)

The conflict the paper describes is real on IDS data, and it has two sources: **unequal noise** (mixed privacy:
AutoGM 0.40 at every alpha) and **non-IID spread** (no DP: AutoGM 0.27-0.36; at alpha 0.1 its model collapses to
macro-F1 0.099 while FedGuard reaches 0.209). With one shared epsilon all clients carry the same noise and AutoGM
rejects almost nobody. FedGuard keeps honest rejection at or below 0.10 in all 12 cells. Macro-F1 of the two is within
+/- 0.02 in the DP cells, except mixed privacy at alpha 100, where AutoGM is better (0.263 vs 0.223). Two seeds per
cell: these are trends.

## 6.9 What each refinement contributes and where FedGuard breaks (E6, E7)

**Ablation, one change at a time from the frozen settings (3 seeds, F57; `e6_ablation.png`).** No single refinement
moves macro-F1 beyond the seed spread (0.196-0.203 without attack, 0.144-0.150 with sign flip). Removing R4 raises
honest rejection to 0.033, removing the s floor to 0.013-0.032; without R1 (the centre-noise correction) every
residual is zero and FedGuard degenerates to equal weights (attackers get exactly their 0.30). At a shared epsilon
the refinements buy fairness, not accuracy.

**Stress tests (seed 42, F58; `e7_stress.png`).** Attackers' weight at 10 / 20 / 30 / 40% sign flip: AutoGM
0.115 / 0.232 / 0.325 / 0.406, FedGuard 0.095 / 0.174 / 0.289 / 0.364 - lower at every share. At 40% attackers every
method breaks (macro-F1 0.080, only Benign predicted). FedGuard's advantage against attackers grows as the noise
shrinks: at epsilon 8 with 30% sign flip it reaches 0.170 (attackers 0.248) against AutoGM's 0.110 (0.313); at
epsilon 1 all methods are at ~0.13. **Breaking points:** 40% attackers, and epsilon <= 3 for seeing a sign-flip attack.

**Summary of the evidence against the paper's objectives.**

| Objective | Answer |
|---|---|
| O1 baselines | Federation costs nothing with IID data; non-IID costs up to 0.16 macro-F1, mostly rare classes (6.2) |
| O2 the conflict | Measured for the first time on IDS data: AutoGM sidelines 27-40% of honest clients under non-IID or mixed privacy (6.8) |
| O3 the noise-aware aggregator | The radius is predictable to ~1% (6.4); FedGuard gives every honest client its fair share at equal accuracy (6.7) |
| O4 robustness under DP + non-IID + attacks | Gaussian attacks stopped; sign / label flip not stopped by any distance-based rule under DP at epsilon <= 3; FedGuard limits attackers more than AutoGM, most at epsilon 8 (6.7, 6.9) |
| O5 honest reporting | Leakage-controlled data, macro-F1 + per-class recall + FPR + HRR, every negative result reported, invalid runs replaced (F46) |

## 6.10 Engineering results worth reporting

- Engine correctness: the Flower parity check found a bug in which clients trained sequentially instead of being
  averaged (F24); after the fix our engine and Flower agree to 6.9e-17 (F26). AutoGM equals the Blades reference
  to 1e-11 (F37).
- Early stopping must be checked against the learning curves: an under-trained comparison picked the wrong class
  weights (F18), patience 8 stopped FedAvg in temporary dips (F28), and a 20-round / patience-5 setting stopped
  private runs during their warm-up, invalidating 129 runs that were then re-run (F46).
- Running unattended on a laptop needs failure-safe engineering: a charger fault hung the GPU driver for hours
  (F41) and a thermal guard that died left training waiting (F51); per-round checkpoints meant no result was lost.
