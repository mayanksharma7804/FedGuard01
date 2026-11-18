# Chapter 6 - Results and discussion

> DRAFT (18 Nov 2026). Sections 6.1-6.5 are final: they rest on completed experiments (Findings F07-F39 and the
> 40-round K = 10 FedAvg runs). Sections 6.6-6.9 (FedGuard) are filled in when the validation choice (F50) and
> the FedGuard experiments are finished **[pending]**. All numbers are test-split values, mean +/- standard
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
| AutoGM (lam_scale 4) | eps 3 | 0.210 +/- 0.022 | **[pending: 2 of 3 seeds done]** | **[pending]** | **[pending]** |
| FedGuard | eps 3 | **[pending]** | | | |

(`results/e5_attacks/summary.csv`, figure `e5_attacks.png` when complete.) Without DP every attack costs FedAvg
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
0.23-0.24 each - the hospital-and-bank story of the paper's introduction **[to be confirmed on the 40-round
mixed-privacy runs; figure `e5_mixed_privacy.png`]**.

## 6.6 Choosing FedGuard's settings on validation **[pending - F50]**

Table: the 8 variants of `configs/sweeps/p7_fedguard_tune.yaml` (mode, R4, s floor, lambda_scale) with mean
validation macro-F1 over no attack and 30% sign flip, HRR and attacker share
(`results/p7_fedguard_tune/selection_val.csv`); the chosen setting and why.

## 6.7 Main comparison: FedGuard vs AutoGM vs FedAvg **[pending - E5, E5m]**

- Main attack table with paired per-seed FedGuard - AutoGM differences (`results/e5_attacks/paired_fedguard_vs_autogm.csv`).
- Mixed privacy: macro-F1, HRR, attacker share, and mean weight of an honest client per privacy level
  (`results/e5_mixed_privacy/`).
- Trust weights of one real run over the rounds (`scripts/demo_dashboard.py --run ... --save`).

## 6.8 Honest rejection across privacy and heterogeneity **[pending - E4]**

HRR and macro-F1 heatmaps, epsilon {inf, 3, 1, mixed} x alpha {0.1, 0.5, 100}, AutoGM vs FedGuard (`e4_conflict.png`).

## 6.9 What each refinement contributes and where FedGuard breaks **[pending - E6, E7]**

One-change-at-a-time ablation (`e6_ablation.png`) and the stress tests over attacker share and epsilon
(`e7_stress.png`); state the breaking points and every case where FedGuard did not beat AutoGM.

## 6.10 Engineering results worth reporting

- Engine correctness: the Flower parity check found a bug in which clients trained sequentially instead of being
  averaged (F24); after the fix our engine and Flower agree to 6.9e-17 (F26). AutoGM equals the Blades reference
  to 1e-11 (F37).
- Early stopping must be checked against the learning curves: an under-trained comparison picked the wrong class
  weights (F18), patience 8 stopped FedAvg in temporary dips (F28), and a 20-round / patience-5 setting stopped
  private runs during their warm-up, invalidating 129 runs that were then re-run (F46).
- Running unattended on a laptop needs failure-safe engineering: a charger fault hung the GPU driver for hours
  (F41) and a thermal guard that died left training waiting (F51); per-round checkpoints meant no result was lost.
