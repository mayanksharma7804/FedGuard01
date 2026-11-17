# Chapter 4 - Methodology

> DRAFT (17 Nov 2026) - written from the code and Findings.md. Every number cites a finding (Fxx) so it can be
> checked against `results/master.csv`. The team must rewrite it in their own words before submission
> (college AI-use policy, plan decision D7).

## 4.1 Overview

FedGuard is evaluated as a ladder of baselines on one intrusion-detection task (Figure 5.3 of the plan): a
centralised model (B0), federated averaging over non-IID organisations (B1), the same with record-level
differential privacy at every client (B2), AutoGM without and with DP (B3, B4), and FedGuard with DP (B5).
Every rung keeps everything else fixed - data split, partitions, model, optimiser, client settings, privacy
accounting - so that a change in a result can be attributed to the one component that changed.

The system has three tiers (paper section 6.2):

1. **Client tier.** Each organisation trains the same CNN-LSTM on its own non-IID data with DP-SGD and sends its
   update (local minus global weights) with its declared privacy metadata (sigma, C, B, T, lr, n).
2. **Aggregation tier.** The server predicts each client's noise radius from the metadata, removes it from the
   distance, sets an allowance for normal heterogeneity, and computes trust weights and a weighted geometric
   median (FedGuard). AutoGM and FedAvg are drop-in alternatives with the same interface.
3. **Distribution tier.** The new global model goes back to every client.

All research code is in `src/fedguard/`; the aggregators are pure NumPy so that our engine and Flower call the
same function.

## 4.2 Data pipeline (Phase 2)

**Dataset.** NF-UNSW-NB15-v2 (University of Queensland NetFlow v2 release): 2,390,275 flows, Benign + 9 attack
classes. The secondary dataset (CICIoT2023) was dropped for time (F40) and is future work.

**Cleaning order (fixed; the order is what prevents leakage):**

1. Drop identifiers: source/destination IPv4 address, source/destination port, and `DNS_QUERY_ID`. The last
   one is a random DNS transaction number that no standard NetFlow feature list flags; keeping it made
   ~131k duplicate benign flows look unique (F08). All attack flows come from 4 of the 40 source IPs (F09), so
   IP columns would let a model memorise hosts.
2. `log1p` on the (clipped at 0) features, as float32, *before* de-duplication - a fixed formula, so it cannot
   leak, and it prevents two slightly different raw values from becoming one float after splitting (the
   leakage test caught exactly this once, F11).
3. Resolve conflicting labels: 1,563 feature vectors carry more than one attack label (never benign vs
   attack); the majority label is kept (F10).
4. De-duplicate: 95.5% of the rows are exact duplicates (F07); 104,022 unique rows with 38 features remain.
5. Stratified split 80/20 (test), then 90/10 of the rest (validation).
6. StandardScaler fitted on the training split only.
7. Cap the training split at 20,000 rows per class (keeps rare classes whole); validation and test keep the
   natural distribution.

Result: train 45,225 / validation 8,322 / test 20,805 rows (F12). The rare classes are tiny (Worms: 36 / 4 /
10 rows), so their per-class scores are noisy; they are reported, not dropped.

**Non-IID organisations.** Dirichlet label partitioning with concentration alpha in {0.1, 0.5, 100}
(0.1 = each client sees a few classes, 100 = nearly IID), K in {5, 10} clients, seeds {42, 43, 44}, minimum
client size 200 rows. The 18 partition files are saved once and loaded by every experiment (never re-drawn),
tied to the processed-data fingerprint.

**Leakage tests** (`tests/test_processed_data.py`, `test_partition.py`, `test_data_pipeline.py`): no
identifier column survives; no row is shared between train, validation and test; the scaler's mean is the
training mean; partitions are disjoint, cover the training split and respect the minimum size; the same seed
gives the same partition.

## 4.3 Model and local training (Phase 3)

One CNN-LSTM is used unchanged in every client and every baseline (40,266 parameters, F06):

```
input (B, 1, 38) -> Conv1d(1->32, k3) -> ReLU -> Conv1d(32->64, k3) -> ReLU -> MaxPool(2)
                 -> read as a sequence of 19 steps -> DPLSTM(64->64) -> last step -> Dropout(0.3) -> Linear(64->10)
```

Opacus' `DPLSTM` is used even without DP, so DP and non-DP runs are the same model; there is no BatchNorm
(incompatible with per-example clipping). The model is small on purpose: the DP noise length grows with the
square root of the number of parameters (section 4.5).

Every client uses **plain SGD** (momentum 0, no weight decay), learning rate 0.2, batch 256, cross-entropy
without class weights. These were chosen on the validation split (F18-F20): an under-trained first comparison
had picked square-root class weights, and a fair comparison (minimum epochs, longer patience) reversed it.
Plain SGD is required by the noise-radius formula: momentum would carry old noise forward and Adam would rescale
it per coordinate.

Seeds are fixed for Python, NumPy and PyTorch; each (run seed, round, client) gets its own derived seed, so runs
are bit-identical when repeated (F17) and a resumed run equals an uninterrupted one.

## 4.4 Federated engine (Phase 4)

Experiments run in our own engine `fl_sim.py` (one process, deterministic, no Ray), and the Flower framework is
used for the live demo and for a parity check (plan decision D1). Each round: every client starts from the
global weights, trains E = 5 local epochs (F27: same peak as E = 1 or 2 with a quarter of the rounds and a much
smoother curve), and returns its update; the server aggregates the updates and adds the result to the global
model; the global model is evaluated on the validation split. Early stopping keeps the best validation round
(patience 15, at least 10 rounds); at the end that model is evaluated once on the test split.

**Correctness.** Running the same FedAvg configuration in Flower (1 SuperLink + 5 SuperNodes) and in our engine
first showed a mismatch, which exposed a real bug: setting the model weights from a vector made them *views* of
the global vector, so clients trained sequentially instead of being averaged (F24). After the fix the two
engines agree to 6.9e-17 in every round (F26). The Flower client reuses the engine's training path, including
DP and attacks, and unit tests check that it produces exactly the engine's update.

## 4.5 Differential privacy and the declared noise radius (Phase 5)

Each client trains with DP-SGD (Opacus, Poisson sampling): every example's gradient is clipped to norm C = 2
(chosen on validation, F33), the clipped gradients are summed and Gaussian noise N(0, sigma^2 C^2) is added, and
the sum is divided by the expected batch size. The noise multiplier sigma is calibrated **once per client for
the whole training budget** (T local steps per round x 40 rounds) with the RDP accountant at delta = 1/N_client,
so a run that uses all 40 rounds spends exactly the target epsilon; a run that stops earlier spends less (both
are logged). Privacy levels: epsilon in {infinity (no DP), 8, 3, 1}, and a mixed setting in which epsilon
1, 3 and 8 are assigned in turn to the clients (section 4.8).

**Noise radius.** Each step adds lr * N(0, (sigma C / B)^2 I_d) to the weights; over T independent steps the
noise adds up like a random walk, so its expected length in one round's update is

    r = lr * sigma * C * sqrt(T * d) / B          (d = 40,266 parameters)

which the server computes from the declared (lr, sigma, C, T, B). B and T must be Opacus' real values: Opacus
samples with rate q = 1/ceil(n/B) and divides by int(n / batches), not by B. Experiment E0 caught our first
version using q = B/n, which under-stated the radius by 5-20% for small clients (F34). After the fix, training
one client twice from the same model with the same batches and dropout - with sigma and with sigma = 0 - gives
measured/predicted = 1.000-1.005 over 150 pairs (sigma 0.5-2, T = 40-375 steps), with a spread close to the
theoretical 1/sqrt(2d); the cosine between the noise and the clean update is -0.002 to -0.008 (F35). The radius
formula therefore needs no correction factor (kappa = 1), and the noise is orthogonal to the signal, as
FedGuard's quadrature rule assumes.

A consequence that shaped the experiments: with one shared epsilon, small clients need a larger sigma but take
fewer steps, and the two effects cancel, so every client declares almost the same radius (~12-15 per round at
epsilon 3, F36). FedGuard's distinctive idea - forgive each client *its own* noise - is only tested when
clients choose different privacy levels, which is why the mixed-privacy scenario was added.

## 4.6 Attacks (Phase 6)

30% of the clients (3 of 10) are attackers, chosen once per run from the seed. They are *smart*: they also run
DP-SGD and declare an honest-looking sigma, so they cannot be spotted by their declared noise alone.

| Attack | Kind | What the attacker does |
|---|---|---|
| Label flip | data poisoning | relabels every attack row of its data as Benign (attacks become invisible) |
| Sign flip | model poisoning | sends minus its honest update |
| Gaussian | model poisoning | sends random noise ten times the length of its honest update |

## 4.7 Aggregators

All aggregators take the clients' updates X (K x d), sample counts n and declared radii r, and return the
aggregated update and one trust weight per client.

**FedAvg** - the sample-weighted mean.

**AutoGM** (Li, Ngai and Voigt, 2023, ref [13]) - minimises sum_k a_k ||z - x_k|| + (lambda/2)||a||^2 over
the centre z and weights a on the simplex, alternating a weighted geometric median (Weiszfeld iterations) and a
simplex projection of -distance/lambda; far clients get less weight and beyond a threshold zero. We use
lambda = lambda_scale x median distance (scale-free) and tuned lambda_scale on validation (F39). Our
implementation matches the AutoGM of the Blades benchmark (by AutoGM's first author) to about 1e-11 on the
same inputs (F37). AutoGM ignores the declared noise.

**FedGuard** - AutoGM in which trust is decided by the displacement that the client's declared noise and
normal non-IID spread cannot explain. The paper's rule e_k = max(0, dist_k - r_k - s) is refined:

| | Refinement | Why |
|---|---|---|
| R1 | effective radius r_eff,k^2 = (1-b_k)^2 r_k^2 + sum_{j != k} b_j^2 r_j^2 (b = Weiszfeld weights) | the centre is itself built from noisy updates |
| R2 | quadrature residual sqrt(dist^2 - r_eff^2) (linear dist - r_eff kept as an ablation) | the noise is orthogonal to the signal (F35) |
| R3 | allowance s = median + k_mad x 1.4826 x MAD of the residuals | raw distances contain noise and would inflate s |
| R4 | noise-length tolerance: forgive z standard deviations of the noise length itself | a very private client's noise length wobbles (F02, F31) |
| s floor | s >= one standard deviation of the residual noise of the median client (F48) | under DP most residuals are hidden by noise, so median = MAD = 0 and s collapses |

Excess e_k = max(0, residual_k - s); trust a = simplex projection of -e / (lambda_scale x s); the centre is
the a-weighted geometric median. Without DP (r = 0) FedGuard reduces to a residual-based AutoGM. FedGuard's
settings (mode, R4, s floor, lambda_scale) are chosen on the validation split with the same protocol as
AutoGM's lambda, then frozen for every test-set experiment (F50).

**Unit tests** (`tests/test_aggregators.py`, plan 12.4): identical updates get equal weights; a far attacker
gets weight 0 and the output is the honest geometric median; an honest client that truly adds a lot of noise
and declares it is not rejected (AutoGM rejects it); permutation and scale invariance; the quiet-majority case
that needs R4; the s-floor cases - each in both modes, with R4 off and on. One variant (quadrature without R4)
is a documented strict expected failure (F31).

## 4.8 Evaluation protocol

**Metrics** (every run): macro-F1 over the 10 classes (main metric - rare attacks count as much as common
ones), per-class recall, benign false-positive rate, attack-detection rate, binary F1 (attack vs benign),
accuracy (for comparison only), epsilon spent at delta, the **honest rejection rate** HRR (share of honest
clients whose weight is below half a fair share, 0.5/K, averaged over rounds) and the **attacker weight share**
(total weight given to attackers, averaged over rounds). For FedAvg, HRR only reflects sample-size weighting
(small clients always weigh less than 0.5/K) and is not a robustness measure.

**Model selection** uses the validation split only; the test split is evaluated once per run, at the best
validation round. Hyper-parameters of AutoGM and FedGuard were fixed on validation before any test result of
the compared configurations was examined, to avoid p-hacking (plan 13.2).

**Statistics.** Each cell is the mean and standard deviation over seeds 42, 43, 44; a seed changes the
partition, the initial model and the DP noise together. Methods are always compared on the same partitions,
and per-seed (paired) differences are reported, because the spread between partitions at alpha = 0.5 is
large (B1: 0.537 +/- 0.076, F29). A difference smaller than the spread is not claimed as a win.

**Reproducibility.** Each run folder stores the full configuration, the git commit, the host, the package
versions and timings; `results/master.csv` is built by `scripts/collect_results.py` and every figure by
`scripts/make_plots.py` - no number or figure is made by hand.
