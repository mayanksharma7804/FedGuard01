# FedGuard - Findings log

Every finding of the project, in the order we found it: **what we did, what we found, the evidence,
and what we decided**. This file feeds the report (Methodology, Results, Limitations) and the viva.
It is updated after every phase.

**Status:** Phases 1-5 complete; Phase 6 paused at 30 of 69 runs (9 Oct 2026, 07:22, clean stop between runs; resume with the Phase 6 queue command in CLAUDE.md).

| ID | Phase | Finding (one line) | Type |
|---|---|---|---|
| [F01](#f01) | Plan | AutoGM wrongly rejects many honest DP clients (synthetic check) | Method |
| [F02](#f02) | Plan | FedGuard rejects a very private honest client when the others are quiet; fix R4 has a trade-off | Method |
| [F03](#f03) | 1 | The laptop's Store Python 3.14 is too new; project pinned to Python 3.11 via uv | Setup |
| [F04](#f04) | 1 | torch 2.14.1 has no CUDA 12.8 build; cu126 used instead | Setup |
| [F05](#f05) | 1 | Flower 1.39 runs on native Windows without Ray, after 3 fixes | Setup |
| [F06](#f06) | 1 | DP-SGD is ~5x slower on CPU than GPU; ~15 min per DP run on the RTX 3050 | Measurement |
| [F07](#f07) | 2 | 95.5% of NF-UNSW-NB15-v2 rows are duplicates once identifiers are removed | Data |
| [F08](#f08) | 2 | `DNS_QUERY_ID` is a hidden identifier that masked 131k duplicate benign rows | Data |
| [F09](#f09) | 2 | All attack flows come from 4 of the 40 source IPs | Data |
| [F10](#f10) | 2 | 1,563 feature vectors carry more than one label | Data |
| [F11](#f11) | 2 | A float32 collision leaked 1 test row into training; fixed | Leakage |
| [F12](#f12) | 2 | Rare classes become tiny after cleaning (Worms: 10 test rows) | Data |
| [F13](#f13) | 2 | Our numbers will not be directly comparable with papers that skip de-duplication | Evaluation |
| [F14](#f14) | 2 | Training set is ~45k rows, so runs are ~2-3x faster than first estimated | Measurement |
| [F15](#f15) | 2 | The CICIoT2023 copy is a third-party pre-split subset (open check) | Data |
| [F16](#f16) | 3 | Without privacy noise, GPU and CPU are equally fast; 3 parallel runs give ~1.9x throughput | Measurement |
| [F17](#f17) | 3 | Same seed gives bit-identical training on the GPU | Reproducibility |
| [F18](#f18) | 3 | Early stopping with patience 10 cut plain-SGD training short; first class-weight comparison was unfair | Method |
| [F19](#f19) | 3 | Fair grid: no class weights + lr 0.2 is best (val macro-F1 0.632); the early "sqrt wins" result reversed | Method |
| [F20](#f20) | 3 | Class weights trade precision for rare-class recall and lower macro-F1 | Method |
| [F21](#f21) | 3 | Long parallel runs heat the laptop (GPU 86 C) and slow epochs up to ~3x | Measurement |
| [F22](#f22) | 3 | **B0 ceiling: test macro-F1 0.597 +/- 0.004**, binary F1 0.912, benign FPR 0.068 | Result |
| [F23](#f23) | 3 | Exploits absorbs most rare-attack errors; 47% of Analysis attacks are missed as Benign | Result |
| [F24](#f24) | 4 | **Critical engine bug caught by the Flower parity check: clients were trained one after another, not with FedAvg** | Bug |
| [F25](#f25) | 4 | 5 local Flower SuperNodes + 2 training jobs ran the laptop out of RAM and killed both jobs | Measurement |
| [F26](#f26) | 4 | After the fix, our engine and Flower agree to 1e-16 over 5 rounds | Verification |
| [F27](#f27) | 4 | 5 local epochs per round beats 1 or 2 at equal compute and needs ~45 rounds instead of ~170 | Method |
| [F28](#f28) | 4 | Non-IID FedAvg curves have deep temporary dips; patience 8 stopped runs early, so B1 was re-run with patience 15 | Method |
| [F29](#f29) | 4 | **B1: FedAvg matches B0 with IID clients (0.603) but loses 0.06 at alpha 0.5 and 0.16 at alpha 0.1**; rare classes held by one small client disappear | Result |
| [F30](#f30) | 5 | Per-example gradient norms: median ~1-4 with a heavy tail; my first measurement was 512x too big (wrong loss reduction) | Measurement |
| [F31](#f31) | 7 (prep) | Synthetic unit tests: without R4, quadrature FedGuard rejects the most private honest client in 30-50% of seeds; linear + R4 has its own failure | Method |
| [F32](#f32) | 5 | A DP round costs ~65-80 s (2.5-3x a plain round); a 40-round DP run ~45-55 min | Measurement |
| [F33](#f33) | 5 | Clipping norm C = 2 chosen on validation (0.264 vs 0.21-0.23); DP at epsilon 3 costs a lot: val macro-F1 ~0.26 vs ~0.57 without DP | Method |
| [F34](#f34) | 5 | **E0 caught a bug: the declared noise radius was 5-20% too small** because Opacus samples with q = 1/ceil(n/B), not B/n; fixed and regression-tested | Bug |
| [F35](#f35) | 5 | **E0: the noise-radius formula predicts the real DP noise to within 1% (kappa = 1.00)**; noise is orthogonal to the signal (cos ~ 0) | Result |
| [F36](#f36) | 5 | With one shared epsilon every client gets almost the same noise radius, whatever its size; FedGuard's idea needs a mixed-privacy scenario | Method |
| [F37](#f37) | 6 (prep) | Our AutoGM matches the Blades benchmark's AutoGM (AutoGM's first author) to 1e-11 on identical inputs | Verification |
| [F38](#f38) | 5 | **B2: privacy is very costly here - macro-F1 0.51 (no DP) -> 0.25 (eps 8) -> 0.22 (eps 3) -> 0.18 (eps 1)**; rare attack classes vanish, attack-vs-benign detection mostly survives | Result |
| [F39](#f39) | 6 | **AutoGM under DP has no good lambda: small lambda collapses onto one client (HRR 0.90), large lambda lets sign-flip attackers keep ~32% of the weight**; lambda_scale = 4 chosen | Result |
| [F40](#f40) | 6 | Fast plan (team request): fewer attacker fractions, seeds and epsilon levels; ~2-2.5 days of GPU instead of ~6 | Decision |
| [F41](#f41) | 6 | A charger connect/disconnect burst made the NVIDIA driver fail; both trainings hung silently for ~3 h 50 min; watchdog added | Measurement |

---

## Before implementation (planning stage, synthetic data)

### F01
**AutoGM wrongly rejects many honest clients when they add DP noise (synthetic logic check).**
- **What we did:** Ran the guide's reference AutoGM and FedGuard code on synthetic rounds: 4,000-dimensional updates, 10 clients, 3 sign-flip attackers who also add DP noise and declare a normal sigma, honest clients with different privacy levels, 10 seeds per noise level.
- **What we found:** AutoGM (lambda x1) wrongly rejected **57-86%** of honest clients at every noise level. With lambda x3 it rejected fewer (up to **29%**) but gave attackers up to **~10%** of the weight at high noise. FedGuard (linear and quadrature) kept both near **0%** in this setting. This is the conflict the paper describes (Gap 1).
- **Evidence:** Implementation plan, section 12.3 and Figure 12.2.
- **Decision / impact:** Synthetic data follows FedGuard's own assumptions, so this only shows the logic is right. The real proof must come from IDS experiments (E4, E5).

### F02
**FedGuard can reject a very private honest client when all the other clients are quiet; the fix (R4) has a trade-off.**
- **What we did:** Tested a second scenario: 9 low-noise honest clients + 1 very private honest client, 200 seeds.
- **What we found:** The guide's FedGuard rejected the private client in **55.5%** (quadrature) / **54.0%** (linear) of seeds. Cause: the 9 quiet clients have near-identical residuals, so the allowance *s* collapses to ~0, while the private client's noise length naturally wobbles by a few percent of a large radius. Adding a per-client noise-length tolerance (**R4**, z = 3) cut this to **0.5%**, and a far attacker still got zero weight in every seed. **Trade-off:** in the F01 scenario, quadrature + R4 let attackers take **6-15%** weight at the highest noise levels; linear + R4 kept attackers at 0% but rejected up to **7%** of honest clients.
- **Evidence:** Implementation plan, section 12.3, Figures 12.2-12.3.
- **Decision / impact:** R4 is a *candidate*. Ablation E6 decides on real data whether to use it and in which mode. Report the result either way.

---

## Phase 1 - Setup and foundations

### F03
**The laptop's default Python (Microsoft Store, 3.14) is too new for the project.**
- **What we did:** Checked installed tools before setup.
- **What we found:** `python` pointed to the Store's Python 3.14.2. Several packages (Ray, some ML libraries) lag behind new Python versions.
- **Decision / impact:** uv creates `.venv` with **Python 3.11.17** inside the project; the Store Python is never used. Same tool works on Windows and Linux.

### F04
**torch 2.14.1 has no CUDA 12.8 wheel; the setup script would have failed.**
- **What we did:** Ran `scripts/windows/setup_windows.ps1` (which defaulted to the cu128 index), then listed the available builds on download.pytorch.org.
- **What we found:** For torch 2.14.1 on Windows / Python 3.11 only **cpu, cu126, cu130, cu132** exist. cu130/cu132 need NVIDIA drivers >= 580; this laptop has **566.07** (CUDA 12.7).
- **Decision / impact:** The script now uses **cu126**. Result: `torch 2.14.1+cu126`, `cuda True (RTX 3050 Laptop GPU)`.

### F05
**Flower 1.39 works on native Windows in deployment mode (no Ray, no WSL) after three fixes.**
- **What we did:** Wrote a small Flower app (`flower_check/`) and a script that starts 1 SuperLink + 2 SuperNodes and runs 3 FedAvg rounds.
- **What we found:**
  1. SuperLink/SuperNode start a helper called `flower-superexec` by name, so `.venv\Scripts` must be on PATH (else `FileNotFoundError: [WinError 2]`).
  2. In flwr 1.39 the Control API is **HTTP on port 8000**, not gRPC on 9093 as in older docs.
  3. `Start-Process` must quote the `--node-config` value.
  Also: `flwr new` now downloads templates from Flower's online hub. After the fixes: 3 rounds, train loss **0.171 -> 0.048 -> 0.025**.
- **Evidence:** `scripts/windows/flower_check.ps1`, `docs/flower_windows.md`.
- **Decision / impact:** Flower stays in deployment mode for the demo; experiments use our own engine (Phase 4). FedGuard will plug into Flower by overriding `FedAvg.aggregate_train`.

### F06
**DP-SGD training must run on the GPU: it is ~5x slower on the CPU.**
- **What we did:** Timed the real CNN-LSTM (with Opacus DPLSTM) with and without DP-SGD on CPU and GPU (`scripts/check_gpu_dp.py`).
- **What we found:** The model has **40,266 parameters** (target 40-60k) and passes the Opacus validator. Seconds per step: GPU no-DP **0.022**, GPU DP **0.045**, CPU no-DP 0.030, CPU DP **0.213**.
- **Decision / impact:** All DP runs on the GPU. (Run time per experiment was later revised by F14.)

---

## Phase 2 - Data pipeline (NF-UNSW-NB15-v2)

### F07
**95.5% of NF-UNSW-NB15-v2 is duplicate rows once identifier columns are removed.**
- **What we did:** Profiled the 2,390,275 raw rows after dropping IPs, ports and `DNS_QUERY_ID`.
- **What we found:** **2,282,035** rows (95.5%) are exact copies of another row. Benign keeps only **3%** of its rows (2,295,222 -> 68,992 before label resolution); Reconnaissance keeps 8.8%; Generic 14.3%.
- **Evidence:** `notebooks/01_eda.ipynb` section 5, `results/figures/eda_duplicates.png`, `data/processed/nf_unsw_manifest.json`.
- **Decision / impact:** De-duplicate **before** the train/test split, otherwise the same flow sits in both and accuracy is fake (paper Gap 4). Final data: **104,022 unique rows, 38 features**.

### F08
**`DNS_QUERY_ID` is a hidden identifier.**
- **What we did:** Compared de-duplication with and without `DNS_QUERY_ID`.
- **What we found:** It is a random DNS transaction number (64,243 distinct values, non-zero in 15.5% of rows). Keeping it made otherwise-identical flows look unique: unique rows **236,999** with it vs **104,025** without it; benign **199,838** vs **68,988** (it hid ~131k duplicate benign rows).
- **Decision / impact:** Dropped together with IPs and ports. Report it as an identifier the standard NetFlow feature set does not flag.

### F09
**All attack flows come from just 4 of the 40 source IP addresses.**
- **What we did:** Grouped flows by `IPV4_SRC_ADDR`.
- **What we found:** 40 source IPs; **36** send only benign flows, all attacks come from the other 4.
- **Evidence:** `notebooks/01_eda.ipynb` section 4.
- **Decision / impact:** Confirms that IP columns would let a model "detect" attacks by memorising hosts. They are dropped.

### F10
**1,563 feature vectors appear with more than one label.**
- **What we did:** After de-duplication, checked whether the same feature vector carries different `Attack` labels.
- **What we found:** **1,563** vectors (covering 5,781 rows), all attack-vs-attack (never benign-vs-attack), e.g. Reconnaissance vs Exploits.
- **Decision / impact:** Keep the **majority label** per vector (ties: alphabetical). Deleting them would have removed most Reconnaissance and Backdoor rows. This is irreducible label noise and caps achievable per-class accuracy for those classes.

### F11
**A float32 collision leaked 1 test row into the training set; caught by a test and fixed.**
- **What we did:** Wrote a leakage test that checks no row is shared between train, val and test (`tests/test_processed_data.py`).
- **What we found:** The test failed: **1 of 20,805** test rows (a Backdoor flow) was identical to a training row. De-duplication had run on raw values, but `log1p` + conversion to float32 turned two slightly different raw values into the same number.
- **Decision / impact:** `log1p` is now applied as float32 **before** de-duplication (a fixed formula, so it cannot leak). This found 3 more hidden duplicates (104,025 -> **104,022**). Now train/val/test share **zero** rows; all 27 tests pass.

### F12
**Rare classes become very small after cleaning.**
- **What we did:** Counted classes after de-duplication, the stratified split and the 20k/class training cap.
- **What we found:** Train / val / test rows - Benign 20,000 / 5,519 / 13,797 · Exploits 15,734 / 1,748 / 4,371 · Fuzzers 4,856 / 540 / 1,349 · DoS 2,234 / 249 / 621 · Generic 1,241 / 138 / 345 · Analysis 524 / 58 / 146 · Reconnaissance 246 / 27 / 68 · Backdoor 200 / 22 / 55 · Shellcode 154 / 17 / 43 · **Worms 36 / 4 / 10**.
- **Decision / impact:** Per-class scores for Worms, Shellcode, Backdoor and Reconnaissance will be noisy (a few test rows). Report them with that caveat; do not drop them. Macro-F1 stays the main metric. With alpha = 0.1 and 10 clients, some clients hold only 2-3 classes and the smallest has 215 rows.

### F13
**Our results will not be directly comparable with papers that do not de-duplicate.**
- **What we did:** Compared our cleaning with the protocol implied by published NF-UNSW-NB15-v2 results.
- **What we found:** Papers reporting high federated scores on this dataset (e.g. ref [1], 91-93%) do not report removing duplicates; with 95.5% duplicates (F07), much of their test set can be copies of training rows.
- **Decision / impact:** The report must state that our numbers come from a de-duplicated, leakage-checked protocol and are expected to be lower; compare trends, not raw numbers, and point to paper section 4.4 (Gap 4).

### F14
**The training set is ~45k rows, so experiments are faster than first estimated.**
- **What we did:** Measured the final training-set size after cleaning and capping.
- **What we found:** **45,225** training rows (the plan assumed 100-150k) = about **177 steps per round** at batch 256.
- **Decision / impact:** Using F06 timings, a 40-round DP run should take roughly **5-7 minutes** on the RTX 3050 instead of ~15. To be confirmed by real timing in Phase 3/4.

### F15
**The CICIoT2023 copy is a third-party pre-split subset (check pending).**
- **What we did:** Inspected the Kaggle copy (himadri07/ciciot2023).
- **What we found:** train 5,491,971 · validation 1,176,851 · test 1,176,851 rows (7.85M of the original ~46M), 46 features, 34 classes.
- **Decision / impact:** Before using it in E8 (Phase 8), check for duplicates *across* the three files; if found, merge, de-duplicate and re-split with our own pipeline. Group 34 labels into 8 classes.

---

## Phase 3 - Centralised CNN-LSTM baseline (B0)

### F16
**Without privacy noise the GPU is no faster than the CPU; running 3 experiments in parallel nearly doubles throughput.**
- **What we did:** Trained B0 for 3 epochs on the real data on GPU and CPU; then ran 3 training jobs at the same time on the GPU.
- **What we found:** One run: **3.9 s/epoch (GPU)** vs **3.8 s/epoch (CPU)**; the 40k-parameter model is too small to keep a GPU busy (~26% use). Three runs in parallel: **6.3 s/epoch each**, GPU at **98%**, i.e. ~**1.9x** more epochs per second in total.
- **Decision / impact:** Keep the GPU as the default (DP-SGD is 5x faster there, F06). In Phase 8, run 2-3 experiments in parallel to roughly halve the total time.

### F17
**Training is exactly reproducible: the same seed gives bit-identical results.**
- **What we did:** Ran the same seed-42 configuration in separate processes on different occasions.
- **What we found:** Identical epoch-1 losses (e.g. 1.5458, 1.9953, 2.2946 for the three class-weight settings) and identical validation scores. `tests/test_train.py` checks the same on CPU.
- **Decision / impact:** Differences between methods can be attributed to the method, not to randomness; 3 seeds measure the genuine run-to-run spread.

### F18
**Early stopping with patience 10 stopped plain-SGD training while the model was still improving, so the first class-weight comparison was unfair.**
- **What we did:** Trained B0 (seed 42, lr 0.05) with class weights none / sqrt / balanced and compared on the **validation** split.
- **What we found:** Val macro-F1 none **0.415**, sqrt **0.554**, balanced **0.108**. But training loss was still falling in every run when it stopped (sqrt 0.67 -> 0.62 in the last 16 epochs), and the balanced run was stopped at **epoch 11** (best epoch 1) while its loss was still dropping (2.29 -> 1.78): plain SGD starts slowly and the early validation score is noisy. Without weights the model never detected Reconnaissance or Worms (recall 0.00) and almost never DoS (0.02).
- **Decision / impact:** Added `min_epochs = 20`, raised patience to 20 and max epochs to 200, and re-ran a fair grid (class weight x learning rate {0.05, 0.1, 0.2}) on the validation split. Result: see the next entry.

### F19
**With fair training, "no class weights, lr 0.2" is the best B0 setting; the first comparison's conclusion reversed.**
- **What we did:** 3x3 grid on seed 42 - class weight {none, sqrt, balanced} x lr {0.05, 0.1, 0.2}, plain SGD, min 20 / patience 20 / max 200 epochs - judged only by **validation** macro-F1.
- **What we found:**

  | class weight \ lr | 0.05 | 0.1 | 0.2 |
  |---|---|---|---|
  | none | 0.590 | 0.625 | **0.632** |
  | sqrt | 0.555 | 0.565 | 0.586 |
  | balanced | 0.469 | 0.506 | 0.484 |

  In the unfair first comparison (F18) sqrt looked much better than none (0.554 vs 0.415); with enough epochs, "none" overtakes it. Plain SGD at lr 0.05 needs ~160 epochs; lr 0.2 converges faster and higher. The winner's 10-epoch average val F1 was ~0.60-0.61 from epoch 160 to 200 (nearly flat).
- **Decision / impact:** Final B0 config: class_weight **none**, lr **0.2**, max 300 epochs, patience 30 (`configs/b0_centralised.yaml`). The same lr will be used by every federated client (the DP noise radius formula includes lr, so it only has to be known, not small). Lesson for the report: an under-trained comparison can pick the wrong design.

### F20
**Class weights buy rare-class recall at the cost of precision, and lower macro-F1 overall.**
- **What we did:** Compared per-class validation recall across the grid (F19).
- **What we found:** Balanced weights raise rare-class recall strongly (Shellcode 0.35 -> **1.00**, Worms 0.25 -> **0.75**, Backdoor 0.36 -> **0.82**, Reconnaissance 0.67 -> **0.85**) but push more benign and Exploits traffic into rare classes: benign FPR **0.07 -> 0.12**, Exploits recall 0.93 -> 0.67. Net effect: macro-F1 drops (0.632 -> 0.48-0.51).
- **Decision / impact:** Keep macro-F1 as the selection metric (paper O5), so no weights. Report the trade-off: a security team that cares most about catching rare attacks could prefer weights and accept more false alarms.

### F21
**Long parallel runs heat the laptop and slow training a lot.**
- **What we did:** Watched epoch times and hardware state during the 3-parallel grid.
- **What we found:** Epoch time rose from 6.2 s to **11-21 s** over ~1 hour. GPU at **86 C**, clock 1732 of 2100 MHz; CPU at 82% load (Python overhead of 3 jobs plus other apps). Battery at 23% while running.
- **Decision / impact:** For Phase 8: run **2** jobs in parallel (not 3), close heavy apps, keep the charger in, use a cooling pad if possible, and measure throughput again after 30+ minutes rather than in the first epochs.

### F22
**Centralised ceiling (B0): test macro-F1 0.597 +/- 0.004 over 3 seeds.**
- **What we did:** Trained the final B0 config (no class weights, plain SGD lr 0.2, early stopping on validation macro-F1, max 300 epochs) with seeds 42, 43, 44 and evaluated the best epoch once on the test set.
- **What we found:**

  | Metric (test) | Mean +/- std |
  |---|---|
  | Macro-F1 | **0.597 +/- 0.004** |
  | Weighted-F1 | 0.896 +/- 0.001 |
  | Accuracy | 0.898 +/- 0.002 |
  | Benign FPR | 0.068 +/- 0.005 |
  | Attack detection rate | 0.950 +/- 0.008 |
  | Binary F1 (benign vs attack) | 0.912 +/- 0.000 |

  Best epochs 107-180 (~4.5 s/epoch, ~10-15 minutes per run). Validation macro-F1 ~0.63 vs test ~0.60: the usual small optimism of choosing settings on validation.
- **Evidence:** `results/b0_centralised/summary.csv`, `per_class.csv`, `selection_grid_val.csv`; `results/figures/b0_centralised_*.png`.
- **Decision / impact:** This is the ceiling B1-B5 are compared with. The binary view (0.91 F1, 95% of attacks flagged) is strong; the multi-class view is hard because of tiny, overlapping rare classes. Our accuracy (0.90) is close to the 91-93% reported by ref [1], but macro-F1 shows the rare-class weakness that accuracy hides (the paper's ref [22] point).

### F23
**Exploits absorbs most rare-attack errors, and 47% of Analysis attacks are missed as Benign.**
- **What we did:** Studied the row-normalised confusion matrix (test, 3 seeds summed).
- **What we found:** Predicted as **Exploits**: Worms **70%**, DoS **49%**, Reconnaissance **31%**, Backdoor **28%**, Generic **24%**. Shellcode goes mostly to Fuzzers (**40%**). The most dangerous error: **47% of Analysis** and **21% of Backdoor** test rows are predicted **Benign** (the attack is missed entirely). Benign itself leaks 4% to Fuzzers and 2% to Exploits (the 0.068 FPR).
- **Decision / impact:** Most multi-class errors are attack-vs-attack (still flagged), matching the Recon/Exploits label conflicts in the data (F10). In the report, discuss Analysis/Backdoor-to-Benign as the real security risk. In later phases, track per-class recall of Analysis, Backdoor and DoS: robust aggregation may hurt exactly these rare classes (paper Gap 3).

---

## Phase 4 - Federated engine and FedAvg (B1)

### F24
**The Flower parity check caught a critical bug: our engine trained the clients one after another instead of doing FedAvg.**
- **What we did:** Ran the same FedAvg config (5 clients, alpha 0.5, 5 rounds, CPU) through Flower (1 SuperLink + 5 SuperNodes, our `AggregatorStrategy`) and through our engine `fl_sim.py`, and compared validation macro-F1 round by round.
- **What we found:** Mismatch of up to **0.084**. Replaying the Flower path inside one process isolated the cause: `set_flat()` used PyTorch's `vector_to_parameters`, which makes the model's parameters *views* of the global vector. Training a client therefore modified the global model in place, every client's delta came out as **zero**, and each client simply continued from where the previous one stopped - sequential training, not federated averaging. The 1-client exactness test could not see it (with one client both are identical). The buggy engine even "learned" faster (val macro-F1 0.19 vs 0.10 after 5 rounds), which made the bug look like good behaviour.
- **Decision / impact:** `set_flat` now **copies** values into the parameters. Two regression tests added: a client never modifies the global vector, and with 3 clients round 1 equals the weighted mean of 3 independently trained local models. All runs made with the buggy engine (one smoke run and the first local-epoch selection runs) were moved to `results/runs/_invalid_engine_bug_F24/` and repeated. **Lesson for the report:** cross-checking against an independent implementation (Flower) is what exposed a bug that unit tests and plausible-looking learning curves had hidden.

### F25
**Running 5 Flower SuperNodes on the laptop while 2 trainings were running exhausted the 16 GB of RAM.**
- **What we did:** Started the parity check (SuperLink + 5 SuperNodes, each spawning its own Python + PyTorch process) while two selection runs were training.
- **What we found:** Both training runs crashed with `numpy._ArrayMemoryError: Unable to allocate 2.75 MiB`; each Flower node process loads PyTorch and the dataset separately.
- **Decision / impact:** Never run Flower checks next to training jobs. For the live demo, prefer the multi-laptop setup; on one laptop keep it to a few SuperNodes and nothing else running.

### F26
**After the fix, our engine and Flower produce the same FedAvg run to within 1e-16.**
- **What we did:** Re-ran `scripts/windows/flower_parity.ps1` (Flower deployment mode, 5 SuperNodes, our `AggregatorStrategy`) and `scripts/compare_flower_parity.py` (our engine, same config, CPU).
- **What we found:** Identical validation macro-F1 and accuracy in all 5 rounds; maximum difference **6.9e-17**. Technical detail: Opacus `DPLSTM`'s `state_dict()` lists each LSTM weight twice under two alias names (e.g. `lstm.weight_ih_l0` and `lstm.l0.ih.weight`), so the parameter vector is always built from `model.parameters()` after `load_state_dict`, never from the state_dict order.
- **Evidence:** `results/flower_parity/parity_comparison.csv`.
- **Decision / impact:** Experiments run in our fast engine; the paper's "implemented with Flower" claim is backed by this exact parity, and the same `AggregatorStrategy` will carry FedGuard in the Flower demo.

### F27
**Five local epochs per round give the best and most stable FedAvg, with ~45 rounds instead of ~170.**
- **What we did:** FedAvg, 5 clients, alpha 0.5, seed 42, the same total training budget (~300 local epochs) split as E = 1, 2 or 5 local epochs per round (patience scaled to 30 local epochs); judged on the **validation** split.
- **What we found:**

  | E | best val macro-F1 | smoothed best | local epochs to best | minutes to best | rounds run |
  |---|---|---|---|---|---|
  | 1 | 0.542 | 0.511 | 171 | 14.7 | 201 |
  | 2 | 0.535 | 0.520 | 230 | 19.7 | 130 |
  | **5** | **0.548** | **0.546** | 215 | 15.7 | **49** |

  All three reach a similar peak, but E = 5 is much less noisy round to round and needs about a quarter of the rounds.
- **Decision / impact:** B1-B5 use **E = 5**, max 80 rounds, patience 8 rounds, min 10 rounds. This is also close to the 30-50 rounds the plan expected. For DP (Phase 5) it should help: the useful part of an update grows roughly with the number of local steps T, while the DP noise grows only with sqrt(T), so the per-round signal-to-noise ratio improves - good for telling honest noise from attacks. Risk to watch: more client drift under strong non-IID (alpha = 0.1).

### F28
**Under non-IID data, FedAvg's validation curve has deep temporary dips; patience 8 stopped runs in a dip, so B1 was re-run with patience 15.**
- **What we did:** Ran B1 (FedAvg, E = 5, alpha {0.1, 0.5, 100} x seeds {42, 43, 44}) with patience 8 / max 80 rounds and inspected the learning curves before accepting the numbers.
- **What we found:** First-pass test macro-F1 - alpha 0.1: 0.380 +/- 0.006 · alpha 0.5: 0.516 +/- **0.058** · alpha 100: 0.573 +/- 0.022. The large alpha 0.5 spread came partly from early stopping: seed 43's validation macro-F1 dropped to **0.187** in one round and was **recovering** (0.368 -> 0.444 -> 0.465) when patience ran out; seed 44 was still at its best when stopped; one alpha 100 run peaked at round 78 of 80. (Alpha 0.5 runs use different partitions per seed, so some spread is real.)
- **Decision / impact:** Patience raised to **15 rounds**, max rounds to **120**, for all federated baselines; first-pass runs moved to `results/runs/_superseded_patience8_F28/` (kept for the record) and B1 re-run. Same lesson as F18: always check curves before trusting an early-stopped number. Final B1 numbers: see F29.

### F29
**B1 (final): federation itself costs nothing with IID clients, but non-IID data costs up to 0.16 macro-F1, and rare attack classes held by one small client disappear.**
- **What we did:** FedAvg, 5 clients, E = 5, patience 15, max 120 rounds, alpha {0.1, 0.5, 100} x seeds {42, 43, 44} (each seed has its own Dirichlet partition); best round chosen on validation, test evaluated once. Compared with B0 (F22) and with the 91-93% reported by ref [1].
- **What we found (test, mean +/- std over 3 seeds):**

  | alpha | macro-F1 | gap to B0 | accuracy | binary F1 | benign FPR | detection | best round |
  |---|---|---|---|---|---|---|---|
  | 100 (IID) | **0.603 +/- 0.005** | +0.006 | 0.896 | 0.911 | 0.074 | 0.959 | 100 (86-110) |
  | 0.5 | **0.537 +/- 0.076** | -0.060 | 0.888 | 0.906 | 0.057 | 0.921 | 53 (34-69) |
  | 0.1 | **0.435 +/- 0.028** | -0.163 | 0.873 | 0.894 | 0.067 | 0.915 | 67 (40-90) |

  - **IID:** FedAvg reaches the centralised ceiling (0.597 +/- 0.004); two of three runs used most of the 120-round budget, but their validation curves were flat (+/- 0.01) over the last 20 rounds.
  - **alpha 0.5** has a large seed spread (0.573 / **0.450** / 0.588). Seed 43's partition gives one client 21.5k of the 45k rows, almost only Benign + Exploits; its curve rose slowly and plateaued at ~0.45-0.48 (it did not stop in a dip - checked).
  - **alpha 0.1:** the binary view hardly changes (binary F1 0.894 vs 0.912 in B0) - the model still separates attack from benign - but **which** attack it is gets lost. Per-class recall (alpha 100 -> alpha 0.1) falls most for rare classes: Worms 0.30 -> 0.03, Backdoor 0.36 -> 0.13, DoS 0.36 -> 0.20, Generic 0.61 -> 0.29. Example (seed 42): a small client (2,971 rows, 6.6% FedAvg weight) holds 241 of the 247 Reconnaissance training rows, and Reconnaissance recall is **0**.
  - **ref [1] (91-93%):** our accuracy is 0.87-0.90 and binary F1 0.89-0.91, a little below. Our data is de-duplicated (95.5% duplicates removed, F07/F13) and identifiers are dropped, which removes easy, memorisable rows, so the numbers are not directly comparable; macro-F1 over 10 classes is our main metric.
- **Evidence:** `results/b1_fedavg/summary_by_alpha.csv`, `per_class_recall_by_alpha.csv`, `results/master.csv`, figures `results/figures/b1_fedavg_by_alpha.png`, `b1_fedavg_rounds.png`, `b1_fedavg_per_class_recall.png`. ~26 s per round, 30-50 minutes per run on the RTX 3050 (2 runs in parallel).
- **Decision / impact:** B1 is the reference for every later federated experiment. Main experiments use **alpha 0.5** (the plan's setting); its large seed variance means differences between methods must be read against +/- 0.08, so every method always runs on the **same three partitions**, and paired (same-seed) comparisons will be reported. For the report: the loss under non-IID is a loss of rare classes (minority clients with unique classes are out-voted by sample-size weighting), not of attack detection as such.

---

## Phase 5 - Differential privacy (B2) and noise-radius calibration (E0)

### F30
**Per-example gradient norms of the CNN-LSTM have a median of about 1-4 and a heavy tail; this sets the range for the clipping norm C. My first measurement was wrong by a factor of 512.**
- **What we did:** `scripts/measure_grad_norms.py` trains the B0 model (no DP) and, after 0/1/3/10/30 epochs, measures the gradient norm of 4,096 single training rows with Opacus' `GradSampleModule`.
- **What we found:** The first run reported medians of ~650-2,000. Checked against plain autograd on single rows (norm ~1.2), Opacus' per-sample gradients were exactly **batch-size (512) times** too large: `GradSampleModule` assumes a **mean**-reduced loss and my script used `reduction="sum"`. The training code uses the mean loss (as `make_private` expects), so DP training itself was never affected. Corrected values:

  | epoch | p10 | median | p90 | share > C=0.5 | > 1 | > 2 |
  |---|---|---|---|---|---|---|
  | 0 | 1.21 | 1.27 | 1.54 | 100% | 100% | 0.4% |
  | 1 | 2.15 | 3.11 | 5.41 | 100% | 100% | 98% |
  | 10 | 0.01 | 0.86 | 13.3 | 60% | 47% | 38% |
  | 30 | 0.004 | 0.90 | 12.2 | 59% | 48% | 37% |

  Once trained, the rows split into well-fitted ones (norm ~0) and a heavy tail of hard rows (rare, confusable classes).
- **Evidence:** `results/dp/grad_norms.csv`.
- **Decision / impact:** Pilot C in {0.5, 1, 2, 4} on the validation split (F33). The script now uses the mean loss with a comment explaining why.

### F31
**Synthetic unit tests of FedGuard: without R4, the quadrature version rejects the most private honest client in 30-50% of seeds; R4 fixes that, but linear + R4 rejects other honest clients when clients are very similar.**
- **What we did:** Wrote the plan's unit tests (`tests/test_aggregators.py`, section 12.4) for `autogm` and `fedguard` (linear / quadrature, R4 off / on). Scenario: 10 honest clients with radii 0.2 ... 2.0, all declaring their true noise, around a shared update of length 1 plus a client-specific non-IID "spread"; 30 seeds per spread level; 4,000-dimensional updates.
- **What we found:** Share of seeds where the most private client is rejected (weight < 0.5/K), and mean HRR:

  | spread | linear | linear + R4 | quadrature | quadrature + R4 |
  |---|---|---|---|---|
  | 0 | 50% (HRR 0.32) | 0% (0.00) | 50% (0.27) | 0% (0.00) |
  | 0.1 | 10% (0.05) | 0% (**0.17**) | 43% (0.09) | 0% (0.17) |
  | 0.2 | 0% (0.02) | 0% (**0.24**) | 40% (0.11) | 0% (0.03) |
  | 0.3 | 0% (0.02) | 0% (0.00) | 40% (0.13) | 0% (0.00) |
  | 0.5 | 0% (0.00) | 0% (0.00) | 30% (0.08) | 0% (0.00) |

  AutoGM rejects the most private client in every case (the O2 conflict). Without R4, quadrature punishes the natural wobble of a large noise length (std r^2 sqrt(2/d) on the squared length) because synthetic non-IID spreads are all almost the same length, so the allowance s is tiny. With R4, the noisy clients' residuals are often cut to 0; at small spread this makes the median and MAD - and so s - collapse, and the quiet clients with a little spread get rejected instead (linear + R4, HRR up to 0.24). All variants still give a far attacker zero weight, are permutation- and scale-invariant.
- **Decision / impact:** The unit test for this case asserts the three variants that work at a realistic spread (0.3) and marks quadrature without R4 as a **strict expected failure** (documented limitation, not hidden). This extends F02: the choice of mode x R4 is not obvious, and the allowance s (median + k*MAD) is the fragile part. Ablation E6 must decide on real IDS data; a floor on s (e.g. relative to the median distance) is a candidate fix to test in Phase 7.

### F32
**A DP-SGD round costs about 65-80 s on the RTX 3050 (2.5-3x a plain round), so a 40-round DP run takes 45-55 minutes.**
- **What we did:** Timed the first DP-FedAvg runs (5 clients, E = 5, Opacus with Poisson sampling, 2 runs in parallel on the GPU).
- **What we found:** **65-68 s per round** next to one plain run, **78-82 s** next to another DP run (plain FedAvg round: ~26 s). A 2-round smoke run spends **epsilon 2.9992 of 3** (budget 2 rounds), and the full pilot spends 2.998 of 3 over 40 rounds - the accountant works as designed.
- **Decision / impact:** B2 (12 runs, 9 with DP) takes ~4-5 hours with 2 parallel runs. The plan's Phase 8 matrix (E5 alone = 144 DP runs at K = 10) would need ~60-70 hours of GPU time; it must be trimmed or split across more machines - to be decided at the start of Phase 6 from these timings.

### F33
**Clipping norm C = 2 is best on validation, but DP at epsilon = 3 is very costly in this setting: validation macro-F1 ~0.21-0.26 against ~0.57 without DP.**
- **What we did:** DP-FedAvg pilot on the **validation** split: epsilon 3 (delta = 1/N per client, budget 40 rounds), alpha 0.5, seed 42, E = 5, lr 0.2, expected batch 256, C in {0.5, 1, 2, 4}; C = 8 added because 0.5 -> 2 kept improving.
- **What we found:**

  | C | best val macro-F1 | best round | final client loss |
  |---|---|---|---|
  | 0.5 | 0.212 | 39 | 1.56 |
  | 1 | 0.223 | 34 | 1.15 |
  | **2** | **0.264** | 37 | 0.96 |
  | 4 | 0.228 | 33 | - |
  | 8 | 0.215 | 24 | - |

  Interior optimum at C = 2, inside the median per-example gradient norm during training (~1-4, F30). Single seed, so differences of ~0.04 are noisy. With noise multipliers sigma = 2.4-6.1 (small clients need more noise), the **aggregated update has length ~2 in every round - exactly the predicted combined noise sqrt(sum w_k^2 r_k^2)** - so the global model moves mostly by noise; the same seed without DP reaches ~0.57 (B1). A signal-to-noise argument explains why tuning cannot rescue it: when sigma is large, the total SNR is roughly epsilon * N * (g/C) / sqrt(d), almost independent of batch size, local epochs and rounds; only the data per client N (2k-19k rows here), the model size d (40k) and how well clipped gradients agree matter.
- **Evidence:** `results/runs/p5_clip_pilot_*` (not committed), `python scripts/compare_runs.py --glob "p5_clip_pilot_*"`. Due to F34 the pilot's sigma was calibrated with a slightly wrong sampling rate; recomputed with Opacus' true sampling, the pilot runs spent epsilon **2.84-3.05** per client (target 3), so the comparison is valid.
- **Decision / impact:** **C = 2** for every DP run (`configs/b2_dpfedavg.yaml`). The cost of privacy will be large; it is reported as it is (B2, F38). Expected consequence for Phases 6-7: honest DP updates are dominated by noise (radius ~13-15 per round vs a clean update of ~1.5), which is exactly the regime where AutoGM should wrongly reject honest clients - and where FedGuard's residual is hardest to estimate.

### F34
**Experiment E0 caught a bug: the declared noise radius was 5-20% too small, because Opacus samples with rate 1/ceil(n/B), not B/n.**
- **What we did:** E0 (`scripts/e0_calibration.py`): train one client twice from the same global model with the same batches and dropout - once with sigma, once with sigma = 0 - and compare ||delta_noisy - delta_clean|| with the declared radius lr*sigma*C*sqrt(T*d)/B.
- **What we found:** Measured/predicted was **1.089** for a client with 3,649 rows and **1.186** for one with 1,856 rows, **identical for sigma = 0.5, 1 and 2** - so the noise is exactly first-order (the formula's shape is right), but T and B were wrong. Opacus' `make_private` sets the sample rate to 1 / len(DataLoader) = **1/ceil(n/B)**, takes **int(1/q)** Poisson batches per epoch (its own float arithmetic gives k-1 for some k, e.g. 93), and divides the noisy sum by **int(n / batches)**. We had used q = B/n, floor(n/B) steps and B = 256. Check: sqrt(75/70) x 256/243 = 1.089 and sqrt(40/35) x 256/232 = 1.18. The privacy accounting used the same too-small q and step count; recomputed, the effect on epsilon is small (pilot: 2.84-3.05 instead of 3.00).
- **Evidence:** `logs/p5_e0_before_fix_F34.log` (not committed; numbers above), Opacus `privacy_engine.make_private` and `data_loader.DPDataLoader.from_data_loader`.
- **Decision / impact:** `fedguard.dp.poisson_params` now mirrors Opacus exactly (q, batches per epoch, expected batch) and is used for sigma calibration, epsilon accounting and the declared radius (engine and Flower client). New tests: our numbers equal what Opacus' loader and optimizer actually use (including the k = 93 float case), and the radius matches the measured noise over several Poisson steps. B2 was started only after the fix. **Lesson for the report:** E0 is not a formality - the server-side formula is only as good as the declared metadata, and a silent 5-20% under-statement would have made honest small clients look suspicious to FedGuard.

### F35
**E0: after the F34 fix, the server's noise-radius formula predicts the real DP-SGD noise to within about 1% (kappa = 1.00), and the noise is orthogonal to the clean update.**
- **What we did:** `scripts/e0_calibration.py` with C = 2, lr 0.2, E = 5: each of the 5 clients (alpha 0.5, seed 42 partition; 1,856-18,953 rows, T = 40-375 local steps) trained from the same global model with sigma in {0.5, 1, 2} and with sigma = 0 (same Poisson batches and dropout), 5 seeds each, from two start models: random initialisation and the global model after 5 FedAvg rounds. 150 pairs.
- **What we found (plan Table 10.1):**

  | start | sigma | predicted r (mean) | measured (mean) | ratio mean +/- std | ratio min-max | pairs |
  |---|---|---|---|---|---|---|
  | init | 0.5 | 2.00 | 2.01 | 1.005 +/- 0.005 | 0.994-1.015 | 25 |
  | init | 1.0 | 3.99 | 4.02 | 1.004 +/- 0.005 | 0.994-1.013 | 25 |
  | init | 2.0 | 7.99 | 8.02 | 1.004 +/- 0.005 | 0.994-1.012 | 25 |
  | warm | 0.5 | 2.00 | 2.00 | 1.000 +/- 0.005 | 0.993-1.008 | 25 |
  | warm | 1.0 | 3.99 | 3.99 | 1.000 +/- 0.005 | 0.993-1.008 | 25 |
  | warm | 2.0 | 7.99 | 7.99 | 1.000 +/- 0.004 | 0.993-1.008 | 25 |

  - Least-squares **kappa = 1.005 (init) and 0.999 (warm)**: no correction factor needed. The ratio does not depend on sigma or on T (40-375 steps), so the first-order formula holds even over hundreds of noisy steps; the spread (std 0.0045) is close to the theoretical wobble of a Gaussian noise length, 1/sqrt(2d) = 0.0035.
  - **cos(noise, clean update) = -0.002 to -0.008**: the DP noise is almost exactly at 90 degrees to the signal, which experimentally supports FedGuard's quadrature refinement R2 (dist^2 = signal^2 + noise^2).
  - Signal-to-noise per round (||clean update|| / r): 1.35 at sigma 0.5 and 0.34 at sigma 2 from initialisation, only **0.55 / 0.14 from the warm model** - with the sigma = 2.4-6 that epsilon 3 needs (F33), an honest client's update is >90% noise by length.
- **Evidence:** `results/e0_calibration/e0_table.csv`, `e0_pairs.csv`, `results/figures/e0_calibration.png`, `logs/p5_e0.log`.
- **Decision / impact:** FedGuard can use the declared radius r = lr*sigma*C*sqrt(T*d)/B as it is (kappa = 1), provided T and B are Opacus' real values (F34). The foundation of the hypothesis holds on real IDS training. The tiny relative wobble (~0.5%) of a large radius is what R4 is meant to forgive (F02, F31): at r ~ 14 it is ~0.07 in length, comparable to small real residuals.

### F36
**With one shared epsilon, every client's declared noise radius is almost the same (~12-15 per round at epsilon 3), whatever its data size - so the plan's uniform-epsilon grids hardly test FedGuard's main idea.**
- **What we did:** Computed sigma and the declared radius r for the real client sizes of the K = 10 partitions (215 to 18,456 rows) with the B2 settings (E = 5, 40 rounds, C = 2, lr 0.2, delta = 1/N).
- **What we found:**

  | epsilon | n = 215 | n = 1,210 | n = 3,345 | n = 10,976 | n = 18,456 |
  |---|---|---|---|---|---|
  | 8 | sigma 6.7, r 5.6 | 3.4, 5.7 | 2.2, 6.3 | 1.4, 6.7 | 1.2, 7.3 |
  | 3 | 14.3, 11.9 | 7.4, 12.3 | 4.8, 13.5 | 3.0, 13.8 | 2.4, 14.6 |
  | 1 | 34.7, 29.0 | 18.8, 31.1 | 12.3, 34.8 | 7.7, 35.5 | 6.1, 37.3 |

  Small clients need a much larger sigma, but they also take fewer steps per round, and the two effects cancel: when sigma is large, sigma ~ q*sqrt(steps) and r ~ sigma*sqrt(T)/B, so n drops out and r depends only on epsilon, E, R, lr, C and d (+/- 15-25% left).
- **Decision / impact:** In uniform-epsilon experiments AutoGM's honest-rejection problem comes from non-IID spread and from every client being noisy - not from some clients being more private than others - and FedGuard mostly subtracts a common constant. Its distinctive idea (forgive each client exactly its own declared noise) is tested only when **privacy levels differ between clients**, as in the paper's hospital-and-bank story. The engine now supports per-client epsilon (`dp.epsilon: [1, 3, 8]` cycles over clients; run names `eps1-3-8`; unit-tested). Proposal for Phases 6-8: keep the plan's uniform-epsilon grids and add a **mixed-privacy scenario** (epsilon {1, 3, 8} across clients) to E4 (HRR) and to the main attack table.

### F37
**Our AutoGM gives the same weights and centre as the AutoGM in the Blades benchmark (by AutoGM's first author), to about 1e-11.**
- **What we did:** Ported Blades v0.1.0's `Autogm` and `Geomed` (`src/blades/aggregators/autogm.py`, `geomed.py`, Apache-2.0) line by line to NumPy and ran both on the same synthetic rounds (9 clients in 2,000 dimensions: 7 honest with different noise levels and non-IID spread, 2 sign-flip attackers), with the same absolute lambda (Blades uses a fixed lambda; ours is lambda_scale x median distance, so `autogm` got an optional `lam` parameter for this check).
- **What we found:** Maximum weight difference **1.4e-11 to 3.9e-11**, relative centre difference **1.8e-11 to 4.2e-11** in all 4 trials; both give the attackers zero weight. Blades' weight step (sort distances, find eta, a_i = max(eta - d_i, 0)/lambda) is algebraically the same as our simplex projection of -d/lambda. Its Weiszfeld loop re-uses the previous iteration's normalised weights instead of the fixed alphas, which looks non-standard, but changes the result by only ~4e-7 here. Side observation: with lambda = median distance, AutoGM keeps only 2-3 of the 7 honest clients in every trial.
- **Evidence:** `tests/test_blades_crosscheck.py` (permanent test, 4 trials).
- **Decision / impact:** The AutoGM baseline is a faithful implementation of the reference. Phase 6 still tunes its lambda_scale on validation (plan 11.2) so FedGuard is compared with AutoGM at its best.

### F38
**B2: DP-SGD costs about half of the macro-F1 even at epsilon = 8; under DP the model keeps only Benign, Exploits and Fuzzers, while attack-vs-benign detection mostly survives.**
- **What we did:** DP-FedAvg, 5 clients, alpha 0.5, E = 5, C = 2, 40-round privacy budget, epsilon in {inf, 8, 3, 1} (delta = 1/N per client) x seeds {42, 43, 44}; best round on validation, test once.
- **What we found (test, mean +/- std over 3 seeds):**

  | epsilon | macro-F1 | binary F1 | benign FPR | detection | sigma (range over clients) | epsilon spent |
  |---|---|---|---|---|---|---|
  | inf (no DP) | **0.513 +/- 0.056** | 0.904 | 0.058 | 0.919 | - | - |
  | 8 | **0.253 +/- 0.036** | 0.851 | 0.095 | 0.880 | 1.2-2.5 | 8.00 |
  | 3 | **0.219 +/- 0.029** | 0.853 | 0.114 | 0.911 | 2.4-5.4 | 2.69-3.00 |
  | 1 | **0.183 +/- 0.028** | 0.800 | 0.110 | 0.812 | 6.2-13.8 | 1.00 |

  - Per-class recall (mean): with DP, **Analysis, Backdoor, DoS, Generic, Reconnaissance, Shellcode and Worms all fall to ~0** (except Analysis 0.21 at epsilon 8); Benign stays ~0.89, Exploits ~0.86-0.94, Fuzzers 0.65 -> 0.22 as epsilon shrinks. So DP removes the ability to tell **which** attack it is, and roughly 5-10 points of attack-vs-benign quality.
  - The largest drop is from no DP to epsilon 8; after that the curve is flat-ish. The accountant spends exactly the budget over 40 rounds (one epsilon-3 run stopped early at round 33 and spent 2.69).
  - The no-DP run with the same 40-round budget (0.513) is close to B1 with 120 rounds (0.537) - the budget, not the round limit, causes the loss.
  - Why so costly: noise dominates every update (radius ~13-15 per round vs a clean update of ~1.5, F35), and rare classes have only tens to hundreds of rows per client, so their gradient signal is the first to drown (clipping also caps their large per-example gradients, F30). This matches F33's signal-to-noise argument (few rows per client, 40k parameters).
- **Evidence:** `results/b2_dpfedavg/summary_by_epsilon.csv`, `per_class_recall_by_epsilon.csv`, figures `results/figures/b2_dpfedavg_by_epsilon.png`, `b2_dpfedavg_rounds.png`, `b2_dpfedavg_per_class_recall.png`, `results/master.csv`.
- **Decision / impact:** Reported as it is - the honest cost of record-level DP with small clients. No "noise as a regulariser" effect (plan expert tip): stronger privacy was never better here. For Phases 6-8 this sets the bar: with DP on, the robust aggregators are compared on a model that mainly separates benign / Exploits / Fuzzers, so **attack-vs-benign metrics (binary F1, benign FPR) and macro-F1 must both be reported**. Option for the report's future work: larger clients, a smaller model, or client-level DP would raise utility.

---

## Phase 6 - Attacks, AutoGM (B3, B4) and the O2 conflict study

### F39
**Under DP noise AutoGM has no good lambda: small lambda collapses onto a single client, large lambda lets sign-flip attackers keep about a third of the weight. lambda_scale = 4 chosen on validation.**
- **What we did:** AutoGM with lambda = lambda_scale x median client distance, lambda_scale in {0.5, 1, 2, 4, 8} (8 added after 0.5-2 collapsed), K = 10, alpha 0.5, epsilon 3 (C = 2), seed 42, without attack and with 30% sign-flip attackers (smart: they run DP-SGD and declare a normal sigma); judged on **validation** macro-F1 averaged over both scenarios.
- **What we found:**

  | lambda_scale | no attack: HRR | no attack: val macro-F1 | sign flip: attackers' weight | sign flip: val macro-F1 | mean val |
  |---|---|---|---|---|---|
  | 0.5 | **0.90** | 0.048 | 0.00 | 0.048 | 0.048 |
  | 1 | **0.90** | 0.048 | 0.015 | 0.048 | 0.048 |
  | 2 | 0.10 | 0.200 | **0.365** | 0.057 | 0.128 |
  | **4** | 0.00 | **0.207** | **0.325** | 0.148 | **0.178** |
  | 8 | 0.00 | 0.204 | 0.310 | 0.144 | 0.174 |

  - **Collapse:** with lambda <= 1x the median distance, AutoGM gives one client weight 1 in every round (weights one-hot, the centre = that client's update). This is the true optimum of AutoGM's objective once lambda is small compared with the distances: the chosen point has distance 0, so it takes all the weight, and the choice reinforces itself. Under DP each update is ~90% noise (length ~13, F35), so the global model receives one client's full noise each round and degenerates (benign FPR 0.999 - everything flagged as an attack). Checked by replaying one round offline: weights [0,0,0,0,1,0,0,0,0,0], centre distance to that client 0.
  - **Attackers get in:** with lambda >= 2x, AutoGM spreads the weight, but the three sign-flip attackers keep 31-37% of it (their fair share is 30%). A sign-flipped DP update is as far from the centre as an honest one: its extra displacement (~2x the signal, length ~1-1.5) is tiny next to the noise radius (~13) and next to the +/-10% spread of radii between honest clients.
  - This is the paper's dilemma (Gap 1) measured on real IDS training: no lambda both keeps honest DP clients and excludes attackers. The 4x setting is AutoGM at its best and is used for B3, B4, E4 and E5.
- **Evidence:** `results/runs/p6_autogm_lambda_*` (not committed), `results/master.csv` rows `p6_autogm_lambda`, `python scripts/compare_runs.py --glob "p6_autogm_lambda_*"`.
- **Decision / impact:** AutoGM lambda_scale = **4** frozen. A quick synthetic check at the same scale (d = 40,266, radii 12.3-14.6, signal ~1) suggests FedGuard will help only partly at epsilon 3 (attackers' share 0.22 instead of 0.29): per round, the attacker's extra squared distance (~4|s|^2) is only 2-3x the noise-length wobble. Phase 7 measures this on real data; it may define FedGuard's breaking point.

### F40
**Fast plan: at the team's request the remaining experiments were cut from ~340 runs (~6 days of GPU) to ~140 runs (~2-2.5 days).**
- **What we did:** Measured cost: a 40-round DP run ~55 min, a plain run ~20 min, 2 in parallel (the GPU is at 93-96% and 86 C with two; a third does not add throughput). The full Phase 6-8 matrix needed ~145 GPU hours.
- **Decision / impact (deviations from the plan, to be stated in the report):**
  - **E5 main table:** 4 methods x 3 attacks x **30% attackers only** x 3 seeds (plan: 10/20/30/40%); the attacker-ratio curve uses sign flip at 10/20/40% with seed 42 only.
  - **E4 conflict:** epsilon {inf, 3, 1, **mixed 1-3-8**} x alpha {0.1, 0.5, 100} x **2 seeds** (plan: epsilon {inf, 8, 3, 1} x 3 seeds). Epsilon 8 dropped because B2 showed it behaves like epsilon 3 (F38); the mixed-privacy column is added (F36).
  - **B3 (AutoGM without DP):** seed 42 only. **E6/E7:** reduced (one change at a time; stress at epsilon 8 and 1). **E8 (CICIoT2023, optional):** dropped - future work.
  - All K = 10 experiments share one run name (`k10`), so a configuration that belongs to several experiments is trained once; `run_experiments.py --sweep a.yaml b.yaml ...` runs several sweeps as one de-duplicated queue.

### F41
**A burst of AC-power changes made the NVIDIA driver fail; both training jobs hung silently for about 3 h 50 min.**
- **What we did:** The run logs stopped at 19:03 while both Python processes stayed alive (still using CPU, GPU shown at 100%). Checked the Windows System event log.
- **What we found:** At 19:04 Windows logged **8 power-source changes in 14 seconds** (Kernel-Power event 105 - charger connecting/disconnecting, e.g. a loose plug), immediately followed by a continuous flood of NVIDIA driver errors (`nvlddmkm` event 13, **25,000 events** by 22:51). The CUDA calls of both jobs never returned, so nothing crashed and no error reached the logs.
- **Decision / impact:** Killed the hung jobs; a CUDA test and a DP smoke round worked again; the queue was restarted (finished runs are skipped, only the 2 interrupted runs restart). Added a **watchdog**: if a shard log is not updated for 12 minutes (a round normally logs every 30-80 s), it alerts so the job can be restarted. Also added **per-round checkpoints** (`results/runs/<run>/checkpoint.pt`, written atomically; a stopped or crashed run resumes from its last round and gives a bit-identical result - tested with and without DP) and a **clean pause**: creating `results/runs/PAUSE` makes every sweep stop after its current round, so the laptop can be closed safely; deleting it and restarting the sweep continues. Team: keep the charger firmly connected, disable sleep on AC, and reboot when possible to clear the driver state. Lesson: on a laptop, long GPU jobs need a stall detector and checkpoints, not just crash handling.

