# FedGuard - Findings log

Every finding of the project, in the order we found it: **what we did, what we found, the evidence,
and what we decided**. This file feeds the report (Methodology, Results, Limitations) and the viva.
It is updated after every phase.

**Status:** Phases 1-4 complete (last updated 8 Oct 2026).

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

