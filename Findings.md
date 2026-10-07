# FedGuard - Findings log

Every finding of the project, in the order we found it: **what we did, what we found, the evidence,
and what we decided**. This file feeds the report (Methodology, Results, Limitations) and the viva.
It is updated after every phase.

**Status:** Phase 1 and Phase 2 complete (last updated 7 Oct 2026).

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
