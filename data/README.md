# Datasets

Raw and processed data are **never committed** to Git (`data/raw/` and `data/processed/` are ignored).
Both datasets were placed in `data/raw/` on 7 Oct 2026.

## 1. NF-UNSW-NB15-v2 (primary dataset)

| Item | Value |
|---|---|
| Source | University of Queensland, UQ eSpace: https://espace.library.uq.edu.au/view/UQ:ffbb0c1 (licence details there). Creator: Mohanad Sarhan. Bagged 2023-05-11. |
| Cite as | Sarhan, Layeghy, Portmann, "Towards a standard feature set for network intrusion detection system datasets", Mobile Networks and Applications (2022) - check the exact citation on the UQ page |
| Location | `data/raw/NF-UNSW-NB15-v2/data/NF-UNSW-NB15-v2.csv` (BagIt bundle, metadata files kept) |
| Size | 441,867,785 bytes (421.4 MB) |
| SHA-1 | `dde9a9e38009ec60121cbb4fd884bf2c7cf5c0b0` - **matches the bundle's manifest-sha1.txt** |
| SHA-256 | `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f` |
| Rows / columns | 2,390,275 rows; 45 columns = 43 NetFlow features + `Label` (binary) + `Attack` (class name) |
| Feature list | `data/raw/NF-UNSW-NB15-v2/data/NetFlow_v2_Features.csv` |

Class counts (`Attack` column):

| Class | Rows | Share |
|---|---|---|
| Benign | 2,295,222 | 96.02% |
| Exploits | 31,551 | 1.32% |
| Fuzzers | 22,310 | 0.93% |
| Generic | 16,560 | 0.69% |
| Reconnaissance | 12,779 | 0.53% |
| DoS | 5,794 | 0.24% |
| Analysis | 2,299 | 0.10% |
| Backdoor | 2,169 | 0.09% |
| Shellcode | 1,427 | 0.06% |
| Worms | 164 | 0.01% |

Phase 2 notes: identifier columns to drop are `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`,
`L4_DST_PORT` (also check `DNS_QUERY_ID`, which is a random ID). `Worms` has only 164 rows, so
some clients will get very few or none after the Dirichlet split - handle this in Phase 2.

## 2. CICIoT2023 (secondary dataset, Phase 8 / E8)

| Item | Value |
|---|---|
| Original source | Canadian Institute for Cybersecurity, UNB: https://www.unb.ca/cic/datasets/iotdataset-2023.html. Paper ref [21] (Neto et al., Sensors 2023). |
| This copy | A **pre-split subset** (train / validation / test CSVs) from Kaggle: https://www.kaggle.com/datasets/himadri07/ciciot2023 (downloaded Oct 2026). Cite the original CIC dataset/paper; mention this Kaggle subset in the report. |
| Location | `data/raw/CICIoT2023/{train,validation,test}/*.csv` |
| Rows | train 5,491,971 · validation 1,176,851 · test 1,176,851 (total 7,845,673; the full original is ~46 M) |
| Columns | 46 features + `label`; 34 classes (33 attacks + `BenignTraffic`) |
| SHA-256 train | `ede08dc57e57369fcf35a2b1ea81b81bad0a9a2ad91bc08eceb646bd89117b04` |
| SHA-256 validation | `b13775e87ef64c8dd66c910d4585323f99b1209f2508c75a0a0a8d584caf20d8` |
| SHA-256 test | `15aeb1f47a7bd3e3fc394d18d7b196b3640329abaae65d5220df4def6119bfc6` |

Phase 8 notes: group the 34 labels into 8 classes (Benign + DDoS, DoS, Mirai, Recon, Spoofing,
Web, BruteForce). The split was made by a third party, so check duplicates *across* the three
files before trusting it; if any are found, merge, de-duplicate and re-split with our own pipeline.
