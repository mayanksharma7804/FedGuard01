# Datasets

Raw and processed data are **never committed** to Git (`data/raw/` and `data/processed/` are ignored).
Fill in this table when the dataset is downloaded (Phase 2).

| Dataset | Source URL | Downloaded on | File name | Size | SHA-256 |
|---|---|---|---|---|---|
| NF-UNSW-NB15-v2 | University of Queensland "NIDS datasets" page | | | | |
| CICIoT2023 (optional) | UNB CIC datasets page | | | | |

Get the hash on Windows with:

```powershell
Get-FileHash data\raw\<file>.csv -Algorithm SHA256
```
