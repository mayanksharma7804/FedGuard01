# Chapters 1 and 3 - what must change from the September 2026 paper

> DRAFT (18 Nov 2026). Chapters 1-3 of the report reuse the team's paper ("FedGuard: A Scoping Review and
> Proposed Framework...", Sept 2026). The paper is a review that "presents no implementation" and proposes
> a design; the report describes the built system. These are the passages that are now wrong or incomplete,
> with replacement text to paraphrase. Everything else in paper sections 1-4 can stay.

## Section 1.5 (contributions) - replace the first paragraph

The paper says it "presents no implementation". Replacement:

> This report goes from the review to a working, measured system. Beyond the review of 25 studies and the four
> comparison matrices (Chapter 2), it contributes: (i) a leakage-checked federated intrusion-detection pipeline on
> NF-UNSW-NB15-v2, which shows that 95.5% of the dataset's rows are duplicates once identifiers are removed;
> (ii) an experimental check that the noise a DP-SGD client adds to its update can be predicted from its declared
> settings to within about 1% (experiment E0); (iii) the first measurement on intrusion-detection data of how a
> robust aggregator (AutoGM) treats honest clients that use differential privacy; and (iv) FedGuard, a noise-aware
> extension of AutoGM, evaluated under differential privacy, non-IID data and three attacks, with every
> negative result reported.

## Section 6.2 (aggregation tier) - corrections

1. **Noise radius.** The paper writes r_i = sigma_i * sqrt(d). DP-SGD adds noise at every step, averaged over the
   batch and scaled by the learning rate, so the radius of one round's update is
   r_i = lr * sigma_i * C * sqrt(T_i * d) / B_i (T = local steps, B = expected batch size). Experiment E0 confirmed
   it with measured/predicted = 1.000-1.005 (Findings F34, F35).
2. **Clients declare** (sigma, C, B, T, lr, n) - not epsilon - because the radius needs these; epsilon is reported.
3. **Allowance s.** The paper takes the MAD of raw inter-client distances. Raw distances contain the noise, which
   inflates s and lets attackers in, so s is computed from the residuals left after the noise is removed (R3), the
   centre's own noise is counted (R1), the residual uses the quadrature rule sqrt(dist^2 - r_eff^2) because the
   noise is orthogonal to the signal (R2; the paper's linear rule dist - r is kept as an ablation), a tolerance for
   the natural wobble of the noise length can be added (R4), and s has a floor so it cannot collapse to zero
   when most residuals are hidden by noise (F48). Settings chosen on validation (F50).
4. **"Trust is continuous ... there is no instant drop-out."** Not true of the implemented rule: like AutoGM, trust
   is a simplex projection, and clients beyond a threshold get weight exactly 0. Replace with: "weights fall
   continuously with the unexplained excess and reach zero beyond a data-driven threshold, as in AutoGM."
5. **Clients**: "a hospital, a bank and a retailer" (three) became K = 5 (FedAvg, DP baselines) and K = 10 (all
   robustness experiments), so that 30% attackers is three clients (plan decision D3).

## Section 6.3 (objectives) - small changes

- O1: "three or more simulated organisational clients" -> 5 and 10 clients.
- O4: "at varying compromise ratios" -> 30% in the main table, 10-40% in the stress test (one seed, F40).
  The target "robustness comparable to AutoGM without DP at accuracy comparable to DP-FL" is answered directly
  in Chapter 6 **[pending]**.

## Section 6.4 (evaluation plan) - replace

> Datasets: NF-UNSW-NB15-v2 only (CICIDS2017 dropped because of its known labelling problems; CICIoT2023 left
> as future work for time). Non-IID partitioning: Dirichlet allocation over classes with alpha in {0.1, 0.5, 100}.
> Baselines: centralised CNN-LSTM (ceiling), FedAvg, DP-FedAvg, AutoGM without and with DP, and FedGuard with
> DP. Implementation: Python and PyTorch with Opacus for DP-SGD; experiments in our own deterministic federated
> engine (Flower's simulation mode relies on Ray, which is unreliable on Windows), with Flower used for an
> engine-parity check (agreement 6.9e-17) and for a multi-laptop live demo, which replaces Docker (Windows 11
> Home runs Docker only through WSL2).

## Section 6.5 (scope) - corrections

- "three to five client nodes" -> five and ten.
- "stratified subsamples" -> the full de-duplicated dataset (104,022 rows) with a 20k-per-class training cap.
- "a straightforward moments-accountant implementation" -> Opacus' RDP accountant, per client, for the full
  40-round budget.
- Add to "out of scope": attackers who lie about their declared noise (the main threat to FedGuard's
  assumption; partial check `too_clean` left as future work).

## Decisions D1-D7 (plan section 3) - to state in the methodology

| | Decision | What was done |
|---|---|---|
| D1 | Flower for experiments? | Own engine for experiments; Flower for parity (6.9e-17) and the demo |
| D2 | Docker? | Replaced by a multi-laptop Flower demo with a one-laptop fallback |
| D3 | 10 clients for robustness? | Yes, K = 10 for every robustness experiment |
| D4 | Adopt R1-R3, test R4? | Adopted; R4 and the s floor tested, settings chosen on validation (F50) |
| D5 | CICIDS2017? | Dropped |
| D6 | Friend's Linux laptop? | Not needed; everything ran on one Windows laptop |
| D7 | AI-assisted code and writing | Declare as the college requires (Claude Code was used for code and drafts) |
