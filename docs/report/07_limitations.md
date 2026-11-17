# Chapter 7 - Limitations and future work

> DRAFT (18 Nov 2026). The breaking points (E7) and the final FedGuard-vs-AutoGM verdict are added when the
> experiments finish **[pending]**.

## 7.1 Limitations

**Honest declarations are assumed.** FedGuard forgives exactly the noise a client says it added. An attacker
that declares a large sigma is forgiven more. In our experiments attackers declare honest-looking settings and
also run DP-SGD, which is the threat model of the paper, but a lying attacker is not evaluated. A partial check
exists (`too_clean`: in high dimension the noise length concentrates, so an update much shorter than its
declared radius is suspicious) and is left as future work.

**Small clients and a high cost of privacy.** With 1-19k records per client and a 40k-parameter model,
record-level DP at epsilon <= 8 roughly halves macro-F1, and the rare attack classes (Analysis, Backdoor, DoS,
Generic, Reconnaissance, Shellcode, Worms) are not learned at all under DP (F38). Robust aggregators are
therefore compared on models that mostly separate Benign, Exploits and Fuzzers; binary attack-vs-benign metrics
are reported alongside macro-F1 for this reason.

**Detecting attackers inside the noise.** At epsilon 3 a sign-flipped update moves about 1-1.5 further than an
honest one, while every update carries noise of length ~13 that wobbles by ~0.07 (F35, F39). Distance-based
methods - AutoGM and FedGuard alike - can only see what sticks out of that noise. Label flipping does not stick
out at all, because the attacker trains normally on wrong labels.

**One dataset.** All results are on NF-UNSW-NB15-v2 after de-duplication (104,022 unique flows). The second
dataset (CICIoT2023, plan E8) was dropped for time. The de-duplicated protocol also makes our numbers lower than,
and not directly comparable with, papers that keep the 95.5% duplicate rows (F13).

**Reduced grids.** To fit one laptop: one attacker share (30%) in the main table, the attacker-share curve and
the epsilon stress test with one seed, E4 with two seeds (F40). Single-seed results are read as trends only.

**Simulation.** Experiments run in our own engine; Flower is used for the parity check (agreement 6.9e-17,
F26) and for the live demo. Network effects (stragglers, dropped clients, partial participation) are not
modelled: all clients take part in every round.

**Hyper-parameters chosen on one setting.** AutoGM's lambda and FedGuard's settings were chosen on the
validation split at alpha 0.5, epsilon 3, seed 42 (F39, F50) and then frozen. Other settings (alpha 0.1, mixed
privacy) might favour other choices; E6 shows how sensitive each choice is **[pending]**.

**Rare-class scores are noisy.** Some classes have 4-27 validation and 10-68 test rows (F12), so their
per-class recall moves in large steps.

## 7.2 Future work

1. **Verify declared noise** - turn `too_clean` into part of the trust rule and test attackers who lie about
   their noise, including adaptive attackers who know FedGuard's rule (paper section 7).
2. **Raise utility under DP** - larger clients, client-level instead of record-level DP, or a smaller model,
   so that rare classes survive; then re-test whether robust aggregation hurts rare classes (paper Gap 3).
3. **Second dataset** - CICIoT2023 with an 8-class grouping, after a cross-file duplicate check (F15).
4. **Partial participation and real deployment** - Flower across organisations with stragglers, secure
   aggregation and authenticated clients.
5. **Learned allowance** - the allowance s (median + k x MAD of residuals) was the fragile part throughout
   (F31, F48); a per-client estimate of normal non-IID spread from past rounds could replace it.
