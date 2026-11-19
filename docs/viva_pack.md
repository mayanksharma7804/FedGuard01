# Viva pack - questions and model answers

> DRAFT (19 Nov 2026). Every member should be able to give each answer in their own words in about a minute
> (plan 13.6 and Appendix E). Numbers cite Findings.md (Fxx). Updated 19 Nov with the final results (F50-F58).

## The idea

**1. FL already keeps data private, so why add DP?**
Federated learning keeps raw logs on site, but the updates themselves leak: gradient-inversion attacks ("Deep
Leakage from Gradients", ref [8]) rebuild training records from a client's update. DP-SGD clips each record's
influence and adds calibrated noise, so what any single record contributes is provably bounded (epsilon, delta).

**2. What does epsilon = 3 mean? What is delta?**
For any two datasets differing in one record, the probability of any output changes by at most a factor
e^epsilon (e^3 ~ 20), except with probability delta. We use delta = 1/N per client (N = its records). Smaller
epsilon = more noise = more privacy. Epsilon is accounted with Opacus' RDP accountant for the whole 40-round
budget, per client (F32, F38).

**3. Why do DP and robust aggregation fight? (the paper's Gap 1)**
A robust aggregator trusts updates close to the crowd. DP noise pushes every honest update away from the crowd
(at epsilon 3 an honest update is >90% noise by length, F35), and non-IID data spreads them further, so the
aggregator either rejects honest private clients or widens its tolerance until attackers fit in. Measured:
AutoGM under DP has no good lambda - small lambda collapses onto one client (HRR 0.90), large lambda lets
sign-flip attackers keep ~1/3 of the weight (F39); under mixed privacy it gives weight 0 to every epsilon-1
client (F47).

**4. What is FedGuard in one sentence?**
AutoGM in which each client declares its DP settings, the server predicts how far that noise should push the
client's update, forgives exactly that much, and distrusts only the displacement that noise and normal non-IID
spread cannot explain.

## The maths

**5. Derive the noise radius. Why sqrt(T), and why divide by B?**
DP-SGD step: g = (sum of clipped per-example gradients + N(0, sigma^2 C^2 I_d)) / B, and the weights move by
-lr g. So each step adds lr * N(0, (sigma C / B)^2 I_d). Over T steps the noise vectors are independent; their
sum has variance T (lr sigma C / B)^2 per coordinate, and in d dimensions its length concentrates at
sqrt(d) times the per-coordinate std: r = lr sigma C sqrt(T d) / B. sqrt(T) because independent steps partly
cancel (a random walk); 1/B because the noise is added to a sum that is then averaged over the batch. E0
measured/predicted = 1.000-1.005 over 150 pairs (F35) - after we caught that Opacus' real B and T differ from
the textbook ones (F34).

**6. Why is the quadrature rule sqrt(dist^2 - r^2) justified?**
In high dimension a random Gaussian vector is almost exactly orthogonal to any fixed vector, so
||signal + noise||^2 ~ ||signal||^2 + ||noise||^2 (Pythagoras). We measured cos(noise, clean update) =
-0.002 to -0.008 (F35).

**7. Geometric median vs normal median? How does Weiszfeld work?**
The geometric median minimises the sum of Euclidean distances to the points (the multivariate version of the
median); one extreme point cannot drag it far. Weiszfeld's algorithm repeatedly takes a weighted average of the
points with weights 1/distance, so far points pull less; it converges to the geometric median.

**8. What do R1-R4 and the s floor do?**
R1: the centre is built from noisy updates too, so the noise in (x_k - z) is (1-b_k)^2 r_k^2 + sum b_j^2 r_j^2.
R2: quadrature residual. R3: allowance s = median + 2 x MAD of the residuals (robust spread). R4: forgive the
natural wobble of the noise length (std ~ r/sqrt(2d)), which otherwise gets a very private client rejected when
the others are quiet (F02, F31). s floor: under DP most residuals are hidden by noise, so median = MAD = 0 and s
collapses, making the trust step a cliff (F48). Which of these to use was decided on the validation split (F50)
and each is ablated (E6, F57): none changes macro-F1 beyond the seed spread; the s floor and R4 lower honest rejection; without R1 FedGuard degenerates to equal weights.

**9. What is R4 and why is it a trade-off?**
It widens tolerance by z standard deviations of the noise length. That saves honest private clients, but an
attacker whose extra displacement is smaller than that tolerance is forgiven too. Synthetic check:
quadrature + R4 let attackers take 6-15% of the weight at the highest noise (F02).

## Data and evaluation

**10. How did you prevent leakage?**
Dropped identifiers (IPs, ports and the hidden identifier DNS_QUERY_ID, F08); 95.5% of rows were duplicates and
were removed before the split (F07); log1p as float32 before de-duplication (a test caught a float collision,
F11); scaler fitted on train only; tests check that no row is shared between splits.

**11. Why macro-F1 and not accuracy?**
Benign is 66% of the test set; a model that predicts only Benign gets 66% accuracy and macro-F1 0.08 (this is
exactly what DP models do in their first ~8 rounds, F46). Macro-F1 weights the rare attacks equally. We also
report binary F1, benign FPR and per-class recall.

**12. Why Dirichlet for non-IID? What does alpha mean?**
Each client's class mix is drawn from Dirichlet(alpha): alpha = 100 gives near-identical mixes (IID), alpha = 0.1
gives clients that see only a few classes. It is the standard way to control label skew with one number.

**13. Why are your scores lower than published ones (91-93%)?**
They do not de-duplicate (95.5% duplicates means test rows are copies of training rows) and many report accuracy.
Our accuracy (0.90) is close to theirs; our macro-F1 (0.60 centralised) shows the rare-class weakness (F13, F22).

**14. Why is DP so costly here?**
Clients hold 1-19k rows and the model has 40k parameters; the per-round signal-to-noise ratio is roughly
epsilon x N x (g/C) / sqrt(d). Rare classes have only tens of rows per client, so their signal drowns first:
B2 macro-F1 0.51 -> 0.25 at epsilon 8 (F33, F38).

## Engine and implementation

**15. Why your own simulation engine? How do you know it is correct?**
Flower's simulation needs Ray (unreliable on Windows) and we needed ~300 exactly repeatable runs. Correctness:
the same FedAvg in Flower deployment mode and in our engine agree to 6.9e-17 (F26) - and that check found a real
bug first (clients trained sequentially, F24). Our AutoGM equals the Blades reference to 1e-11 (F37).

**16. How would this run in a real consortium?**
Each organisation runs a Flower SuperNode with its own data and privacy settings; the coordinator runs the
SuperLink with the FedGuard strategy (our live demo, docs/demo.md). Missing for production: secure aggregation,
authenticated clients, and checks that declared noise is real (next question).

## Limits

**17. What if an attacker lies and declares a lot of noise?**
Then FedGuard forgives more for that client - the main weakness (paper section 7). Partial defence: in high
dimension the noise length concentrates tightly, so a client that declares a large radius but sends an update
much shorter than it is suspicious (`too_clean`, implemented as future work). Attackers in our experiments
declare honest-looking noise.

**18. Which attack was hardest, and where does FedGuard fail?**
Label flip is invisible to every distance-based method: the attacker trains normally on flipped labels, so its
DP-noisy update is as far from the centre as an honest one. Sign flip under DP is hard because its extra
displacement (~2 x signal, length ~1-1.5) is small next to the noise radius (~13) (F39). Result: at one shared epsilon of 3 FedGuard and AutoGM are tied and both give sign / label-flip attackers ~0.30 of the weight (F54); FedGuard limits attackers more than AutoGM at every attacker share and helps most at epsilon 8 (0.170 vs 0.110), and every method breaks at 40% attackers (F58). Under mixed privacy with sign flip both robust aggregators fall below FedAvg (F55).

**19. What went wrong during the project, and what did you learn?**
Three lessons, each logged: an independent cross-check (Flower) found a bug that plausible learning curves hid
(F24); the calibration experiment E0 found a 5-20% error in the declared radius (F34); and shortening early
stopping to save time silently invalidated a week of DP runs, which we caught by comparing paired runs and
re-ran (F46). Always check learning curves and cross-check against an independent implementation.

**20. What would you do with more time?**
Second dataset (CICIoT2023, plan E8); client-level DP or larger clients to raise utility; verifying declared
noise (too_clean); adaptive attackers who know FedGuard's rule; secure aggregation.

## The results in one minute (added 19 Nov)

**21. What is your main result?**
Under mixed privacy (epsilon 1 / 3 / 8 across clients), AutoGM gives every epsilon-1 client weight 0 in every round
(honest rejection 0.40), while FedGuard gives every honest client its fair share 0.10 (honest rejection 0.004) at the
same macro-F1 (0.214 vs 0.219, 3 seeds) (F55). Across non-IID levels AutoGM sidelines 27-40% of honest clients,
FedGuard at most 10% (F56).

**22. Did FedGuard beat AutoGM on accuracy?**
No - it matched it. At one shared epsilon the two are tied on every attack (paired differences -0.018 to +0.011,
F54). FedGuard's gain is fairness to honest private clients, not macro-F1. Without DP at alpha 0.1 it did reach 0.209
vs AutoGM's 0.099 (F56, two seeds).

**23. How did you choose FedGuard's settings without cheating?**
On the validation split only, with a rule fixed before looking (mean validation macro-F1 over no attack + sign flip,
the same protocol as AutoGM's lambda), then frozen before any test result was seen (F50).

**24. Your runs were trained on different machines - is that a problem?**
No: a run is defined by its configuration and seed. The same run on two Kaggle T4s is bit-identical (36 of 36 pairs);
on a T4 vs our laptop's RTX 3050 it differed by 0.0034 macro-F1, below the seed spread (F53). Each run records its
machine in `master.csv`.
