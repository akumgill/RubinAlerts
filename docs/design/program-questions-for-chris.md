# MAGNETS follow-up — program-definition questions for Chris

*2026-08-07. These are the choices that set the exposure calculator, the
selection cuts, and the budget model in the shared queue/scheduler. Grouped;
each notes why it matters / what it changes in the tool. Where we've made a
working assumption, it's stated so you can just confirm or correct.*

**Our current working picture (please confirm):** we point the LLAMAS IFU at
suspected Ia near max, one spectrum per target, at a modest binned S/N. That one
spectrum gives us (i) the **type** (Ia vs contaminant) and (ii) a **redshift**
from the SN's broad features. Because we target high *z*, that SN-feature
redshift (~0.005 ≈ 1500 km/s) is good enough for a Hubble diagram (~0.015 mag in
distance modulus at *z*~0.7, negligible vs the ~0.13 mag intrinsic scatter), so
we do **not** need separate host-galaxy redshift follow-up. The standardized
*brightness* for the Hubble diagram comes from the Rubin/ZTF **light curve**
(SALT2 fit), not the spectrum. So: spectrum = type gate + redshift; photometry =
standardizable flux.

## A. Program intent & data products
1. **Is the goal typing + redshift** (a clean, spec-confirmed, spec-*z*'d high-*z*
   Ia sample), and **not** spectral-feature standardization physics (velocities,
   subtypes, twins)? *Sets the S/N target, the phase window, and the binning — a
   standardization goal would demand higher S/N and less binning (both → longer
   exposures).*
2. **Redshift source:** are SN-feature redshifts (~0.005) acceptable for the
   Hubble diagram at 0.6–0.8, or do you want **host-line** redshifts
   (~10–30 km/s; IFU-captured at peak, or a deferred host campaign) for the
   cosmology-grade sample? *Determines whether we ever schedule host-only
   spectroscopy — which is not time-critical and wouldn't touch the nightly
   queue.*

## B. Exposure / ETC parameters (these scale every exposure the tool computes)
3. **Target binned S/N.** Your curve is SNR = 5 **per pixel**; what's the target
   **post-binning** S/N — 10, 15, 25? And is it enough to *type* and get a
   *redshift* at *z*~0.7 (faint, features shifted toward bad sky)? *Exposure ∝
   (S/N)², so this is the single biggest exposure knob. It's the one open value
   in our ETC today.*
4. **Binning / resolution.** Is R ≈ 200 (n_bin ≈ 10, i.e. ~7 elements across a
   ~10,000 km/s Ia feature) enough? Does **any** planned measurement need narrow
   features — Na I D for host/CSM dust, narrow high-velocity features, host
   emission lines — that would force **less** binning? *n_bin sets the
   exposure saving; less binning → proportionally longer exposures. If the
   answer differs by measurement, we make n_bin per-mode instead of global.*
5. **Max per-target exposure + floor/overhead.** What's the longest you'll spend
   on one target (this caps how faint / high-*z* we can chase)? Confirm the
   ~10-min minimum exposure and ~5-min target-switch overhead.

## C. Selection / cuts
6. **High-*z* reach.** Pursue the 0.6–0.8 Rubin-era sample now by relaxing the
   current cuts (z ≤ 0.4, r ≤ 21.5) — accepting extrapolated ETC and long
   exposures — or stay in the ZTF-reachable ≤ 0.4 and open it up when Rubin's
   deeper stream returns? *These cuts move together; today ZTF can't feed 0.6–0.8
   anyway, so this is really "set the policy for when the stream is back."*
7. **Host-morphology selection.** OK to **neutralize** the elliptical-host boost
   for the cosmology sample (it correlates with the mass step — the one avoidable
   bias aligned with our dominant systematic) and instead **record** the applied
   selection function? Keep the boost only for a separate typing-efficiency mode?
8. **Phase window.** Tighten from ±25 d toward **±7–10 d of max** (cleaner type,
   better-constrained standardization epoch, brighter target), or keep it wide
   for yield?

## D. Operations / budget
9. **Charging / parity.** Confirm the model: observed time (exposure + overhead)
   is docked from the **target's** program budget regardless of whose night it
   is; priority = parity bucket × budget availability (no tokens). *This is the
   queue's fairness rule; we currently display allocations but don't yet debit
   them live (manual reconciliation for now).*
10. **Deliverables — anything we're not capturing?** e.g. finder charts,
    acquisition / offset stars, per-target expected S/N printed on the plan.
