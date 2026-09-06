# GYRE — research proposal (project #298)

**Certified vortex-ring-state exclusion atlas, minimum-height-loss escape maps,
incipience-aware detection, and escape-governed reinforcement learning for
tiltrotor eVTOL steep descent (KADA KP-2C, PX4+JSBSim/KFDS).**

Session bef42d86 · claimed 2026-08-02 · lock `.project-numbering/298` ·
GitHub `298-GYRE-...` created 2026-08-02T09:09:28Z (push-time authoritative).

---

## 1. The gap

Every sibling project (280–296) constrains the KP-2C conversion along one
physical axis: nominal corridor/rate (280/281/283), faults (284/286), aborts
(285), charge/sag (288-AMPERE), sensing (287), wind (289), noise (290),
payload (291), latency (292), icing (293), electro-thermal (296). **None of
them models descent-phase rotor aerodynamics.** The family plant clamps rotor
axial inflow at `v_ax >= -5 m/s` (296-EMBER `vehicle.py`) because the parsed
JSBSim propeller tables `CT(J)`, `CP(J)` are advance-ratio maps with a linear
windmill extension — momentum-equilibrium physics that is *invalid* in the
vortex-ring state (VRS). 281-RECAP says so explicitly: *"JSBSim propeller aero
has no vortex-ring-state model; no VRS claims are made."*

Yet the KP-2C numbers put VRS squarely inside its useful envelope. Hover
thrust per rotor is `T_h = m g / 4 ≈ 29.0 N`; with disc area
`A = π D²/4 = 0.1297 m²` (D = 0.4064 m) the hover induced velocity is

    v_h = sqrt(T_h / (2 ρ A)) ≈ 9.55 m/s.

The classical VRS band — descent rate between ~0.3 v_h and ~1.5 v_h with
edgewise speed below ~0.9 v_h — is therefore descent rates of roughly
**3–14 m/s at forward speeds below ~9 m/s**: exactly the terminal segment of
any steep decelerating approach or reconversion-to-hover. Steep approaches are
the operationally valuable ones (obstacle clearance, noise, confined
vertiports), so the question *"how steep may the KP-2C descend, where is the
VRS boundary, how is incipient VRS detected, and what is the certified escape
if it is entered"* is an open, safety-critical axis. GYRE builds it.

## 2. Contributions (family-style ladder)

1. **VRS-augmented descent plant (new physics).** Longitudinal KP-2C plant
   (family conventions: batched NumPy, RK4 @ 100 Hz, pair-shared front/rear
   rotors, parsed `kp2.xml` tables) extended below the `-5 m/s` clamp with a
   per-pair **ring-state intensity state** `s ∈ [0,1]`: first-order lag
   (formation time constant ~ R/v_h scaled) driven by a smooth quasi-steady
   severity field `s_qs(V̄z, V̄x)` calibrated to the classical VRS region
   (Johnson NASA/TP-2005-213477; ONERA wake-transport criterion), with
   **entry/exit hysteresis** (exit edge displaced outward, slower collapse
   than formation). Intensity feeds (a) mean thrust deficit
   `T = T_table (1 − k_T s)`, (b) power inflation `P = P_table (1 + k_P s)`,
   (c) band-limited thrust/pitching fluctuations (OU-filtered, RMS ∝ s), and
   (d) reduced control effectiveness — reproducing the documented phenomenology:
   sink-rate divergence, collective ineffectiveness, buffet onset *before*
   mean-thrust loss. **Tiltrotor-specific asymmetry:** the tilting front pair
   leaves VRS both by axis rotation (tilt converts axial descent into edgewise
   flow) and by acceleration; the fixed rear pair only by forward speed — so
   recovery is aerodynamically asymmetric and pitch-coupled. No sibling has
   any of this.

2. **Certified VRS exclusion atlas.** Over the descent state space
   (Vx, Vz, σ): interval-verified bounds on `s_qs` for both pairs (the severity
   field is built from monotone smoothstep factors, so cell-wise interval
   bounds are exact by monotone decomposition); comparison-lemma transfer from
   quasi-steady bound to the lagged intensity along governed trajectories;
   SAFE / BUFFER / EXCLUDED classification with explicit margins; derived
   placards: certified descent-rate cap `Vz_cap(Vx, σ)` and washout speed
   `Vx_wash(Vz)`. Audit-round verification (family convention): monotone
   interval soundness, comparison constants, grid-refinement stability,
   adversarial falsification sampling, governor cross-check.

3. **Minimum-height-loss escape maps (DP).** Backward value iteration on the
   reduced longitudinal model under the **worst-case credible VRS model**
   (max deficit, max fluctuation bias, detection delay): guaranteed
   height-loss-to-recovery `ΔH*(Vx, Vz, σ)` and the optimal escape policy
   (tilt-forward vs power-only — quantifying the tiltrotor's forward-escape
   advantage over the helicopter power response). Derived **floor placard**
   `h_min(Vx, Vz, σ)`: the minimum altitude at which a steep segment may be
   flown such that even worst-case VRS entry is recoverable above ground —
   the escape-commit boundary.

4. **Incipience-aware estimation.** 100 Hz EKF over per-pair intensity and
   thrust-scale using only family-available measurements (rotor speeds, ESC
   currents → motor torque, vertical specific force, baro climb rate);
   torque-residual and lift-residual channels make `s` observable. A windowed
   **GLRT energy detector** on the innovation sequence exploits the buffet
   precursor (fluctuation RMS rises before mean deficit) to declare incipient
   VRS earlier than any kinematic criterion; UCB inflation of `ŝ` (β·σ plus
   split-conformal residual quantile, SAVANT lineage) feeds the governor.
   Detection-delay and false-alarm characterization by Monte Carlo.

5. **Escape-commit governor + escape-governed SAC.** Analytic 10 Hz governor
   (family style, no QP in the loop): descent-rate ceiling from the atlas
   placard shrunk by estimator inflation; tilt-back rate cap near the boundary;
   and an **escape-commit supervisor**: when the UCB intensity crosses the
   trigger, control authority hands to the certified DP escape policy until
   the hysteresis-aware recovery set is re-entered. On top: **from-scratch
   SAC** (226 lineage — no SB3) with a margin-Lagrangian dual on expected
   exclusion violation, trained *through* the governor (shielded exploration),
   observation-augmented with `ŝ`, margins and detector state. Task: steep
   decelerating approach to a low hover gate; baselines: unshielded SAC,
   kinematic-trigger governor (no incipience detector), fixed shallow and
   fixed steep classical profiles.

6. **KFDS closed loop.** Family patch workflow (`VT_*_ENABLE` lineage): a
   `VT_VRS_ENABLE` patch injects the *same* intensity/deficit/fluctuation law
   into the KFDS propulsion path (PX4-Custom `jsbsim_bridge`, kp2 model);
   frozen-intensity open-loop crosscheck (SLEET lineage) plus closed-loop
   governed approaches; python-side parity harness when the native toolchain
   is unavailable (documented honestly, family convention).

## 3. What is certified (honesty ladder, family convention)

- Certified: interval soundness of `s_qs` bounds on every atlas cell
  (monotone decomposition, float64 with enforced margins ≥ 5e-4); the
  comparison-lemma transfer `s(t) ≤ s̄(t)` under governed commands; DP value
  consistency under grid refinement with worst-case model constants; the
  governor lemma (intensity never exceeds `s*` given detector/estimator error
  bounds that are themselves empirically calibrated + conformal-inflated).
- Calibrated, not proved: the VRS empirical constants (deficit magnitude,
  band edges, fluctuation spectra) — taken from the rotorcraft literature and
  frozen in `config.py` with provenance notes; certificates are *conditional
  on the frozen disturbance class*, stated as such in the paper.
- Not claimed: blade-element/CFD fidelity, lateral-directional VRS coupling,
  ground effect (excluded by scope), real-flight validation.

## 4. Model sketch

State (batched, N×14):
`u, w, q, θ, h, ω_f, ω_r, σ, s_f, s_r, ξT_f, ξT_r, ξM, hover-clock-free`
(ξ = OU fluctuation carriers). Controls (N×4): `duty_f, duty_r, δ_rv, σ̇`
(identical to family action set). Environment: headwind, light OU gusts
(nuisance level only — wind capability is 289-GALE's axis), air density.

Per-pair disk-frame velocities (front tilts, rear fixed — 296 conventions):
`v_ax_f = u_a cos σ − w sin σ`, `v_pp_f = |u_a sin σ + w cos σ|`,
`v_ax_r = −w`, `v_pp_r = |u_a|`. Normalize by instantaneous
`v_h = sqrt(max(T̂,T_floor)/(2ρA))`: `V̄z = −v_ax/v_h` (positive in descent),
`V̄x = v_pp/v_h`.

Quasi-steady severity (smooth, monotone factors; ONERA/Johnson calibration):

    s_qs = B_z(V̄z) · B_x(V̄x)
    B_z: smoothstep up from V̄z_on≈0.30 to core≈[0.65, 1.15], down to 0 at V̄z_wm≈1.80
    B_x: 1 at V̄x=0, smoothstep down to 0 at V̄x_wash≈0.90 (exit edge ×1.15 hysteresis)

Intensity: `ṡ = (s_qs − s)/τ(s_qs − s)` with `τ_form ≈ 4 R/v_h ≈ 0.85·`
(formation), `τ_coll ≈ 0.5 τ_form` (collapse), hysteresis via displaced exit
edge. Effects: thrust `T_pair = T_tab(J) (1 − k_T s)`, torque
`Q_pair = Q_tab(J) (1 + k_P s)`, fluctuation `δT = c_T(s) T_h ξ(t)` with OU
carrier `ξ` (τ_ξ ≈ 0.35 s, Strouhal-scaled), pitch fluctuation from front/rear
differential. Constants (frozen, cited): `k_T = 0.30`, `k_P = 0.18`,
`c_T(s) = 0.16 s` (RMS at s=1 → ±16% ≈ flight/model-test band).

## 5. Experiments → figures map (paper §VIII)

F1 platform/architecture diagram · F2 severity field + region chart with
sibling-clamp overlay · F3 exclusion atlas slices (σ = 90/60/30°) + placards ·
F4 escape maps ΔH* + strategy regions + floor placard · F5 detector ROC +
delay vs ramp rate · F6 training curves (5 seeds) + baseline table ·
F7 governed vs ungoverned steep-approach trajectories (time histories with s,
margins, escape-commit event) · F8 KFDS crosscheck parity + closed-loop
transfer table.

## 6. Repository layout (family convention)

    gyre/            config, tables, aircraft/{vehicle,powertrain,trim},
                     vrs/{severity,inflow,fluct}, certify/{atlas,comparison,escape_dp,governor},
                     estimation/{ekf,glrt,conformal}, rl/{env,sac,train}, kfds/{patch,crosscheck}
    experiments/     run_atlas.py, run_escape.py, run_observer.py, run_certs.py,
                     run_train.py, run_eval.py, run_kfds.py, make_figures.py, make_macros.py
    data/            kp2_parsed.json (first-hand parse, scripts/parse_kp2.py)
    paper/           IEEEtran main.tex + refs.bib
    tests/           pytest suite

## 7. Disjointness statement

GYRE certifies **where the rotor wake itself forbids flight** — an
aerodynamic-state axis. It does not certify wind capability (289), noise
(290), energy (288/296), timing (292), contamination (293), faults (284/286),
sensing faults (287 — GYRE's sensors are healthy; it estimates a *flow* state),
aborts from power loss (285), nor nominal corridor/rate (280/281/283).
Escape maps answer "how much sky does the wake demand", not HAVEN's "can I
reach the pad on degraded power". The estimator infers an internal aerodynamic
intensity, not an air-data replacement. No sibling integrates a descent-phase
inflow state, a VRS region, or any escape-commit supervisor.
