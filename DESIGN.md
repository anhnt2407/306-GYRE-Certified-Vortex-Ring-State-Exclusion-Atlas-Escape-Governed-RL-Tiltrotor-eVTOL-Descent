# GYRE design (frozen before implementation; session bef42d86, 2026-08-02)

Axis: **DESCENT AERODYNAMIC STATE — vortex-ring-state (VRS) exclusion** on the
KADA KP-2C. See `docs/01-proposal.md` for the research argument and
`PROVENANCE.md` for the inheritance ledger.

## Architecture

    parsed kp2.xml ──► tables ──► plant (13-state, VRS-augmented) ──► KFDS crosscheck
                          │                                              (in-process jsbsim,
                          ▼                                               external_reactions
        trim (Vx,Vz,σ) descent cube                                        deficit injection)
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
     exclusion atlas   escape DP (4-D)   ring-budget lemma
     (interval-sound   G = minimal       (comparison ODE,
      severity bounds,  height loss to    trigger→limit time)
      placards)         recovered set)
          │               │                   │
          └───────┬───────┴───────┬───────────┘
                  ▼               ▼
        RingEKF + GLRT ──► governor (MONITOR/RESTRICT/ESCAPE,
        (incipience)         published ceiling, escape-commit)
                                  │
                                  ▼
                     constrained SAC (cost critic + dual),
                     trained through the governor; MC + KFDS campaigns

## Plant (gyre/aircraft/vehicle.py) — x (N, 13)

    0 u   1 w   2 q   3 theta   4 h   5 om_f   6 om_r   7 sigma   8 soc
    9 s_f  10 s_r     (ring intensity per pair, [0,1])
    11 xi_f 12 xi_r   (OU buffet carriers, unit variance)

Controls (N,4): duty_f, duty_r in [0,1]; delta_rv deg (±23); sigma_dot rad/s.
Env: wind_x headwind, gust OU (nuisance level), rho. 100 Hz RK4 (buffet OU
integrated exactly per substep); agent/governor at 10 Hz.

Per-pair disk kinematics (front tilts, rear fixed):
    v_ax_f = ua cosσ − w sinσ     v_pp_f = |ua sinσ + w cosσ|
    v_ax_r = −w                   v_pp_r = |ua|
Normalization: v_h = sqrt(max(T_tab, T_floor)/(2 ρ A_disc)) per pair;
V̄z = −v_ax/v_h (positive descending-through-disk), V̄x = v_pp/v_h.

Severity field (quasi-steady, smooth, piecewise-monotone):
    s_qs = B_z(V̄z) · B_x(V̄x; e)
    B_z: smoothstep 0→1 over [Z_ON, Z_CORE0], 1 on [Z_CORE0, Z_CORE1],
         smoothstep 1→0 over [Z_CORE1, Z_WM]
    B_x: 1 − smoothstep over [X_IN, X_OUT(e)]; exit edge X_OUT scaled by
         (1 + HYST · s) — entry/exit hysteresis
Intensity lag: ds/dt = (s_qs − s)/τ, τ = TAU_FORM if rising else TAU_COLL.
Effects: T_pair = T_tab(J)(1 − K_T s + C_FL s ξ);  Q_pair = Q_tab(J)(1 + K_P s).
Buffet: dξ = −ξ/τ_ξ dt + sqrt(2/τ_ξ) dW  (unit-variance OU, per pair).

Constants (frozen; provenance in config comments): Z_ON 0.28, Z_CORE 0.60–1.20,
Z_WM 1.70, X_OUT 0.95, HYST 0.20, K_T 0.30, K_P 0.18, C_FL 0.12,
TAU_FORM 1.1 s [0.7, 1.6], TAU_COLL 0.45 s [0.3, 0.8], τ_ξ 0.35 s,
s* = 0.35 (exclusion), s_trig = 0.22, s_safe = 0.15.

## Certified objects

- **C1 exclusion atlas** over Vx∈[0,22]×1, Vz_desc∈[0,12]×0.5, σ∈{0..90}×6:
  descent trim (α, duty_f, duty_r, δ_rv); per-cell **interval** bound of s_qs
  (both pairs) by monotone decomposition of B_z, B_x over the cell box with
  v_h interval from thrust interval; SAFE/BUFFER/EXCLUDED at s* with margin;
  placards Vz_cap(Vx, σ) (monotone envelope along Vz) and Vx_wash(Vz).
- **C2 ring budget**: on any governed segment with cell bound s_qs^UB < s*,
  intensity from s0 ≤ s_trig needs at least
  t_min = TAU_FORM_LO · ln((s_qs^UB − s0)/(s_qs^UB − s*)) to reach s*
  (comparison lemma; = ∞ if s_qs^UB ≤ s*). Audited against the 13-state plant.
- **C3 escape maps**: HAVEN-form running-max DP on (vx, vz, σ, s̄), worst-pair
  deficit (1 − K_T_HI s̄) on all rotors, s̄ lag with worst-case constants
  (rise TAU_FORM_LO, fall TAU_COLL_HI), haven = {s̄ ≤ s_safe ∧ descent
  arrested}; G* = certified min height loss; strategy split: full controls vs
  σ̇ = 0 (power-only); floor placard h_min = G* + margin; category maps.
- **C4 governor soundness**: ceilings consume ŝ_ucb (EKF UCB + held-out
  quantile) and the placard's monotone axis; escape-commit fires at ŝ_ucb ≥
  s_trig or GLRT persistence; audited invariance (never s > s* in governed
  campaigns; violations counted against three thresholds s_trig/s*/1.0).

## Estimation

RingEKF (50 Hz): state per env [s_f, s_r, b_az]; measurements: 2 pair shaft
torque residuals (Kt(I−I0) − I_xx ω̇ vs CP(J)(1+K_P s)), a_z specific force
(thrust deficit channel), ḣ. FD Jacobians, Joseph update, χ² gate.
GLRT incipience: windowed innovation-energy statistic on the a_z + torque
channels vs H0 noise floor, m-of-k persistence, latching; delay reported
det/pre/never (SAVANT convention).

## RL

Constrained SAC from scratch (torch CPU): twin Q + **cost critic** Q_c,
Lagrangian dual λ on E[cost] ≤ d; actions [descent-rate command inside the
published ceiling, pitch offset, tilt-rate inside cap]; task steep decelerating
approach entry (V 14–24 m/s, γ −5..−13°, h 60–120 m) → hover gate (V ≤ 1.5,
sink ≤ 1, σ ≥ 80°, h_gate 25 m); cost = exclusion exceedance + escape events;
variants gyre / ungoverned / blind / kinematic-trigger; 5+2+2+2 seeds;
paired ID/OOD MC (OOD: K_T ↑, TAU_FORM ↓, fluct ↑, wind).

## KFDS

In-process `jsbsim` (KFDS fork). A: hover anchor parity. B: descending-flight
rotor parity sweep — quantifies exactly where native KFDS physics ends (the
281-RECAP disclaimer made measurable). C: closed-loop governed vs ungoverned
steep approaches on a **project-local model copy** with `external_reactions`
per-pair deficit+buffet forces (magnitudes driven from python each step by the
same frozen law; direction split into body-x/z components for the tilting
pair). D: envelope/placard audit on the KFDS trajectories.

## Paper (paper/main.tex, IEEEtran journal, ~10 pp)

I Intro · II Vehicle and VRS-augmented model (+ arch TikZ) · III Exclusion
atlas + ring budget (defs, lemmas, algorithm) · IV Escape value maps ·
V Incipience estimation · VI Escape-commit governor + constrained SAC ·
VII Results (atlas/DP/detector/RL/KFDS + audits + limitations) · VIII
Conclusion. Abstract ≈200 words and conclusion 200–250 words, both
number-free. All numbers via generated macros.tex.
