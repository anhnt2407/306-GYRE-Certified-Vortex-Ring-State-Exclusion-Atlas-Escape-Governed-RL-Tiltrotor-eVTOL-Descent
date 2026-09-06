# Provenance ledger — what GYRE inherits and what it adds

Family convention (296-EMBER lineage): every methodological pattern is credited
to the sibling that introduced it; the delta column is what is genuinely new
here. Constants of the new physics are frozen in `gyre/config.py` with
literature provenance notes.

| Pattern | Source sibling | GYRE's delta |
|---|---|---|
| First-hand `kp2.xml` parse -> `data/kp2_parsed.json` | 287-SAVANT / 296-EMBER | own parser; adds SHA-256 + dual-copy identity check |
| Batched NumPy longitudinal plant, RK4 100 Hz, pair-shared rotors | 296-EMBER | 13-state descent plant; removes the `v_ax >= -5 m/s` clamp; adds per-pair ring-intensity + OU buffet states |
| CT(J)/CP(J) windmill extension | 296-EMBER | extended to J = -0.45 (deep-descent side); VRS multipliers on top |
| Ring-state severity field + intensity lag + hysteresis | — (new physics) | smoothstep region calibrated to Johnson NASA/TP-2005-213477 + ONERA wake-transport criterion; formation/collapse asymmetry; front/rear asymmetric via tilt geometry |
| Trim grid, continuation ordering, mp.Pool | 293-SLEET / 296-EMBER | trim over (Vx, Vz, sigma) descent cube, not (V, sigma) plane |
| Monotone safe-envelope via running extremum | 293-SLEET | applied along the descent-rate axis of the exclusion margin |
| Interval bounds by monotone decomposition | 281-RECAP posture | exact per-cell bounds of the severity bumps (piecewise-monotone factors) |
| Comparison-lemma transfer quasi-steady -> lagged state | 296-EMBER | scalar ring-intensity comparison; yields the "ring budget" trigger-to-limit time |
| Running-max height-loss DP `G = max(0, min_u[dd + G])` | 285-HAVEN | 4-D state (vx, vz, sigma, s_bar) with worst-pair VRS deficit dynamics; recovered-set haven; power-only vs tilt-forward strategy split |
| CAP-absorbing propagation, RK2 characteristics, quiet-sweep stop | 285-HAVEN | reused as-is (credited) |
| Commit/category maps `cat = 2A + B` thresholded vs height | 285-HAVEN | floor placard h_min over descent cube at intensity slices |
| Torque-residual EKF (Kt(I-I0) - Ixx domega) | 293-SLEET | residual target is ring intensity (aero-state), not map erosion; per-pair split via tilt geometry |
| Shadow/NIS persistence detection (m-of-k) | 287-SAVANT | windowed GLRT *energy* detector on innovation variance (buffet precursor) — new detector family member |
| UCB + held-out quantile inflation of estimate | 287-SAVANT / 293-SLEET | plain quantiles (conformal machinery stays 287's lane) |
| Monotone absorption of estimation error | SLEET/HAVEN/SAVANT | placard non-increasing in descent rate; decide on s_ucb and V_hat - beta |
| Governor modes + published certified edge in obs | 293-SLEET | MONITOR/RESTRICT/ESCAPE; publishes descent-rate ceiling; escape-commit handover to DP policy (one-step lookahead, HAVEN HJGreedy pattern) |
| From-scratch SAC (no SB3), CPU torch | 296-EMBER | constrained SAC with **cost critic + Lagrangian dual** (no Lyapunov critic — that is EMBER's LAC lane) + escape supervisor in the loop |
| Audit rounds closing declared margins | 296-EMBER | rounds close the DP-vs-plant model gap and the placard margins |
| Audits measured with certificate switched off | 293-SLEET | reused for the dynamic-bite and detection-delay batteries |
| In-process KFDS crosscheck (`jsbsim` module) | 293-SLEET / 296-EMBER | adds project-local model copy with `external_reactions` VRS deficit-force injection; nominal-branch parity documents where KFDS physics ends (the RECAP disclaimer, quantified) |
| RK4 reference integrator in domination audits | 296-EMBER (log F5) | inherited as a rule |
| `make_macros.py` single-sourcing + honesty rules | 296-EMBER | inherited verbatim as a rule |
| Greyscale/Times paper style, `\genfig` | 296-EMBER | inherited |

## Genuinely new in #298 (claim ledger)

1. A vortex-ring-state intensity model *of any kind* on this platform — the
   family plant clamps descending axial inflow; 281-RECAP explicitly
   disclaims VRS.
2. The tiltrotor-specific asymmetric VRS geometry: tilting front pair exits by
   axis rotation + acceleration; fixed rear pair only by washout — and the
   pitch coupling that follows.
3. Certified exclusion atlas over the descent cube (Vx, Vz, sigma) with exact
   interval bounds of the severity field and a comparison-lemma ring budget.
4. Minimum-height-loss escape maps under worst-case VRS deficit, and the
   power-only vs tilt-forward strategy frontier (the quantified "tiltrotor
   advantage" over the helicopter response).
5. Buffet-precursor incipience detection (GLRT innovation-energy) — declares
   incipient VRS before any kinematic placard crossing.
6. Escape-commit governed RL for steep decelerating approaches.

## External prior art (DON'T-CLAIM ledger)

- VRS physics/regions/fluctuations: Johnson NASA/TP-2005-213477; ONERA
  criterion (Jimenez/Taghizad); Wolkovitch/Peters wake-transport arguments;
  Vuichard recovery (lateral — excluded here, longitudinal study). GYRE claims
  the *certification/detection/learning composition on a tiltrotor*, not the
  aerodynamic phenomenology.
- Flight-test VRS boundaries (V-22 HROD, Betzina): cited as magnitude
  provenance for deficit constants; no new aerodynamic data claimed.
- Safe RL / shielding / CBF governors: standard citations; the contribution is
  the certified-escape-commit composition, not shielding per se.
