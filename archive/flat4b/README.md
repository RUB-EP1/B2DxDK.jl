# Flat 4-body phase space archive

Cross-check of TF-PWA vs CascadeDecays for $B^+ \to D^- D^{*+} K^+$ when all four momenta are
sampled from flat 4-body phase space (`PhaseSpaceGenerator`).

Production analysis uses **cascade** phase space ($D^*$ fixed at nominal mass) — see
`scripts/all_resonances_model.jl` at the repo root.

## Story

Alex Kazatsky's notebook (`notebooks/all_resonances_sampled_comparison.jl`) samples flat 4b phsp.
There $m(D^0,\pi)$ varies event-by-event; it is not pinned to $m(D^*)$.

We rebuilt the amplitude in Julia (`scripts/all_resonances_model_4b.jl`) and compared to TF-PWA on
the same events. That exposed a bug class: **Breit–Wigner propagators must use event breakup
masses**, not nominal $m(D^*)$, $m(D)$ in the running-width denominator (and related places).

| Generator | $m(D^0,\pi)$ | BW channel masses |
|-----------|--------------|-------------------|
| Cascade | Fixed | Nominal OK |
| Flat 4b | Varies | Must be event-dependent |

**B2DxDK physics:** unaffected — cascade phsp keeps $D^*$ on-shell, so nominal masses are correct.

**Other analyses:** may matter if intermediate masses vary under the sampled phase space and
multichannel/running BW lineshapes are used.

Agreement after the fix: $\sim 10^{-10}$ vs TF-PWA on 1000 events (`data/crosscheck_4b.arrow`,
seed `4040404`).

## Layout

```
data/crosscheck_4b.arrow
notebooks/all_resonances_sampled_comparison.jl   # original notebook
scripts/all_resonances_model_4b.jl               # structured model
scripts/generate_crosscheck_4b.jl
scripts/all_resonances_amplitude_crosscheck_4b.jl
```

## Commands

From repo root:

```bash
julia --project=. archive/flat4b/scripts/all_resonances_amplitude_crosscheck_4b.jl
julia --project=. archive/flat4b/scripts/generate_crosscheck_4b.jl          # needs TF-PWA Python
julia --project=. archive/flat4b/notebooks/all_resonances_sampled_comparison.jl
```
