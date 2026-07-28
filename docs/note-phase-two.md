# Particle-2 phase and partial-wave mixing (X1(2900) dk root)

Notes on Issue A in the B2DxDK ↔ TF-PWA mapping: why the Jacob–Wick
particle-2 phase cannot be dropped or absorbed into a scalar matching factor,
even for integer-spin mesons.

**Context:** $B^+ \to D^{*+}\,X_1(2900)$ with $X_1 \to D^- K^+$ (dk topology).
Three production partial waves $(L,S)=(0,0),(1,1),(2,2)$ with LS couplings
$g_{00}, g_{11}, g_{22}$.

**Code:** validated against `RecouplingLS`, `CascadeDecays.routed_vertex_amplitude`,
and `MissingParticleTwoPhaseLS` — see
[`test/particle2_algebra.jl`](test/particle2_algebra.jl).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{A}$ | Two-body helicity amplitude |
| $D$ | Wigner $D$-function (angular part) |
| $H_{\lambda_1,\lambda_2}$ | **Bare** JW helicity coupling: $\mathcal{A} = D\,H$ |
| $h_{\lambda_1,\lambda_2}$ | **TF-PWA** coupling (no particle-2 phase): $\mathcal{A} = D\,h$ |
| $\eta(\lambda_2)$ | Particle-2 phase: $(-1)^{j_2-\lambda_2}$; for $j_2=1$: $\eta(\lambda)=(-1)^{1-\lambda}$ |
| $g_{LL}$ | LS scalar for $(L,S)=(L,L)$, $L=0,1,2$ |
| $C^L_\lambda$ | CG factor $\langle 1,\lambda;\,1,-\lambda \mid L,\,0\rangle$ |

For integer-spin mesons $(-1)^{2\lambda}=1$, so $\eta(\lambda)=(-1)^{1-\lambda}$ is
**invariant under** $\lambda\to-\lambda$; it distinguishes $\lambda=0$ from
$\lambda=\pm1$, not $+1$ from $-1$.

**Fundamental relation**

$$H_{\lambda_1,\lambda_2}=\eta(\lambda_2)\,h_{\lambda_1,\lambda_2}
\qquad\Longleftrightarrow\qquad
h_{\lambda_1,\lambda_2}=\eta(\lambda_2)\,H_{\lambda_1,\lambda_2}$$

$J=0$ selection: $\lambda_1=\lambda_2\equiv\lambda$; all off-diagonal couplings
vanish. Only $L=S$ couples to $J=0$.

---

## LS → helicity (correct small $h$)

$$h_{\lambda,\lambda} = \sum_{L=0,1,2} g_{LL}\, C^L_\lambda,
\qquad
C^L_\lambda \equiv \langle 1,\lambda;\,1,-\lambda \mid L,\,0\rangle$$

Analytic CG values (validated vs `RecouplingLS` to $10^{-16}$):

$$C^0_\lambda = \frac{(-1)^{1-\lambda}}{\sqrt{3}},
\qquad
C^1_{+1} = -\frac{1}{\sqrt{2}},\;
C^1_{-1} = +\frac{1}{\sqrt{2}},\;
C^1_0 = 0$$

$$C^2_{+1} = C^2_{-1} = \frac{1}{\sqrt{6}},
\qquad
C^2_0 = \sqrt{\frac{2}{3}}$$

**Matrix form** with $\mathbf{g}=(g_{00},g_{11},g_{22})^T$ and
$\mathbf{h}=(h_{-1,-1},h_{0,0},h_{+1,+1})^T$:

$$\mathbf{h} = C\,\mathbf{g}$$

$$C =
\begin{pmatrix}
+\frac{1}{\sqrt{3}} & +\frac{1}{\sqrt{2}} & +\frac{1}{\sqrt{6}} \\
-\frac{1}{\sqrt{3}} &  0                 & +\sqrt{\frac{2}{3}} \\
+\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & +\frac{1}{\sqrt{6}}
\end{pmatrix},
\quad \det C = 1$$

LS extraction: $\mathbf{g} = C^{-1}\mathbf{h}$.

---

## Bare JW coupling $H = \eta\,h$

$$H_{\lambda,\lambda}
= \eta(\lambda)\sum_{L=0,1,2} g_{LL}\,C^L_\lambda
= \sum_{L=0,1,2} M^L_\lambda\, g_{LL},
\qquad
M^L_\lambda = \eta(\lambda)\,C^L_\lambda$$

Only the $\lambda=0$ row is flipped ($\eta(0)=-1$, $\eta(\pm1)=+1$):

$$\mathbf{H} = P\,C\,\mathbf{g},
\qquad
P = \mathrm{diag}(+1,-1,+1)$$

$$M = PC =
\begin{pmatrix}
+\frac{1}{\sqrt{3}} & +\frac{1}{\sqrt{2}} & +\frac{1}{\sqrt{6}} \\
+\frac{1}{\sqrt{3}} &  0                 & -\sqrt{\frac{2}{3}} \\
+\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & +\frac{1}{\sqrt{6}}
\end{pmatrix}$$

At $\lambda=0$: **$h$** gives $-g_{00}/\sqrt{3} + \sqrt{2/3}\,g_{22}$;
**$H$** gives $+g_{00}/\sqrt{3} - \sqrt{2/3}\,g_{22}$.
Same two partial waves, **opposite relative sign**.

---

## Per partial wave

### $(L,S)=(0,0)$, coupling $g_{00}$

| $\lambda$ | $h_{\lambda,\lambda}/g_{00}$ | $\eta(\lambda)$ | $H_{\lambda,\lambda}/g_{00}$ |
|:---:|:---:|:---:|:---:|
| $+1$ | $+1/\sqrt{3}$ | $+1$ | $+1/\sqrt{3}$ |
| $-1$ | $+1/\sqrt{3}$ | $+1$ | $+1/\sqrt{3}$ |
| $0$ | $-1/\sqrt{3}$ | $-1$ | $+1/\sqrt{3}$ |

### $(L,S)=(1,1)$, coupling $g_{11}$

| $\lambda$ | $h_{\lambda,\lambda}/g_{11}$ | $\eta(\lambda)$ | $H_{\lambda,\lambda}/g_{11}$ |
|:---:|:---:|:---:|:---:|
| $+1$ | $-1/\sqrt{2}$ | $+1$ | $-1/\sqrt{2}$ |
| $-1$ | $+1/\sqrt{2}$ | $+1$ | $+1/\sqrt{2}$ |
| $0$ | $0$ | $-1$ | $0$ |

For $L=1$ alone: $\eta$ hits only the $\lambda=0$ component, which is already
zero — **$H=h$** and no TF-PWA workaround is needed.

### $(L,S)=(2,2)$, coupling $g_{22}$

| $\lambda$ | $h_{\lambda,\lambda}/g_{22}$ | $\eta(\lambda)$ | $H_{\lambda,\lambda}/g_{22}$ |
|:---:|:---:|:---:|:---:|
| $+1$ | $+1/\sqrt{6}$ | $+1$ | $+1/\sqrt{6}$ |
| $-1$ | $+1/\sqrt{6}$ | $+1$ | $+1/\sqrt{6}$ |
| $0$ | $+\sqrt{2/3}$ | $-1$ | $-\sqrt{2/3}$ |

---

## Incorrect convention (phase forgotten)

Mistake: use $H^{\mathrm{wrong}}_{\lambda,\lambda} \equiv h_{\lambda,\lambda}$ in
$\mathcal{A}=D\,H$:

Wrong: $\mathbf{H}^{\mathrm{wrong}} = C\,\mathbf{g}$. Correct: $\mathbf{H} = P\,C\,\mathbf{g}$.

All three $g_{LL}$ still appear wherever CG is nonzero — **no partial wave is
missing**. The error is the **wrong mixing matrix**:

$$\Delta H_\lambda \equiv H_\lambda - H^{\mathrm{wrong}}_\lambda
= \begin{cases}
0, & \lambda=\pm1 \\
2\left(\dfrac{g_{00}}{\sqrt{3}} - \sqrt{\dfrac{2}{3}}\,g_{22}\right), & \lambda=0
\end{cases}$$

($g_{11}$ drops out of $\Delta H_0$ because $C^1_0=0$.)

### Schematically at $\lambda=0$

Expand $h_{0,0}$ into partial-wave pieces ($g_{11}$ does not contribute because
$C^1_0=0$):

$$h_{0,0} = \frac{g_{00}}{\sqrt{3}} + \sqrt{\frac{2}{3}}\,g_{22}$$

Bare JW coupling: $H_{0,0}=\eta(0)\,h_{0,0}$ with $\eta(0)=-1$:

$$H_{0,0} = -\frac{g_{00}}{\sqrt{3}} + \sqrt{\frac{2}{3}}\,g_{22}$$

**Wrong** (forget $\eta$ but still write $\mathcal{A}=D\,H$): use $h_{0,0}$ in
place of $H_{0,0}$.

Same partial waves present; the S-wave and D-wave terms have the **wrong relative
sign** compared to JW:

| coupling | $g_{00}$ term | $g_{11}$ term | $g_{22}$ term |
|---|:---:|:---:|:---:|
| correct $h_{0,0}$ | $+g_{00}/\sqrt{3}$ | $0$ | $+\sqrt{2/3}\,g_{22}$ |
| correct $H_{0,0}$ | $-g_{00}/\sqrt{3}$ | $0$ | $+\sqrt{2/3}\,g_{22}$ |
| wrong ($h$ used as $H$) | $+g_{00}/\sqrt{3}$ | $0$ | $+\sqrt{2/3}\,g_{22}$ |

The wrong row equals $h_{0,0}$, not $H_{0,0}$.

---

## Bottom line

| | Correct TF-PWA | Correct JW bare | Incorrect (forget $\eta$) |
|---|:---:|:---:|:---:|
| Helicity map | $\mathbf{h}=C\mathbf{g}$ | $\mathbf{H}=PC\mathbf{g}$ | $\mathbf{H}^{\mathrm{wrong}}=C\mathbf{g}$ |
| All $g_{LL}$ present? | yes | yes | yes |
| Fixable by one overall sign? | — | — | **no** |

Forgetting the particle-2 phase **mixes partial waves inside the helicity basis**
(wrong relative weight of $g_{00}$ vs $g_{22}$ at $\lambda=0$), rather than
dropping any LS term. This is not removable by a scalar `matching_factor`;
`MissingParticleTwoPhaseLS` is required for X1 L=0 and L=2 chains only
(`root_recoupling=:missing_particle2` in `src/resonance_table.jl`).

---

## Code validation

Run from the repo root:

```bash
julia --project=. test/particle2_algebra.jl
```

| Test | Statement | Result |
|------|-----------|--------|
| 1 | `routed_vertex_amplitude` $= \eta(\lambda_2)\times$ `RecouplingLS` | PASS ($<10^{-15}$) |
| 2 | `MissingParticleTwoPhaseLS` + outer $\eta$ $= h$ | PASS |
| 3 | $\mathbf{h}=C\mathbf{g}$, $\mathbf{H}=PC\mathbf{g}$ | PASS |
| 4 | Analytic CG vs `RecouplingLS` | PASS ($<10^{-15}$) |
| 5 | Wrong vs correct at $\lambda=0$ only | $\Delta \neq 0$ |
| 9 | $L=1$ only: $H=h$ | PASS |
| 10 | Auto-cancel: L0=true, L1=false | matches code |
| 11 | Full model vs TF-PWA reference (100 events) | PASS ($<10^{-13}$) |

**Implementation map**

| Layer | Object | Role |
|-------|--------|------|
| `ThreeBodyDecays` | `RecouplingLS` | returns small $h$ (CG with $\langle j_2,-\lambda_2 \mid \ldots\rangle$) |
| `CascadeDecays` | `_particle_two_phase` in `_vertex_coupling_value` | multiplies by $\eta(\lambda_2)$ → bare $H$ |
| `B2DxDK` | `MissingParticleTwoPhaseLS` | applies $\eta$ again so net factor is $h$ (compensates TF-PWA's missing phase) |
| `B2DxDK` | `root_recoupling` | `:standard` or `:missing_particle2` in `resonance_table.jl` |
