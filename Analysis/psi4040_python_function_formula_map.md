# Isolated Python Notebook: Function Calls and LaTeX Formula Map

This file maps the functions and execution calls in `psi4040_independent_amplitude_flow.ipynb` to the formulas they implement. The goal is that a reader can reproduce the same complex amplitude from the formulas alone, using the same four-vectors and scalar parameters loaded in the notebook.

## Conventions

Four-vectors are stored as

```math
p^\mu=(E,p_x,p_y,p_z)=(E,\vec p),
\qquad
p^2=E^2-\vec p^{\,2}.
```

The isolated chain is

```math
B^+\to \psi(4040)K^+,\qquad
\psi(4040)\to D^{*-}D^+,\qquad
D^{*-}\to D^0\pi^-.
```

The active LS waves are

```math
(L,S)_{B\to\psi K}=(1,1),\qquad
(L,S)_{\psi\to D^*D}=(1,1),\qquad
(L,S)_{D^*\to D^0\pi}=(1,0).
```

All selected couplings are set to

```math
g_{\mathrm{total}}=g_{B\to\psi K}=g_{\psi\to D^*D}=g_{D^*\to D^0\pi}=1+0i.
```

## Vector, Lorentz, and Frame Functions

| Python function / call | LaTeX formula implemented |
|---|---|
| `dot3(a, b)` | \(\displaystyle \operatorname{dot3}(\vec a,\vec b)=\vec a\cdot\vec b=\sum_{i=1}^{3}a_i b_i\). |
| `norm3(a)` | \(\displaystyle \operatorname{norm3}(\vec a)=\vert\vec a\vert=\sqrt{\vec a\cdot\vec a}\). |
| `unit(v)` | \(\displaystyle \widehat v=\frac{\vec v}{\vert\vec v\vert}\). |
| `cross_unit(a, b)` | \(\displaystyle \widehat n=\frac{\vec a\times\vec b}{\vert\vec a\times\vec b\vert}\). If \(\vert\vec a\times\vec b\vert\simeq0\), the code replaces \(\vec b\) by \(\vec b+\vec 1\) before normalizing. |
| `angle_from(v, x_axis, y_axis)` | \(\displaystyle \phi=\operatorname{atan2}\!\left(\vec v\cdot\widehat y,\vec v\cdot\widehat x\right)\). |
| `angle_zx_z_getx(z1, x1, z2)` | \(\displaystyle \widehat y_1=\widehat{z_1\times x_1},\quad \widehat x_1=\widehat{y_1\times z_1},\quad \widehat y_r=\widehat{z_1\times z_2},\quad \widehat x_r=\widehat{y_r\times z_1}\). Then \(\displaystyle \alpha=\operatorname{atan2}(\widehat x_r\cdot\widehat y_1,\widehat x_r\cdot\widehat x_1),\quad \beta=\operatorname{atan2}(\widehat z_2\cdot\widehat x_r,\widehat z_2\cdot\widehat z_1),\quad \gamma=0\). |
| `invariant_mass(p4)` | \(\displaystyle m(p)=\sqrt{\left\vert E^2-p_x^2-p_y^2-p_z^2\right\vert}=\sqrt{\vert p^2\vert}\). |
| `boost_vector(p4)` | \(\displaystyle \vec\beta(p)=\frac{\vec p}{E}\). |
| `boost(p4, beta)` | \(\displaystyle \gamma=\frac{1}{\sqrt{1-\vec\beta^{\,2}}},\quad b=\vec\beta\cdot\vec p,\quad c=\frac{\gamma-1}{\vec\beta^{\,2}}\). The boosted vector is \(\displaystyle E'=\gamma(E+b),\quad \vec p'=\vec p+c\,b\,\vec\beta+\gamma E\,\vec\beta\). |
| `rest_vector(core_p4, other_p4)` | \(\displaystyle p_{\mathrm{other}}^{*\mu}=\Lambda\!\left[-\vec\beta(p_{\mathrm{core}})\right]p_{\mathrm{other}}^\mu\). |

## Phase-Space Generation Functions

| Python function / call | LaTeX formula implemented |
|---|---|
| `two_body_momentum(m0, m1, m2)` | \(\displaystyle q(m_0;m_1,m_2)=\frac{\sqrt{\lambda(m_0^2,m_1^2,m_2^2)}}{2m_0}\), with \(\displaystyle \lambda(x,y,z)=x^2+y^2+z^2-2xy-2xz-2yz\). Equivalently, \(\displaystyle q=\frac{\sqrt{[m_0^2-(m_1+m_2)^2][m_0^2-(m_1-m_2)^2]}}{2m_0}\). |
| `PhaseSpaceGeneratorNP(m0, daughter_masses)` | Defines an \(n\)-body decay at rest: \(\displaystyle P^\mu=(m_0,0,0,0)\to \sum_{i=1}^{n}p_i^\mu,\quad p_i^2=m_i^2,\quad \sum_i p_i^\mu=P^\mu\). |
| `get_mass_range()` | For recursive intermediate masses \(M_i\), the allowed interval is \(\displaystyle M_i^{\min}=M_{i-1}^{\min}+m_{\mathrm{new}},\quad M_i^{\max}=m_0-\sum_{\mathrm{spectators}}m_j\). In the notebook this yields the nested intervals used by TF-PWA. |
| `generate_mass(n_events)` | Samples each intermediate mass uniformly: \(\displaystyle M_i=a_i+(b_i-a_i)u_i,\quad u_i\sim U(0,1)\). |
| `get_weight(masses)` | Computes the accept/reject phase-space weight \(\displaystyle w(M_1,\ldots,M_{n-2})=\frac{1}{w_{\max}}\prod_{r=1}^{n-1}q_r\), where \(\displaystyle q_r=q(M_r;M_{r-1},m_r)\) for the recursive two-body splits. |
| `flatten_mass(masses)` | Rejection rule: \(\displaystyle \text{accept event if }w>u,\quad u\sim U(0,1)\). |
| `generate_momentum_i(m0, m1, m2, ...)` | Isotropic two-body decay in the \(m_0\) rest frame: \(\displaystyle \cos\theta=2u_1-1,\quad \phi=2\pi u_2,\quad \vec q=q(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)\). The new daughter is \(\displaystyle p_2^\mu=(\sqrt{q^2+m_2^2},\vec q)\), and the recoil is \(\displaystyle p_1^\mu=(\sqrt{q^2+m_1^2},-\vec q)\). |
| `rest_vector(p_boost, old_p)` inside `generate_momentum_i` | Recursively boosts already generated daughters: \(\displaystyle p_{\mathrm{old}}^{\mu,\mathrm{new}}=\Lambda[-\vec\beta(p_{\mathrm{boost}})]\,p_{\mathrm{old}}^\mu\). |
| `generate_b2dxdk_phase_space(n_events, seed)` | Generates \(\displaystyle B^+(m_B)\to D^+(m_D)K^+(m_K)D^0(m_{D^0})\pi^-(m_\pi)\) with \(\displaystyle \sum_i p_i^\mu=(m_B,0,0,0)\). |
| `validate_generated_phase_space(...)` | Checks \(\displaystyle \max_i\left\vert\sqrt{p_i^2}-m_i\right\vert<\epsilon\) and \(\displaystyle \max_\mu\left\vert\sum_i p_i^\mu-(m_B,0,0,0)^\mu\right\vert<\epsilon\). |
| `TFPWAPhaseSpaceGenerator(...).generate(...)` | Same mathematical generator as above, from TF-PWA: \(\displaystyle \{p_i^\mu\}_{\mathrm{TF-PWA}}\sim d\Phi_n(P;p_1,\ldots,p_n)\). |
| `pair_mass(sample, names)` | Subsystem invariant mass: \(\displaystyle m_{S}=\sqrt{\left(\sum_{i\in S}p_i^\mu\right)^2}\). |
| `distribution_summary(values)` | MC comparison vector: \(\displaystyle s(x)=\left(\overline{x},\sigma_x,Q_{0.10},Q_{0.25},Q_{0.50},Q_{0.75},Q_{0.90}\right)\). |

## Event Reconstruction Calls

| Python call | LaTeX formula implemented |
|---|---|
| `p4["Dst"] = p4["D0"] + p4["pi"]` | \(\displaystyle p_{D^*}^\mu=p_{D^0}^\mu+p_\pi^\mu\). |
| `p4["Psi(4040)"] = p4["Dst"] + p4["D"]` | \(\displaystyle p_{\psi}^\mu=p_{D^*}^\mu+p_D^\mu\). |
| `p4["Bp"] = p4["Psi(4040)"] + p4["K"]` | \(\displaystyle p_B^\mu=p_\psi^\mu+p_K^\mu\). |
| `event_mass = {name: invariant_mass(vec) ...}` | \(\displaystyle m_X=\sqrt{p_X^2}\) for \(X\in\{B,\psi,D^*,D,K,D^0,\pi\}\). |

## Helicity-Angle Calls

| Python function / call | LaTeX formula implemented |
|---|---|
| `chain = [("Bp", ["Psi(4040)", "K"]), ...]` | Decay graph: \(\displaystyle B\to\psi K,\quad \psi\to D^*D,\quad D^*\to D^0\pi\). |
| `compute_chain_boosts(p4, chain)` | For each vertex \(A\to BC\), compute daughter momenta in the parent rest frame: \(\displaystyle p_B^{*(A)}=\Lambda[-\vec\beta(p_A)]p_B,\quad p_C^{*(A)}=\Lambda[-\vec\beta(p_A)]p_C\). |
| `calculate_helicity_angles(p4, chain)` | For each vertex, compute \(\displaystyle \Omega_A^B=(\alpha_A^B,\beta_A^B,\gamma_A^B)\) from the frame formula in `angle_zx_z_getx`. |
| `angles["Bp"]["Psi(4040)"]` | \(\displaystyle \Omega_B^\psi=(\alpha_B^\psi,\beta_B^\psi,0)\), the direction of \(\psi\) in the \(B\) rest frame. |
| `angles["Psi(4040)"]["Dst"]` | \(\displaystyle \Omega_\psi^{D^*}=(\alpha_\psi^{D^*},\beta_\psi^{D^*},0)\), the direction of \(D^*\) in the \(\psi\) rest frame. |
| `angles["Dst"]["D0"]` | \(\displaystyle \Omega_{D^*}^{D^0}=(\alpha_{D^*}^{D^0},\beta_{D^*}^{D^0},0)\), the direction of \(D^0\) in the \(D^*\) rest frame. |

## Breakup Momenta and Barrier Factors

| Python function / call | LaTeX formula implemented |
|---|---|
| `get_relative_p2(m0, m1, m2)` | \(\displaystyle q^2(m_0;m_1,m_2)=\frac{[m_0^2-(m_1+m_2)^2][m_0^2-(m_1-m_2)^2]}{4m_0^2}\). |
| `q2_bp = get_relative_p2(event_mass["Bp"], event_mass["Psi(4040)"], event_mass["K"])` | \(\displaystyle q_B^2=q^2(m_B^{\mathrm{evt}};m_\psi^{\mathrm{evt}},m_K^{\mathrm{evt}})\). |
| `q02_bp = get_relative_p2(nominal_mass["Bp"], nominal_mass["Psi(4040)"], nominal_mass["K"])` | \(\displaystyle q_{0,B}^2=q^2(m_B^0;m_\psi^0,m_K^0)\). |
| `q2_psi = get_relative_p2(event_mass["Psi(4040)"], event_mass["Dst"], event_mass["D"])` | \(\displaystyle q_\psi^2=q^2(m_\psi^{\mathrm{evt}};m_{D^*}^{\mathrm{evt}},m_D^{\mathrm{evt}})\). |
| `q02_psi = get_relative_p2(nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])` | \(\displaystyle q_{0,\psi}^2=q^2(m_\psi^0;m_{D^*}^0,m_D^0)\). |
| `q2_dst = get_relative_p2(event_mass["Dst"], event_mass["D0"], event_mass["pi"])` | \(\displaystyle q_{D^*}^2=q^2(m_{D^*}^{\mathrm{evt}};m_{D^0}^{\mathrm{evt}},m_\pi^{\mathrm{evt}})\). |
| `bprime_polynomial(l, z)` | \(\displaystyle B_0(z)=1,\quad B_1(z)=1+z,\quad B_2(z)=z^2+3z+9,\quad B_3(z)=z^3+6z^2+45z+225\). |
| `bprime_q2(l, q2, q02, d)` | \(\displaystyle B'_L(q,q_0,d)=\sqrt{\frac{B_L(q_0^2d^2)}{B_L(q^2d^2)}}\). |
| `barrier_factor2(l, mass, q2, q02, d, barrier_factor_norm=True)` | \(\displaystyle F_L(q,q_0,d)=\frac{q^L}{\vert q_0\vert^L}B'_L(q,q_0,d)\). For this notebook \(L=1\), so \(\displaystyle F_1=\frac{q}{\vert q_0\vert}\sqrt{\frac{1+q_0^2d^2}{1+q^2d^2}}\). |

## Wigner-D and LS Recoupling

| Python function / call | LaTeX formula implemented |
|---|---|
| `small_d_weight(j2)` | Coefficients of the Wigner small-\(d\) expansion. With \(J=j2/2\), \(\displaystyle d^J_{mn}(\beta)=\sum_k w^J_{mnk}\sin^k\!\frac{\beta}{2}\cos^{2J-k}\!\frac{\beta}{2}\). |
| `small_d_matrix(beta, j2)` | \(\displaystyle d^J_{mn}(\beta)=\sum_k(-1)^{k+m-n}\frac{\sqrt{(J+m)!(J-m)!(J+n)!(J-n)!}}{(J-m-k)!(J+n-k)!k!(k+m-n)!}\cos^{2J+n-m-2k}\!\frac{\beta}{2}\sin^{2k+m-n}\!\frac{\beta}{2}\), with the sum over terms whose factorials are non-negative. |
| `d_matrix_conj(alpha, beta, gamma, j2)` | \(\displaystyle D^{J*}_{mn}(\alpha,\beta,\gamma)=e^{+im\alpha}\,d^J_{mn}(\beta)\,e^{+in\gamma}\). |
| `dfun_delta_v2(d, ja, la, lb, lc)` | Selects \(\displaystyle D^{J*}_{\lambda_a,\Delta\lambda}\) with \(\displaystyle \Delta\lambda=\lambda_b-\lambda_c\). If \(\vert\Delta\lambda\vert>J\), then \(\displaystyle D^{J*}_{\lambda_a,\Delta\lambda}=0\). |
| `get_d_matrix_lambda(angle, ja, la, lb, lc)` | \(\displaystyle \mathcal{D}_{\lambda_a\lambda_b\lambda_c}^{J}(\Omega)=D^{J*}_{\lambda_a,\lambda_b-\lambda_c}(\alpha,\beta,\gamma)\). |
| `cg_coef(j1, j2, m1, m2, j, m)` | \(\displaystyle C(j_1,j_2,j;m_1,m_2,m)=\langle j_1m_1;j_2m_2\vert jm\rangle\). |
| `cg_matrix(ja, jb, jc, ls_list, out_spins)` | For each \((L,S)\): \(\displaystyle C^{LS}_{\lambda_b\lambda_c}=\sqrt{\frac{2L+1}{2J_a+1}}\langle J_b,\lambda_b;J_c,-\lambda_c\vert S,\lambda_b-\lambda_c\rangle\langle L,0;S,\lambda_b-\lambda_c\vert J_a,\lambda_b-\lambda_c\rangle\). |

## Two-Body Helicity Decay Amplitude Functions

| Python function / call | LaTeX formula implemented |
|---|---|
| `helicity_decay_amp(...)` | \(\displaystyle A^{a\to bc}_{\lambda_a\lambda_b\lambda_c}=\mathcal{D}_{\lambda_a\lambda_b\lambda_c}^{J_a}(\Omega)\,H_{\lambda_b\lambda_c}\). |
| `m_dep = g_ls * barrier_factor2(...)` | \(\displaystyle M_{LS}(m)=g_{LS}F_L(q,q_0,d)\). If `has_barrier=False`, then \(\displaystyle M_{LS}(m)=g_{LS}\). |
| `h = sum(m_dep * cg)` | \(\displaystyle H_{\lambda_b\lambda_c}=\sum_{(L,S)}M_{LS}(m)\,C^{LS}_{\lambda_b\lambda_c}\). |
| `amp_bp = helicity_decay_amp(...)` | \(\displaystyle A^B_{\lambda_B,\lambda_\psi,\lambda_K}=D^{0*}_{\lambda_B,\lambda_\psi-\lambda_K}(\Omega_B^\psi)\,H^B_{\lambda_\psi\lambda_K}\), with \(\lambda_B=0,\lambda_K=0,(L,S)=(1,1)\). |
| `amp_psi = helicity_decay_amp(...)` | \(\displaystyle A^\psi_{\lambda_\psi,\lambda_{D^*},\lambda_D}=D^{1*}_{\lambda_\psi,\lambda_{D^*}-\lambda_D}(\Omega_\psi^{D^*})\,H^\psi_{\lambda_{D^*}\lambda_D}\), with \(\lambda_D=0,(L,S)=(1,1)\). |
| `amp_dst = helicity_decay_amp(..., has_barrier=False)` | \(\displaystyle A^{D^*}_{\lambda_{D^*},\lambda_{D^0},\lambda_\pi}=D^{1*}_{\lambda_{D^*},\lambda_{D^0}-\lambda_\pi}(\Omega_{D^*}^{D^0})\,H^{D^*}_{\lambda_{D^0}\lambda_\pi}\), with \(\lambda_{D^0}=\lambda_\pi=0,(L,S)=(1,0),F_L=1\). |

## Psi(4040) Particle Factor Functions

| Python function / call | LaTeX formula implemented |
|---|---|
| `gamma_running(m, gamma0, q, q0, l, m0, d)` | \(\displaystyle \Gamma(m)=\Gamma_0\left(\frac{q}{q_0}\right)^{2L+1}\frac{m_0}{m}\frac{B_L(q_0^2d^2)}{B_L(q^2d^2)}\). |
| `bwr(m, m0, gamma0, q, q0, l, d)` | \(\displaystyle R_{\mathrm{BWR}}(m)=\frac{m_0^2-m^2+i\,m_0\Gamma(m)}{(m_0^2-m^2)^2+\left[m_0\Gamma(m)\right]^2}\). |
| `q_psi = np.sqrt(q2_psi)` | \(\displaystyle q_\psi=\sqrt{q_\psi^2}\). |
| `q0_psi = np.sqrt(q02_psi)` | \(\displaystyle q_{0,\psi}=\sqrt{q_{0,\psi}^2}\). |
| `psi_factor = bwr(...)` | \(\displaystyle R_\psi=R_{\mathrm{BWR}}(m_\psi^{\mathrm{evt}};m_\psi^0,\Gamma_\psi^0,q_\psi,q_{0,\psi},L=1,d=3)\). |
| `dst_factor = np.ones_like(psi_factor)` | \(\displaystyle R_{D^*}=1\). |
| `particle_factor = g_total * psi_factor * dst_factor` | \(\displaystyle P_{\mathrm{chain}}=g_{\mathrm{total}}R_\psi R_{D^*}=R_\psi\). |

## Final Helicity-Chain Contraction

| Python function / call | LaTeX formula implemented |
|---|---|
| `np.einsum("...agd,...gfb,...fce,...->...abcde", amp_bp, amp_psi, amp_dst, particle_factor)` | \(\displaystyle A_{abcde}=P_{\mathrm{chain}}\sum_{\lambda_\psi=-1}^{1}\sum_{\lambda_{D^*}=-1}^{1}A^B_{a,\lambda_\psi,d}\,A^\psi_{\lambda_\psi,\lambda_{D^*},b}\,A^{D^*}_{\lambda_{D^*},c,e}\). |
| `amplitude = amplitude_tensor.reshape(-1)[0]` | Since all external particles are spinless, \(\displaystyle A_{\psi(4040)}=A_{0,0,0,0,0}\). |
| `tfpwa_reference = complex(...)` | \(\displaystyle A_{\mathrm{TF-PWA}}=-0.0006049977354135942-0.0030870270680673287\,i\). |
| `delta = amplitude - tfpwa_reference` | \(\displaystyle \Delta A=A_{\psi(4040)}-A_{\mathrm{TF-PWA}}\). |

## Full Hand-Calculation Recipe

For the given input four-vectors, first reconstruct:

```math
p_{D^*}=p_{D^0}+p_\pi,\qquad
p_\psi=p_{D^*}+p_D,\qquad
p_B=p_\psi+p_K.
```

Then compute:

```math
m_X=\sqrt{p_X^2},\qquad
q_B^2=q^2(m_B;m_\psi,m_K),\qquad
q_\psi^2=q^2(m_\psi;m_{D^*},m_D),\qquad
q_{D^*}^2=q^2(m_{D^*};m_{D^0},m_\pi).
```

The vertex amplitudes are:

```math
A^B_{0,\lambda_\psi,0}
=
D^{0*}_{0,\lambda_\psi}(\Omega_B^\psi)
F_1(q_B,q_{0,B},d)
\sqrt{3}
\langle 1,\lambda_\psi;0,0\vert 1,\lambda_\psi\rangle
\langle 1,0;1,\lambda_\psi\vert 0,\lambda_\psi\rangle,
```

```math
A^\psi_{\lambda_\psi,\lambda_{D^*},0}
=
D^{1*}_{\lambda_\psi,\lambda_{D^*}}(\Omega_\psi^{D^*})
F_1(q_\psi,q_{0,\psi},d)
\sqrt{1}
\langle 1,\lambda_{D^*};0,0\vert 1,\lambda_{D^*}\rangle
\langle 1,0;1,\lambda_{D^*}\vert 1,\lambda_{D^*}\rangle,
```

```math
A^{D^*}_{\lambda_{D^*},0,0}
=
D^{1*}_{\lambda_{D^*},0}(\Omega_{D^*}^{D^0})
\sqrt{1}
\langle 0,0;0,0\vert 0,0\rangle
\langle 1,0;0,0\vert 1,0\rangle.
```

Finally:

```math
A_{\psi(4040)}
=
R_{\mathrm{BWR}}(m_\psi)
\sum_{\lambda_\psi=-1}^{1}
\sum_{\lambda_{D^*}=-1}^{1}
A^B_{0,\lambda_\psi,0}
A^\psi_{\lambda_\psi,\lambda_{D^*},0}
A^{D^*}_{\lambda_{D^*},0,0}.
```

With the notebook inputs, this evaluates to:

```math
A_{\psi(4040)}
=
-0.0006049977356379838
-0.003087027069212293\,i,
```

which agrees with the live TF-PWA reference within numerical precision.
