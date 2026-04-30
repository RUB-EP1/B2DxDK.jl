# Execution Flow Steps 1-7

This document describes Steps 1-7 of the execution flow of the isolated
Python notebook that computes the complex amplitude for `\Psi(4040)`.

## Step 1: Reconstruct the Intermediate Four-Vectors

Given the final-state four-vectors

\[
p_{D^0}, \quad p_{\pi}, \quad p_D, \quad p_K,
\]

construct the intermediate and parent four-vectors by four-momentum addition:

\[
p_{D^*} = p_{D^0} + p_{\pi},
\]

\[
p_{\psi(4040)} = p_{D^*} + p_D,
\]

\[
p_{B^+} = p_{\psi(4040)} + p_K.
\]

This fixes the complete decay chain

\[
B^+ \to \psi(4040)\,K,
\qquad
\psi(4040) \to D^* D,
\qquad
D^* \to D^0 \pi.
\]

## Step 2: Compute the Invariant Masses

For any four-vector

\[
p = (E, p_x, p_y, p_z),
\]

the invariant mass is defined as

\[
m(p) = \sqrt{\left|E^2 - p_x^2 - p_y^2 - p_z^2\right|}
= \sqrt{|p^2|}.
\]

Apply this to all reconstructed particles:

\[
m_{B^+} = m(p_{B^+}),
\]

\[
m_{\psi(4040)} = m(p_{\psi(4040)}),
\]

\[
m_{D^*} = m(p_{D^*}),
\]

\[
m_D = m(p_D),
\qquad
m_K = m(p_K),
\qquad
m_{D^0} = m(p_{D^0}),
\qquad
m_{\pi} = m(p_{\pi}).
\]

## Step 3: Compute the Helicity Angles Along the Chain

The decay chain is represented as

```python
chain = [
    ("B+", ["psi4040", "K"]),
    ("psi4040", ["D*", "D"]),
    ("D*", ["D0", "pi"]),
]
```

The helicity-angle routine is written schematically as

```python
calculate_helicity_angles(particle_p4, chain)
```

and internally uses

```python
compute_chain_boosts(...)
```

to move all daughter four-vectors into the rest frame of their parent.

### 3.1 Rest-Frame Boost

The rest-frame mapping is

```python
rest_vector(core_p4, other_p4)
boost(other_p4, -boost_vector(core_p4))
```

Here `core_p4` denotes the parent-particle four-vector and `other_p4`
denotes the daughter-particle four-vector that is boosted into the parent
rest frame. To distinguish their components clearly, write

\[
p_{\mathrm{core}} = (E_{\mathrm{core}}, \mathbf{p}_{\mathrm{core}}),
\qquad
p_{\mathrm{other}} = (E_{\mathrm{other}}, \mathbf{p}_{\mathrm{other}}).
\]

The boost is constructed from the parent (`core`) momentum, with

\[
\beta_i = \frac{(p_{\mathrm{core}})_i}{E_{\mathrm{core}}},
\qquad
\beta^2 = \sum_i \beta_i^2,
\qquad
\gamma = \frac{1}{\sqrt{1-\beta^2}}.
\]

For a boost with velocity vector \(\boldsymbol{\beta}\), the transformed
daughter four-vector

\[
p'_{\mathrm{other}} = (E'_{\mathrm{other}}, \mathbf{p}'_{\mathrm{other}})
\]

is

\[
E'_{\mathrm{other}}
=
\gamma \left(
E_{\mathrm{other}}
-
\sum_i \beta_i \,(p_{\mathrm{other}})_i
\right),
\]

\[
\mathbf{p}'_{\mathrm{other}}
=
\mathbf{p}_{\mathrm{other}}
+
\left[
\frac{\gamma - 1}{\beta^2}
\left(\sum_i \beta_i \,(p_{\mathrm{other}})_i\right)
- \gamma E_{\mathrm{other}}
\right]
\boldsymbol{\beta}.
\]

The explicit `+` signs are intentional here: the boosted spatial momentum is
the original daughter momentum plus the boost correction term.

Equivalently, component by component,

\[
E'_{\mathrm{other}}
=
\gamma \left(
E_{\mathrm{other}}
-
\sum_i \beta_i \,(p_{\mathrm{other}})_i
\right),
\]

\[
(p'_{\mathrm{other}})_j
=
(p_{\mathrm{other}})_j
+
\left[
\frac{\gamma - 1}{\beta^2}
\left(\sum_i \beta_i \,(p_{\mathrm{other}})_i\right)
- \gamma E_{\mathrm{other}}
\right]
\beta_j.
\]

### 3.2 Boost Catalogue Along the Chain

The boosted particle data are organized as follows:

1. In the \(B^+\) rest frame:

\[
\psi(4040)', \quad K'
\]

2. In the \(\psi(4040)\) rest frame:

\[
D^{*\,\prime\prime}, \quad D^{\prime\prime}
\]

3. In the \(D^*\) rest frame:

\[
D^0,\quad \pi
\]

where each pair is expressed in the rest frame of its parent.

### 3.3 Euler-Angle Construction

The angles are defined through

```python
angle_zx_z_getx(z1=(0,0,1), x1=(1,0,0), z2=p_i(part_data))
```

with the normalized basis vectors

\[
\hat{\mathbf{z}}_1 = \frac{\mathbf{z}_1}{|\mathbf{z}_1|} = (0,0,1),
\qquad
\hat{\mathbf{x}}_1 = (1,0,0).
\]

Then

\[
\hat{\mathbf{z}}_2 = \frac{\mathbf{z}_2}{|\mathbf{z}_2|},
\]

\[
\hat{\mathbf{y}}_1
=
\frac{\mathbf{z}_1 \times \mathbf{x}_1}{|\mathbf{z}_1 \times \mathbf{x}_1|}
= (0,1,0),
\]

\[
\hat{\mathbf{x}}_1
=
\frac{\hat{\mathbf{y}}_1 \times \mathbf{z}_1}{|\hat{\mathbf{y}}_1 \times \mathbf{z}_1|}
= (1,0,0),
\]

\[
\hat{\mathbf{y}}_r
=
\frac{\mathbf{z}_1 \times \mathbf{z}_2}{|\mathbf{z}_1 \times \mathbf{z}_2|},
\]

\[
\hat{\mathbf{x}}_r
=
\frac{\hat{\mathbf{y}}_r \times \mathbf{z}_1}{|\hat{\mathbf{y}}_r \times \mathbf{z}_1|}.
\]

The Euler angles are then written as

\[
\alpha = \operatorname{atan2}
\left(
\hat{\mathbf{x}}_r \cdot \hat{\mathbf{x}}_1,\,
\hat{\mathbf{x}}_r \cdot \hat{\mathbf{y}}_1
\right),
\]

\[
\beta = \operatorname{atan2}
\left(
\hat{\mathbf{z}}_2 \cdot \hat{\mathbf{z}}_1,\,
\hat{\mathbf{z}}_2 \cdot \hat{\mathbf{x}}_r
\right),
\]

\[
\gamma = 0.
\]

The transported \(x\)-axis for the next decay step is

\[
\hat{\mathbf{x}}_2
=
\frac{\hat{\mathbf{y}}_r \times \hat{\mathbf{z}}_2}
{|\hat{\mathbf{y}}_r \times \hat{\mathbf{z}}_2|}.
\]

After one decay step, the reference axes are updated by

\[
\mathbf{z}_1 \leftarrow \mathbf{z}_2,
\qquad
\mathbf{x}_1 \leftarrow \hat{\mathbf{x}}_2.
\]

The wrapped version of \(\alpha\) is

\[
\alpha \mapsto ((\alpha + \pi) \bmod 2\pi) - \pi,
\]

with an iterative bias shift from one decay level to the next.

### 3.4 Output of Step 3

The result is the set of helicity angles

\[
(\alpha, \beta, \gamma)
\]

for every decay vertex in the chain.

## Step 4: Relative Momenta

The relative two-body momentum squared is defined as

\[
q^2(m_0,m_1,m_2)
=
\frac{
\left(m_0^2 - (m_1+m_2)^2\right)
\left(m_0^2 - (m_1-m_2)^2\right)
}{
4m_0^2
}.
\]

This is applied to the event-by-event masses:

\[
q_{B^+}^2 = q^2(m_{B^+}, m_{\psi(4040)}, m_K),
\]

\[
q_{\psi}^2 = q^2(m_{\psi(4040)}, m_{D^*}, m_D),
\]

\[
q_{D^*}^2 = q^2(m_{D^*}, m_{D^0}, m_{\pi}).
\]

Nominal reference values are also computed:

\[
q_{0,B^+}^2 = q^2(m_{0,B^+}, m_{0,\psi(4040)}, m_{0,K}),
\]

\[
q_{0,\psi}^2 = q^2(m_{0,\psi(4040)}, m_{0,D^*}, m_{0,D}).
\]

## Step 5: Helicity Decay Amplitudes

The vertex calculation is summarized under the label

```python
helicity_decay_amp
```

### 5.1 Barrier Factor

If the decay has a barrier factor,

\[
L = \texttt{ls\_list}[0][0].
\]

The intermediate quantity is

\[
\texttt{tmp} = (q^2)^{L/2}\,B'_L,
\]

with the normalized Blatt-Weisskopf factor

\[
B'_L
=
\sqrt{
\frac{B_L(q_0^2 d^2)}
{B_L(q^2 d^2)}
}.
\]

The relevant polynomials are

\[
B_0(z) = 1,
\]

\[
B_1(z) = 1 + z,
\]

\[
B_2(z) = z^2 + 3z + 9,
\]

\[
B_3(z) = z^3 + 6z^2 + 45z + 225.
\]

If `barrier_factor_norm` is active, then

\[
\texttt{tmp}
\mapsto
\frac{\texttt{tmp}}{|q_0^2|^{L/2}},
\]

so the barrier factor becomes

\[
\mathrm{bf}
=
\frac{q^{L}}{|q_0|^{L}}
\sqrt{
\frac{B_L(q_0^2 d^2)}
{B_L(q^2 d^2)}
}.
\]

The barrier radius is

\[
d = 3.
\]

Thus the mass-dependent vertex factor is

\[
m_{\mathrm{dep}} = g_{A \to BC}\,\mathrm{bf}.
\]

If no barrier factor is used,

\[
m_{\mathrm{dep}} = g_{A \to BC}.
\]

### 5.2 Clebsch-Gordan Structure

The coupling coefficient is

\[
\mathrm{cg}
=
\sqrt{\frac{2L+1}{2j_a+1}}
\,
\langle j_b\,\lambda_b;\,j_c\,(-\lambda_c)\mid S,\,\lambda_b-\lambda_c\rangle
\,
\langle L\,0;\,S,\,\lambda_b-\lambda_c\mid j_a,\,\lambda_b-\lambda_c\rangle,
\]

for the decay

\[
a \to b\,c,
\]

where

\[
j_a = \text{spin of parent},
\quad
j_b = \text{spin of daughter } b,
\quad
j_c = \text{spin of daughter } c.
\]

The helicity coupling is then

\[
h = \sum \mathrm{cg}\; m_{\mathrm{dep}}.
\]

### 5.3 Wigner-\(D\) Factor

The conjugated \(D\)-matrix element is

\[
D^{j *}_{mn}(\alpha,\beta,\gamma)
=
e^{i m \alpha}
e^{i n \gamma}
d^j_{mn}(\beta),
\]

with

\[
m \in [-j_a, j_a].
\]

The reduced Wigner function is

\[
d^j_{mn}(\beta)
=
\sum_k
(-1)^{k+m-n}
\frac{
\sqrt{(j+m)!(j-m)!(j+n)!(j-n)!}
}{
(j-m-k)!(j+n-k)!k!(k+m-n)!
}
\cos^{\,2j+n-m-2k}\!\left(\frac{\beta}{2}\right)
\sin^{\,2k+m-n}\!\left(\frac{\beta}{2}\right).
\]

The helicity selection rule is

\[
D^{j *}_{\lambda_a,\lambda_b-\lambda_c}
=
\begin{cases}
0,
& |\lambda_b-\lambda_c| > j,
\\[6pt]
D^{j *}_{\lambda_a,\lambda_b-\lambda_c}(\alpha,\beta,\gamma),
& |\lambda_b-\lambda_c| \le j.
\end{cases}
\]

### 5.4 Vertex Amplitude

The final vertex amplitude for one particle decay is

\[
A^{\text{particle}} = h \, D^{j *}.
\]

## Step 6: Resonance Line Shape for \(\psi(4040)\)

Next, take square roots of the relative momenta:

\[
q_{\psi} = \sqrt{q_{\psi}^2},
\qquad
q_{0,\psi} = \sqrt{q_{0,\psi}^2}.
\]

Then the \(\psi(4040)\) factor is computed with a running-width
Breit-Wigner, denoted `bwr`.

The resonance factor is introduced as

\[
\Psi_{\mathrm{factor}} = R(m_{\psi(4040)}),
\]

and is written as

\[
R(m)
=
\frac{
m_0^2 - m^2 + i\,m_0\,\Gamma(m)
}{
\left(m_0^2 - m^2\right)^2 + \left(m_0\Gamma(m)\right)^2
}.
\]

The running width is

\[
\Gamma(m)
=
\Gamma_0
\left(\frac{q}{q_0}\right)^{2L+1}
\left(\frac{m_0}{m}\right)
\frac{B_L(q_0^2 d^2)}{B_L(q^2 d^2)}.
\]

For the \(D^*\) propagator, the factor is set to

\[
\mathrm{D}^*_{\mathrm{factor}} = \operatorname{ones\_like}(\Psi_{\mathrm{factor}}) = 1,
\]

written in code form as `ones_like(psi_factor)`.

## Step 7: Final Chain Contraction

The total chain factor is

\[
P_{\text{chain}}
=
g_{\text{total}}
\times
\Psi_{\mathrm{factor}}
\times
\mathrm{D}^*_{\mathrm{factor}}.
\]

The full tensor amplitude is then

\[
A_{abcde}
=
P_{\text{chain}}
\sum_{\lambda_{\psi}=-1}^{1}
\sum_{\lambda_{D^*}=-1}^{1}
A^B_{a,\lambda_{\psi},d}
A^{\psi}_{\lambda_{\psi},\lambda_{D^*},b}
A^{D^*}_{\lambda_{D^*},c,e}.
\]

Finally, the complex amplitude of interest is the helicity component

\[
A_{00000} = A_{\psi(4040)}.
\]

This is the final complex number returned by the isolated execution flow.
