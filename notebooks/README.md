# Cascade Decay Amplitude Computation

This document describes the amplitude computation for the cascade decay of a $B$ meson implemented in `tfpwa_model.jl`.

## Decay Chain

The decay proceeds through the following cascade:

$$B \to \psi K$$

where $\psi$ subsequently decays as:

$$\psi \to D^* \bar{D}$$

and $D^*$ decays as:

$$D^* \to D^0 \pi$$

The full decay chain is: $B \to \psi K \to (D^* \bar{D}) K \to (D^0 \pi) \bar{D} K$.

## Kinematic Variables

The decay is described using invariant masses and helicity angles:

- **Invariant masses:**
  - $m_B^2 = (p_1 + p_2 + p_3 + p_4)^2$ - parent $B$ meson
  - $m_\psi^2 = (p_1 + p_2 + p_3)^2$ - intermediate $\psi$ resonance
  - $m_{D^*}^2 = (p_1 + p_2)^2$ - intermediate $D^*$ resonance
  - Individual particle masses: $m_1^2 = p_1^2$, $m_2^2 = p_2^2$, $m_3^2 = p_3^2$, $m_4^2 = p_4^2$

- **Helicity angles:**
  - $(\cos\theta_B, \phi_B)$ - angles of the $(1,2,3)$ system in the $B$ rest frame
  - $(\cos\theta_\psi, \phi_\psi)$ - angles of the $(1,2)$ system in the $\psi$ rest frame
  - $(\cos\theta_{D^*}, \phi_{D^*})$ - angles of particle 1 in the $D^*$ rest frame

## Two-Body Decay Amplitude

For a two-body decay $0 \to 1 + 2$, the amplitude in the helicity basis is:

$$\mathcal{A}_{\lambda_1 \lambda_2}^{\lambda_0} = \sqrt{2j_0 + 1} \, D_{\lambda_0,\lambda_1-\lambda_2}^{j_0*}(\phi, \cos\theta, 0) \, \mathcal{A}_{\text{rec}} \, F(m_0^2, m_1^2, m_2^2)$$

where:
- $j_0$ is the spin of the parent particle (in units of $\hbar$)
- $\lambda_0, \lambda_1, \lambda_2$ are the helicities of particles 0, 1, and 2
- $D_{\lambda,\Delta\lambda}^{j*}(\phi, \cos\theta, 0)$ is the complex conjugate of the Wigner $D$-function
- $\mathcal{A}_{\text{rec}}$ is the recoupling amplitude (LS coupling coefficient)
- $F(m_0^2, m_1^2, m_2^2)$ is the form factor (Blatt-Weisskopf barrier factor)

### Wigner D-Function

The Wigner $D$-function describes the rotation from the helicity frame to the lab frame:

$$D_{\lambda,\Delta\lambda}^{j}(\phi, \cos\theta, 0) = d_{\lambda,\Delta\lambda}^{j}(\cos\theta) \, e^{i\lambda\phi}$$

where $d_{\lambda,\Delta\lambda}^{j}(\cos\theta)$ is the Wigner small $d$-function.

### Recoupling Amplitude

The recoupling amplitude $\mathcal{A}_{\text{rec}}$ connects the helicity basis to the LS coupling basis:

$$\mathcal{A}_{\text{rec}} = \sqrt{\frac{2l+1}{2j_0+1}} \, \langle j_1, \lambda_1; j_2, -\lambda_2 | s, \Delta\lambda \rangle \, \langle l, 0; s, \Delta\lambda | j_0, \Delta\lambda \rangle$$

where:
- $l$ and $s$ are the orbital angular momentum and total spin in the LS coupling scheme
- $\Delta\lambda = \lambda_1 - \lambda_2$ is the helicity difference
- The Clebsch-Gordan coefficients couple the daughter spins to the total spin $s$, and then couple the orbital and spin angular momenta to the parent spin $j_0$

### Form Factor

The Blatt-Weisskopf form factor accounts for the angular momentum barrier:

$$F_l(q^2) = \frac{1}{\sqrt{1 + (q R)^2}}$$

for $l=0$, and more complex expressions for higher $l$, where:
- $q$ is the breakup momentum
- $R = d_0$ is the interaction radius (typically $d_0 = 3$ GeV$^{-1}$)

## Cascade Amplitude

For the full cascade decay, the amplitude is constructed by summing over intermediate helicities:

$$\mathcal{A} = \sum_{\lambda_\psi = -1}^{1} \sum_{\lambda_{D^*}=-1}^{1} \mathcal{A}_{B,\lambda_\psi 0}^{\lambda_B} \, \mathcal{A}_{\psi,\lambda_{D^*} 0}^{\lambda_\psi} \, \mathcal{A}_{D^*,00}^{\lambda_{D^*}}$$

where:
- $\mathcal{A}_{B,\lambda_\psi 0}^{\lambda_B}$ is the amplitude for $B \to \psi K$ with $\psi$ helicity $\lambda_\psi$ and $K$ helicity 0 (parent $B$ has helicity $\lambda_B = 0$)
- $\mathcal{A}_{\psi,\lambda_{D^*} 0}^{\lambda_\psi}$ is the amplitude for $\psi \to D^* \bar{D}$ with $D^*$ helicity $\lambda_{D^*}$, $\bar{D}$ helicity 0, and parent $\psi$ helicity $\lambda_\psi$
- $\mathcal{A}_{D^*,00}^{\lambda_{D^*}}$ is the amplitude for $D^* \to D^0 \pi$ with both daughters having helicity 0 and parent $D^*$ helicity $\lambda_{D^*}$

## Spin Configurations

The spin assignments are:
- $B$: $J^P = 0^-$ (pseudoscalar)
- $\psi$: $J^P = 1^-$ (vector)
- $D^*$: $J^P = 1^-$ (vector)
- $K, \bar{D}, D^0, \pi$: $J^P = 0^-$ (pseudoscalars)

For each two-body decay:
- $B \to \psi K$: $0^- \to 1^- + 0^-$ (requires $l=1, s=1$)
- $\psi \to D^* \bar{D}$: $1^- \to 1^- + 0^-$ (requires $l=1, s=1$)
- $D^* \to D^0 \pi$: $1^- \to 0^- + 0^-$ (requires $l=1, s=0$)

## Implementation Details

The implementation uses:
- **TwoBodySystem**: Stores masses and spin quantum numbers
- **TwoBodyDecay**: Combines a TwoBodySystem with a VertexFunction
- **VertexFunction**: Contains the recoupling coefficient and form factor
- **SimpleCascade**: Chains three TwoBodyDecay objects together
- **SphericalAngles**: Stores $(\cos\theta, \phi)$ for each decay vertex

The amplitude computation follows the cascade structure, computing each two-body decay in sequence and summing over intermediate helicity states.
