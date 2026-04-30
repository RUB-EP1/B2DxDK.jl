# CascadeDecays aligned to the isolated TF-PWA Psi(4040) probe
#
# This script establishes a running CascadeDecays setup inside the playground
# project and evaluates the same isolated Psi(4040) amplitude as
# Analysis/psi4040_independent_amplitude_flow.ipynb.
#
# The file keeps only the pieces needed for the aligned amplitude calculation
# and the remaining constant normalization-factor correction to the TF-PWA
# convention.

using CascadeDecays
using FourVectors
using HadronicLineshapes
using ThreeBodyDecays: VertexFunction, RecouplingLS
using Printf

const TARGET_TFPWA = -0.0006049977356379836 - 0.003087027069212291im

# Same nominal masses as Analysis/config_a.yml and the same Psi(4040)
# pole parameters as Analysis/final_params_full.json.
const nominal_mass = Dict(
    "Bp" => 5.27934,
    "D" => 1.86965,
    "K" => 0.493677,
    "D0" => 1.86483,
    "pi" => 0.13957039,
    "Dst" => 2.01026,
    "Psi(4040)" => 4.039,
)
const psi_width = 0.08

# Same hardcoded four-vectors as the isolated Python probe, ordered as
# (E, px, py, pz) in the source notebook. FourVectors.FourVector expects
# (px, py, pz; E=...).
const pDminus = FourVector(-0.1467, 0.2235, -0.7847; E = 2.0452)
const pD0 = FourVector(0.2284, -0.3689, 1.2019; E = 2.2606)
const pKplus = FourVector(-0.0873, 0.1803, -0.5584; E = 0.7718)
const piplus = FourVector(0.0056, -0.0349, 0.1413; E = 0.2017)

const objs = (pD0, piplus, pDminus, pKplus)
const P_Dx = pD0 + piplus
const P_psi = P_Dx + pDminus
const P_B = P_psi + pKplus

const topology = DecayTopology((((1, 2), 3), 4))
const system = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., mass(P_B)^2))
const x = cascade_kinematics(topology, system, objs)

get_relative_p2(m0, m1, m2) = ((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4 * m0^2)

function build_package_native_chain()
    vertices = (
        (((1, 2), 3), 4) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        ((1, 2), 3) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        (1, 2) => VertexFunction(RecouplingLS((2, 0))),
    )
    propagators = (
        (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0.0im)),
        ((1, 2), 3) => (
            two_j = 2,
            lineshape = BreitWigner(
                nominal_mass["Psi(4040)"],
                psi_width,
                nominal_mass["Dst"],
                nominal_mass["D"],
                1,
                3.0,
            ),
        ),
    )
    return DecayChain(topology; propagators, vertices)
end

function mismatch_factor(l, d, m0, m1, m2)
    # Package-native BlattWeisskopf is unnormalized at the nominal point q0.
    # TF-PWA's normalized barrier convention differs by exactly 1 / FF(q0).
    ff = BlattWeisskopf{l}(d)
    return 1 / ff(m0^2, m1^2, m2^2)
end

function main()
    println("CascadeDecays aligned Psi(4040) amplitude")
    println("=========================================")
    println("TF-PWA target amplitude = ", TARGET_TFPWA)
    println()

    package_chain = build_package_native_chain()
    psi_bw = BreitWigner(
        nominal_mass["Psi(4040)"],
        psi_width,
        nominal_mass["Dst"],
        nominal_mass["D"],
        1,
        3.0,
    )
    a_package = amplitude(package_chain, system, x, (0, 0, 0, 0, 0))

    event_mass = Dict(
        "Bp" => mass(P_B),
        "Psi(4040)" => mass(P_psi),
        "Dst" => mass(P_Dx),
        "D" => mass(pDminus),
        "K" => mass(pKplus),
        "D0" => mass(pD0),
        "pi" => mass(piplus),
    )

    root_angles = vertex_angles(topology, x, (((1, 2), 3), 4))
    psi_angles = vertex_angles(topology, x, ((1, 2), 3))
    dst_angles = vertex_angles(topology, x, (1, 2))

    fb = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["Psi(4040)"], nominal_mass["K"])
    fpsi = mismatch_factor(1, 3.0, nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])
    total_factor = fb * fpsi
    a_corrected = a_package * total_factor
    psi_factor_package = psi_bw(event_mass["Psi(4040)"]^2)

    println("Step 1: Reconstruct intermediate four-vectors by summing daughters.")
    println("  Final-state four-vectors are direct package inputs.")
    println(@sprintf("  p_D0  = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pD0[4], pD0[1], pD0[2], pD0[3]))
    println(@sprintf("  p_pi  = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", piplus[4], piplus[1], piplus[2], piplus[3]))
    println(@sprintf("  p_D   = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pDminus[4], pDminus[1], pDminus[2], pDminus[3]))
    println(@sprintf("  p_K   = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pKplus[4], pKplus[1], pKplus[2], pKplus[3]))
    println("  Intermediate four-vectors (Dst, Psi(4040), Bp) are not provided directly by the package-native interface.")
    println()

    println("Step 2: Compute invariant masses from the event kinematics.")
    println("  External/root masses are direct package inputs through CascadeSystem.")
    for name in ["D0", "pi", "D", "K", "Bp"]
        println(@sprintf("  m(%s) = %.12f GeV", name, event_mass[name]))
    end
    println("  Internal masses m(Dst) and m(Psi(4040)) are routed internally by cascade_kinematics and are not printed here as direct package inputs.")
    println()

    println("Step 3: Compute helicity angles in the package convention.")
    println("  theta(beta) is reconstructed from the package-native cos(theta) output.")
    println(@sprintf("  Bp -> Psi(4040) K:   theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(root_angles.cosθ), root_angles.ϕ))
    println(@sprintf("  Psi -> Dst D:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(psi_angles.cosθ), psi_angles.ϕ))
    println(@sprintf("  Dst -> D0 pi:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(dst_angles.cosθ), dst_angles.ϕ))
    println()

    println("Step 4: Compute breakup momenta.")
    println("  q2 and q0^2 are not provided directly by the package-native interface.")
    println()

    println("Step 5: Vertex model inputs from HadronicLineshapes / ThreeBodyDecays.")
    println("  Bp -> Psi K vertex:   RecouplingLS((2,2)) with BlattWeisskopf{1}(3.0)")
    println("  Psi -> Dst D vertex:  RecouplingLS((2,2)) with BlattWeisskopf{1}(3.0)")
    println("  Dst -> D0 pi vertex:  RecouplingLS((2,0))")
    println()

    println("Step 6: Particle factor for Psi(4040).")
    println("  Propagator: BreitWigner(4.039, 0.08, 2.01026, 1.86965, 1, 3.0)")
    println("  Package-native Psi(4040) factor at sigma = m(Psi)^2")
    println("    ", psi_factor_package)
    println()

    println("Step 7: Final complex amplitude and normalization correction.")

    println("Package-native amplitude                  = ", a_package)
    println("Mismatch factor Bp vertex                = ", fb)
    println("Mismatch factor Psi vertex               = ", fpsi)
    println("Total mismatch factor                    = ", total_factor)
    println("Corrected package amplitude              = ", a_corrected)
    println("Corrected delta to TF-PWA target         = ", a_corrected - TARGET_TFPWA)
end

main()