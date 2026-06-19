using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using ThreeBodyDecays
using LinearAlgebra
using Parameters
using JSON
using ThreeBodyDecays.StaticArrays

# --- Configuration ---
# Constants (matching final_params_full.json and config_a.yml)
const mB = 5.27934
const mD = 1.86965
const mDx = 2.01026
const mK = 0.493677
const mD0 = 1.86483
const mpi = 0.13957039

# Helpers
x2(x) = Int(2x)
d2(x) = div(x, 2)

# --- Lineshapes ---
struct SafeBreitWigner
    m0::Float64
    Γ0::Float64
    ma::Float64
    mb::Float64
    l::Int
    d::Float64
end

function breakup_safe(m, ma, mb)
    val = (m^2 - (ma + mb)^2) * (m^2 - (ma - mb)^2)
    return val < 0.0 ? sqrt(abs(val)) / (2 * m) : sqrt(val) / (2 * m)
end

function (bw::SafeBreitWigner)(σ::Float64)
    m = sqrt(σ)
    q = breakup_safe(m, bw.ma, bw.mb)
    q0 = breakup_safe(bw.m0, bw.ma, bw.mb)
    z = (q * bw.d)^2
    z0 = (q0 * bw.d)^2
    F2 = 1.0; F2_0 = 1.0
    if bw.l == 1
        F2 = 1.0 / (1.0 + z); F2_0 = 1.0 / (1.0 + z0)
    elseif bw.l == 2
        F2 = 1.0 / (9.0 + 3 * z + z^2); F2_0 = 1.0 / (9.0 + 3 * z0 + z0^2)
    end
    ratio_q = (q / q0)^(2 * bw.l + 1)
    Γm = bw.Γ0 * ratio_q * (bw.m0 / m) * (F2 / F2_0)
    return 1.0 / (bw.m0^2 - m^2 - im * bw.m0 * Γm)
end

# --- Kinematics ---
function calculate_angles(pD, pK, pDx, pD0)
    beta_Dx = pDx[2:4] ./ pDx[1]
    function boost(p, beta)
        b2 = sum(beta .^ 2)
        gamma = 1.0 / sqrt(1.0 - b2)
        bp = sum(p[2:4] .* beta)
        gamma2 = b2 < 1e-10 ? 0.5 : (gamma - 1.0) / b2
        E_new = gamma * (p[1] - bp)
        p_new = p[2:4] .+ (gamma2 * bp - gamma * p[1]) .* beta
        return [E_new; p_new]
    end
    pD0_DxRF = boost(pD0, beta_Dx)
    z_axis = pDx[2:4] / norm(pDx[2:4])
    y_axis = cross(pD[2:4], pK[2:4]); y_axis /= norm(y_axis)
    x_axis = cross(y_axis, z_axis)
    pD0_3 = pD0_DxRF[2:4]
    px = dot(pD0_3, x_axis); py = dot(pD0_3, y_axis); pz = dot(pD0_3, z_axis)
    return pz / norm(pD0_3), atan(py, px)
end

# Load Data
vecs_path = joinpath(@__DIR__, "..", "..", "..", "archive", "data", "crosscheck_event.json")
data = JSON.parsefile(vecs_path)
pD_vec = [data["four_vectors"]["D"]["E"], data["four_vectors"]["D"]["px"], data["four_vectors"]["D"]["py"], data["four_vectors"]["D"]["pz"]]
pD0_vec = [data["four_vectors"]["D0"]["E"], data["four_vectors"]["D0"]["px"], data["four_vectors"]["D0"]["py"], data["four_vectors"]["D0"]["pz"]]
pK_vec = [data["four_vectors"]["K"]["E"], data["four_vectors"]["K"]["px"], data["four_vectors"]["K"]["py"], data["four_vectors"]["K"]["pz"]]
ppi_vec = [data["four_vectors"]["pi"]["E"], data["four_vectors"]["pi"]["px"], data["four_vectors"]["pi"]["py"], data["four_vectors"]["pi"]["pz"]]
pDx_vec = pD0_vec + ppi_vec

m2_vec(p) = p[1]^2 - p[2]^2 - p[3]^2 - p[4]^2
σs = (σ1=m2_vec(pK_vec + pDx_vec), σ2=m2_vec(pDx_vec + pD_vec), σ3=m2_vec(pD_vec + pK_vec))
cosθ, ϕ = calculate_angles(pD_vec, pK_vec, pDx_vec, pD0_vec)

# --- Model ---
struct DalitzAndDecay{T}
    σs::MandelstamTuple{T}
    cosθ::T
    ϕ::T
end

function ThreeBodyDecays.amplitude(three_body_model::ThreeBodyDecay, dd::DalitzAndDecay)
    @unpack σs, cosθ, ϕ = dd
    _O = amplitude(three_body_model, σs)
    _D = [wignerD(1, λ, 0, ϕ, cosθ, 0.0) for λ in -1:1] .|> conj
    return sum(reshape(_O, 3) .* _D)
end

params = JSON.parsefile(joinpath(@__DIR__, "..", "..", "..", "data", "final_params_full.json"))["value"]
tbs = ThreeBodySystem(ThreeBodyMasses(mD, mK, mDx; m0=mB), ThreeBodySpinParities("0-", "0-", "1-"; jp0="0+")[1])

# Filter and iterate for Psi(4040)
res_name = "Psi(4040)"
mR, ΓR = params["Psi(4040)_mass"], params["Psi(4040)_width"]
lineshape = SafeBreitWigner(mR, ΓR, mD, mDx, 1, 3.0)

# Psi(4040) is 1- -> D*D (1- + 0-) -> l=1, s=1
# Prod: B+(0-) -> Psi(4040)(1-) + K(0-) -> L=1, S=1
ch = DecayChain(; k=2, two_j=2, Xlineshape=lineshape, 
    HRk=VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)), 
    Hij=VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)), tbs)

# Normalization Factor (calculate_normalization_factor simplified)
q0_p = breakup_safe(mB, mR, mK)
q0_d = breakup_safe(mR, mD, mDx)
bw_f(q0, L) = L == 0 ? 1.0 : (L == 1 ? sqrt(1.0 / (1.0 + (q0 * 3.0)^2)) : sqrt(1.0 / (9.0 + 3 * (q0 * 3.0)^2 + (q0 * 3.0)^4)))
norm_factor = (q0_p^1 * bw_f(q0_p, 1)) * (q0_d^1 * bw_f(q0_d, 1))

model = ThreeBodyDecay(["Psi(4040)_l1_L1"] .=> zip([1.0/norm_factor], [ch]))
val = amplitude(model, DalitzAndDecay(σs, cosθ, ϕ))

# Save Result
output_path = joinpath(@__DIR__, "..", "..", "..", "archive", "investigation", "Analysis", "julia_psi4040_amplitudes.txt")
open(output_path, "w") do f
    println(f, "Resonance (Psi(4040) [L=1, l=1]): $(real(val)) $(imag(val) >= 0 ? "+" : "-") $(abs(imag(val)))im")
end

println("Psi(4040) result saved to $output_path")
println("Value: $(real(val)) $(imag(val) >= 0 ? "+" : "-") $(abs(imag(val)))im")
