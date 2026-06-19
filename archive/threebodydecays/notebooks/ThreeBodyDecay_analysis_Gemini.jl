using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
# Pkg.instantiate()

using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using ThreeBodyDecays
using LinearAlgebra
using Parameters
using JSON
using Random
using ThreeBodyDecays.StaticArrays

# --- Configuration ---
Random.seed!(42)

# Helpers
x2(x) = Int(2x)
d2(x) = div(x, 2)

begin
    const mB = 5.27934 # B+
    const mD = 1.86965 # D+
    const mDx = 2.01026 # Dx+: m(D) + Δm(D*,D) from PDG
    const mK = 0.493677 # K+
end;

(two_js, pc), (_, pv) = map(["0+", "0-"]) do jp0
    ThreeBodySpinParities("0-", "0-", "1-"; jp0)
end;

tbs = let
    ms = ThreeBodyMasses(mD, mK, mDx; m0=mB)
    ThreeBodySystem(ms, two_js)
end;

# --- Lineshapes ---
const EFF = BreitWigner(3.87165, 0.00119);

begin
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))

    const ConstantLineshape = WrapFlexFunction(x -> 1.0)

    # Custom Safe Breit-Wigner (matches tf-pwa for sub-threshold)
    struct SafeBreitWigner
        m0::Float64
        Γ0::Float64
        ma::Float64
        mb::Float64
        l::Int
        d::Float64
    end

    function breakup_safe(m, ma, mb)
        # Kallen lambda
        val = (m^2 - (ma + mb)^2) * (m^2 - (ma - mb)^2)
        if val < 0.0
            # Return magnitude for sub-threshold (important for normalization)
            return sqrt(abs(val)) / (2 * m)
        end
        return sqrt(val) / (2 * m)
    end
end

# Polyfill for validation/LS logic
function complete_l_s_L_S(jp, two_js, parities, dc; k)
    # Simplified Logic
    J_Res = Int(jp.two_j / 2)
    P_Res = jp.p

    # 1. Production
    J_sys = [0, 0, 1]
    J_spec = J_sys[k] # 2->K(0), 3->Dx(1)

    if k == 2
        S_prod = J_Res
        L_prod = J_Res
    elseif k == 3 # Dx spec
        # Need L=S. Lowest allowed
        L_prod = abs(J_Res - 1)
        S_prod = L_prod
    else
        L_prod = J_Res
        S_prod = J_Res
    end
    # Override
    if haskey(dc, :L)
        L_prod = dc.L
        S_prod = L_prod
    end

    # 2. Decay
    if k == 2 # D+Dx
        s_decay = 1
        valid_ls = []
        for l_cand in abs(J_Res - 1):(J_Res+1)
            p_check = (l_cand % 2 == 0) ? '+' : '-'
            if p_check == ((P_Res == '+') ? '+' : '-')
                push!(valid_ls, l_cand)
            end
        end
    elseif k == 3 # D+K
        s_decay = 0
        valid_ls = [J_Res]
    else
        s_decay = 0
        valid_ls = [0]
    end

    if haskey(dc, :l)
        l_decay = dc.l
    else
        l_decay = isempty(valid_ls) ? 0 : minimum(valid_ls)
    end

    return (L=L_prod, S=S_prod, l=l_decay, s=s_decay)
end


function (bw::SafeBreitWigner)(σ::Float64)
    m = sqrt(σ)
    q = breakup_safe(m, bw.ma, bw.mb)
    q0 = breakup_safe(bw.m0, bw.ma, bw.mb)

    # Blatt-Weisskopf Form Factors (F_l^2)
    z = (q * bw.d)^2
    z0 = (q0 * bw.d)^2

    F2 = 1.0 + 0.0im
    F2_0 = 1.0 + 0.0im

    if bw.l == 1
        F2 = 1.0 / (1.0 + z)
        F2_0 = 1.0 / (1.0 + z0)
    elseif bw.l == 2
        F2 = 1.0 / (9.0 + 3 * z + z^2)
        F2_0 = 1.0 / (9.0 + 3 * z0 + z0^2)
    end

    # Running Width
    ratio_q = (q / q0)^(2 * bw.l + 1)
    Γm = bw.Γ0 * ratio_q * (bw.m0 / m) * (F2 / F2_0)

    # Propagator
    denom = bw.m0^2 - m^2 - im * bw.m0 * Γm
    return 1.0 / denom
end


# --- Decay Chains ---
# Definitions will be built using parameters loaded from JSON later.

# --- Amplitude Calculation Helper ---
struct DalitzAndDecay{T}
    σs::MandelstamTuple{T}
    cosθ::T
    ϕ::T
end

function ThreeBodyDecays.amplitude(three_body_model::ThreeBodyDecay, dd::DalitzAndDecay)
    @unpack σs, cosθ, ϕ = dd
    total_amp = 0.0
    jDx = 1
    _O = amplitude(three_body_model, σs) # order: -1,0,1
    _D = [wignerD(jDx, λ, 0, ϕ, cosθ, 0.0) for λ in -1:1] .|> conj # order: -1,0,1
    total_amp = sum(reshape(_O, 3) .* _D)
    return total_amp
end

# --- Kinematics (Loaded from event_vectors.json and calculated) ---
println("\n--- Calculation with vectors from event_vectors.json ---")

# Load vectors
vecs_path = joinpath(@__DIR__, "..", "..", "..", "archive", "investigation", "Analysis", "event_vectors.json")
vecs_data = JSON.parsefile(vecs_path)

# Helper to convert list to SVector or Vector
function to_vec(v)
    return Float64[v[1], v[2], v[3], v[4]]
end

pD_vec = to_vec(vecs_data["vectors"]["D"])
pD0_vec = to_vec(vecs_data["vectors"]["D0"])
pK_vec = to_vec(vecs_data["vectors"]["K"])
ppi_vec = to_vec(vecs_data["vectors"]["pi"])
pDx_vec = to_vec(vecs_data["vectors"]["D"] + vecs_data["vectors"]["D0"] + vecs_data["vectors"]["pi"]) # Wait, Dx is sum or vectors?
# Usually vectors["D"] is list. Julia + is concat for arrays?
# Need elementwise sum. to_vec returns array. + is elementwise for arrays in Julia. Correct.
# BUT pDx_vec is actually calculated below as pD0 + ppi.
# Let's clean up line 130.
# In this decay B -> D K Dx; Dx -> D0 pi.

pD_vec = [2.0452, -0.1467, 0.2235, -0.7847]
pD0_vec = [2.2606, 0.2284, -0.3689, 1.2019]
pK_vec = [0.7718, -0.0873, 0.1803, -0.5584]
ppi_vec = [0.2017, 0.0056, -0.0349, 0.1413]

pDx_vec = pD0_vec + ppi_vec

# Calculate Invariants (σs)
function m2_vec(p)
    return p[1]^2 - p[2]^2 - p[3]^2 - p[4]^2
end

s1 = m2_vec(pK_vec + pDx_vec) # m23 (K, Dx)
s2 = m2_vec(pDx_vec + pD_vec) # m31 (Dx, D)
s3 = m2_vec(pD_vec + pK_vec)  # m12 (D, K)
σs_new = (σ1=s1, σ2=s2, σ3=s3)
println("Calculated σs: ", σs_new)

# Calculate Angles (Helicity Frame of Dx)
# Replicating logic consistent with tf_pwa reference
function calculate_angles(pD, pK, pDx, pD0)
    # Boost D0 to D* rest frame
    beta_Dx = pDx[2:4] ./ pDx[1]

    function boost(p, beta)
        b2 = sum(beta .^ 2)
        gamma = 1.0 / sqrt(1.0 - b2)
        bp = sum(p[2:4] .* beta)
        gamma2 = (gamma - 1.0) / b2
        if b2 < 1e-10
            gamma2 = 0.5
        end
        E_new = gamma * (p[1] - bp)
        p_new = p[2:4] .+ (gamma2 * bp - gamma * p[1]) .* beta
        return [E_new; p_new]
    end

    pD0_DxRF = boost(pD0, beta_Dx)

    # Axes in B frame (Lab)
    z_axis = pDx[2:4]
    z_axis = z_axis / norm(z_axis)

    pD_3 = pD[2:4]
    pK_3 = pK[2:4]
    y_axis = cross(pD_3, pK_3)
    y_axis = y_axis / norm(y_axis)

    x_axis = cross(y_axis, z_axis)

    pD0_3 = pD0_DxRF[2:4]
    px = dot(pD0_3, x_axis)
    py = dot(pD0_3, y_axis)
    pz = dot(pD0_3, z_axis)

    pmag = sqrt(px^2 + py^2 + pz^2)
    return pz / pmag, atan(py, px)
end

cos_theta_new, phi_new = calculate_angles(pD_vec, pK_vec, pDx_vec, pD0_vec)
println("Calculated angles: cosθ = $cos_theta_new, ϕ = $phi_new")

test_point_new = DalitzAndDecay(σs_new, cos_theta_new, phi_new)

# --- Update parameters from JSON (Masses/Widths only) ---
println("\n--- Updating parameters from final_params_full.json ---")

# Load JSON
json_path = joinpath(@__DIR__, "..", "..", "..", "archive", "data", "final_params_full.json")
println("Loading parameters from: ", json_path)
params_json = JSON.parsefile(json_path)
val = params_json["value"]

# Helper to get mass/width
get_mass(name) = val["$(name)_mass"]
get_width(name) = val["$(name)_width"]

# Update resonances with loaded parameters
# Using SafeBreitWigner to handle sub-threshold masses (Energy Dependent)
resonances_updated = [
    (; jp=jp"1+", name="EFF(1++)", lineshape=SafeBreitWigner(get_mass("X(3872)"), get_width("X(3872)"), mD, mDx, 0, 3.0)),
    (; jp=jp"0-", name="ηc(3945)", lineshape=SafeBreitWigner(get_mass("X(3915)(0-)"), get_width("X(3915)(0-)"), mD, mDx, 1, 3.0)),
    (; jp=jp"2+", name="χc2(3930)", lineshape=SafeBreitWigner(get_mass("chi(c2)(3930)"), get_width("chi(c2)(3930)"), mD, mDx, 2, 3.0)),
    (; jp=jp"1+", name="hc(4000)", lineshape=SafeBreitWigner(get_mass("X(3940)(1.)"), get_width("X(3940)(1.)"), mD, mDx, 0, 3.0)),
    (; jp=jp"1+", name="χc1(4010)", lineshape=SafeBreitWigner(get_mass("X(3993)"), get_width("X(3993)"), mD, mDx, 0, 3.0)),
    (; jp=jp"1-", name="ψ(4040)", lineshape=SafeBreitWigner(get_mass("Psi(4040)"), get_width("Psi(4040)"), mD, mDx, 1, 3.0)),
    (; jp=jp"1+", name="hc(4300)", lineshape=SafeBreitWigner(get_mass("X(4300)"), get_width("X(4300)"), mD, mDx, 0, 3.0)),
    (; jp=jp"0+", name="Tcs0(2870)", lineshape=SafeBreitWigner(get_mass("X0(2900)"), get_width("X0(2900)"), mD, mK, 0, 3.0)),
    (; jp=jp"1-", name="Tcs1(2900)", lineshape=SafeBreitWigner(get_mass("X1(2900)"), get_width("X1(2900)"), mD, mK, 1, 3.0)),
    (; jp=jp"1-", name="NR(1--)", lineshape=ConstantLineshape),
    (; jp=jp"0-", name="NR(0--)", lineshape=ConstantLineshape),
    (; jp=jp"1+", name="NR(1++)", lineshape=ConstantLineshape),
    (; jp=jp"0-", name="NR(0-+)", lineshape=NRexp(αβ=val["NR(0-)SPp_alpha"] + val["NR(0-)SPp_beta"] * 1im, m0=4.35))
]; # Removed DataFrame conversion

# Define decay chains structure
decay_chains = [
    (k=2, resonance_name="EFF(1++)", l=0),
    (k=2, resonance_name="EFF(1++)", l=2),
    (k=2, resonance_name="ηc(3945)"),
    (k=2, resonance_name="χc2(3930)"),
    (k=2, resonance_name="hc(4000)", l=0),
    (k=2, resonance_name="hc(4000)", l=2),
    (k=2, resonance_name="χc1(4010)", l=0),
    (k=2, resonance_name="χc1(4010)", l=2),
    (k=2, resonance_name="ψ(4040)"),
    (k=2, resonance_name="hc(4300)", l=0),
    (k=2, resonance_name="hc(4300)", l=2),
    (k=2, resonance_name="NR(1--)"),
    (k=2, resonance_name="NR(0--)"),
    (k=2, resonance_name="NR(1++)", l=0),
    (k=2, resonance_name="NR(0-+)"),
    (k=3, resonance_name="Tcs0(2870)"),
    (k=3, resonance_name="Tcs1(2900)", L=0),
    (k=3, resonance_name="Tcs1(2900)", L=1),
    (k=3, resonance_name="Tcs1(2900)", L=2),
];

# Rebuild chains with updated resonances
chains_updated = let
    # Removed LittleDict
    resonance_dict = Dict(
        r.name => r for r in resonances_updated)

    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k)
        @unpack L, S, l, s = comprete_data
        two_j = _jp.two_j
        d = 3.0
        Xlineshape = resonance_dict[resonance_name].lineshape
        HRk = VertexFunction(RecouplingLS((L, S) .|> x2), BlattWeisskopf{div(x2(L), 2)}(d))
        Hij = VertexFunction(RecouplingLS((l, s) .|> x2), BlattWeisskopf{div(x2(l), 2)}(d))
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

# Define the model with updated chains
const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* string.([ch.Hij.h.two_ls[1] |> d2 for ch in chains_updated])
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains_updated]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains_updated)), chains_updated))
end;

# --- Output Groups ---
# Define resonance groups (summing over LS/Production chains)
# User requested Decay LS splits but no Prod LS splits.
resonance_groups = [
    ("X(3872) [L=1, l=0]", [1]),
    ("X(3872) [L=1, l=2]", [2]),
    ("X(3915)(0-) [L=1, l=1]", [3]),
    ("chi(c2)(3930) [L=1, l=1]", [4]),
    ("X(3940)(1.) [L=0, l=0]", [5]),
    ("X(3940)(1.) [L=0, l=2]", [6]),
    ("X(3993) [L=0, l=0]", [7]),
    ("X(3993) [L=0, l=2]", [8]),
    ("Psi(4040) [L=1, l=1]", [9]),
    ("X(4300) [L=0, l=0]", [10]),
    ("X(4300) [L=0, l=2]", [11]),
    ("NR(1-)PPm [L=1, l=1]", [12]),
    ("NR(0-)SPm [L=0, l=1]", [13]),
    ("NR(1+)PSp [L=1, l=0]", [14]),
    ("NR(0-)SPp [L=0, l=1]", [15]),
    ("X0(2900) [L=0, l=0]", [16]),
    ("X1(2900) [L=0, l=1]", [17]),
    ("X1(2900) [L=1, l=1]", [18]),
    ("X1(2900) [L=2, l=1]", [19])
]

# --- Amplitude Correction Logic ---
# Calculate normalization factor based on physical convention difference
# tf-pwa uses normalized barrier factors: B(q)/B(q0) * (q/q0)^L
# Julia uses unnormalized tensor amplitudes (~ q^L * M^-L)
# The conversion factor is approx (M_parent / q0)^L

function calculate_normalization_factor(chain, res_name, mB, m1, m2, m3)
    # Extract q0 and L for Production
    # Production: B -> Res + Spectator
    # Decay: Res -> Child1 + Child2

    # Identify Res Mass
    mR = 4.0 # Default
    try
        mR = chain.Xlineshape.m0
    catch
        if occursin("NR", res_name)
            mR = 4.35 # Approx NR mass
        end
    end

    # Spectator handling
    if chain.k == 2 # Spec K (2)
        spec_mass = m2
        d_mass1 = m1
        d_mass2 = m3
        q0_prod = abs(breakup_safe(mB, mR, spec_mass))
        q0_dec = abs(breakup_safe(mR, d_mass1, d_mass2))
    elseif chain.k == 3 # Spec Dx (3)
        spec_mass = m3
        d_mass1 = m1
        d_mass2 = m2
        q0_prod = abs(breakup_safe(mB, mR, spec_mass))
        q0_dec = abs(breakup_safe(mR, d_mass1, d_mass2))
    elseif chain.k == 1 # Spec D (1)
        spec_mass = m1
        d_mass1 = m2
        d_mass2 = m3
        q0_prod = abs(breakup_safe(mB, mR, spec_mass))
        q0_dec = abs(breakup_safe(mR, d_mass1, d_mass2))
    else
        return 1.0
    end

    # Ad-Hoc Mass Logic from tf-pwa for sub-threshold resonances
    function ad_hoc_mass(m0, m_max, m_min)
        k = (m_max - m_min) / 2.0
        val = (2 * m0 - (m_max + m_min)) / k / 4.0
        m_eff = k * (1.0 + tanh(val)) + m_min
        return m_eff
    end

    # Determine bounds for ad-hoc mass
    # Parent: B -> R + Spec. m_max = mB - mSpec.
    # Daughter: R -> d1 + d2. m_min = md1 + md2.

    local q0_p_val = q0_prod
    local q0_d_val = q0_dec

    # Check if resonance is likely sub-threshold or near threshold (X(3872))
    # Or strict definition: if mR < m_min or explicit flags.
    # We will apply it for X(3872) explicitly or based on mass.

    is_sub_threshold = (mR < (d_mass1 + d_mass2)) || contains(res_name, "X(3872)")

    if is_sub_threshold
        # Decay side q0 adjustment
        # m_max: Available energy from production step? 
        # For q0 decay normalization, tf-pwa uses the resonant mass m0.
        # But calculates m_eff based on phase space limits.

        # Max mass available for R is mB - spec_mass
        m_max_R = mB - spec_mass
        m_min_R = d_mass1 + d_mass2

        m_eff = ad_hoc_mass(mR, m_max_R, m_min_R)

        # Re-calculate q0_dec using m_eff
        q0_d_val = breakup_safe(m_eff, d_mass1, d_mass2)
    end

    L = chain.HRk.h.two_ls[1] |> d2
    l = chain.Hij.h.two_ls[1] |> d2

    # Factor logic: To match tf-pwa normalized amplitude
    # tf-pwa: A ~ g * (q^L B(q)) / (q0^L B(q0))
    # Julia:  A ~ g * q^L
    # So we must divide Julia by (q0^L B(q0)).

    L_num = (L isa String) ? parse(Int, L) : L
    l_num = (l isa String) ? parse(Int, l) : l

    d = 3.0 # default radius

    # Blatt-Weisskopf Factor B(q0)
    function bw_factor(q0, L, d)
        z0 = (q0 * d)^2
        if L == 0
            return 1.0
        elseif L == 1
            return sqrt(1.0 / (1.0 + z0))
        elseif L == 2
            return sqrt(1.0 / (9.0 + 3 * z0 + z0^2))
        else
            return 1.0
        end
    end

    # Production
    # Usually production q0 is fine (mB >> mR + mK)
    f_p = (q0_p_val^L_num) * bw_factor(q0_p_val, L_num, d)

    # Decay
    f_d = (q0_d_val^l_num) * bw_factor(q0_d_val, l_num, d)

    return f_p * f_d
end

# Define the model with updated chains
const model_names = getproperty.(decay_chains, :resonance_name) .*
                    "_l" .* string.([ch.Hij.h.two_ls[1] |> d2 for ch in chains_updated])
model_names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains_updated]

output_file = joinpath(@__DIR__, "..", "..", "..", "archive", "investigation", "Analysis", "amplitudes.txt")
println("Writing results to: ", output_file)

open(output_file, "a") do io
    println(io, "\n--- Julia pure_model Results ---")
    println("\n--- Julia pure_model Results ---")

    sum_corrected = 0.0 + 0.0im

    for (name, indices) in resonance_groups
        # Set couplings: 1.0 for indices in group, 0.0 otherwise
        couplings_list = zeros(ComplexF64, length(decay_chains))

        for idx in indices
            try
                # Calculate factor for this specific chain
                chain_obj = chains_updated[idx]
                res_name = decay_chains[idx].resonance_name
                factor = calculate_normalization_factor(chain_obj, res_name, mB, mD, mK, mDx)
                couplings_list[idx] = (1.0 + 0.0im) / factor
            catch e
                println("Error processing chain idx $idx (Res: $name): $e")
                println("Chain: ", chains_updated[idx])
                rethrow(e)
            end
        end
        new_couplings_updated = SVector{19}(couplings_list)

        # Create updated model (using global model_names and chains_updated)
        model_updated = ThreeBodyDecay(
            model_names .=> zip(new_couplings_updated, chains_updated)
        )

        # Calculate amplitude
        val = amplitude(model_updated, test_point_new)
        # Format output
        r = real(val)
        i = imag(val)
        sign = i >= 0 ? "+" : "-"

        # Match original format "Resonance (Name): ..."
        out_str = "Resonance ($name): $r $sign $(abs(i))im"

        println(out_str)
        println(io, out_str)
    end
end