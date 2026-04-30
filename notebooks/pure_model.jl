
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
# 
using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using OrderedCollections
using ThreeBodyDecays
using LinearAlgebra
using Measurements
using Statistics
using Parameters
using DataFrames.PrettyTables
using DataFrames
using Setfield
using QuadGK
using Plots
using Optim
using JSON
using Cuba
# 
using Random
Random.seed!(1234)





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
    ms = ThreeBodyMasses(mD, mK, mDx; m0 = mB)
    ThreeBodySystem(ms, two_js)
end;


# Various lineshapes

function BreitWignerSDwaves(; m, Γ, γS, d)
    fr = [γS, 1 - γS]
    channels = map(zip([0, 2], fr)) do (l, x)
        ma, mb = mD, mDx
        # 
        p = HadronicLineshapes.breakup(m, ma, mb)
        gsq = m * Γ / (2 * p / m) * x / BlattWeisskopf{l}(d)(p)^2
        # 
        (; gsq, ma, mb, l, d)
    end
    MultichannelBreitWigner(; m, channels)
end

const EFF = BreitWigner(3.87165, 0.00119);


begin
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))
    # 
    const ConstantLineshape = WrapFlexFunction(x -> 1.0)
end


# Decay chains

resonances =
    [
        (; jp = jp"1+", name = "EFF(1++)", lineshape = EFF),
        # 
        (; jp = jp"0-", name = "ηc(3945)", lineshape = BreitWigner(3.936, 0.096)),
        (; jp = jp"2+", name = "χc2(3930)", lineshape = BreitWigner(3.9272, 0.024)),
        # (; jp = jp"1+", name = "hc(4000)", lineshape = BreitWignerSDwaves(m = 3.987, Γ = 0.368, γS = cos(0.8)^2, d = 3.0)),
        (; jp = jp"1+", name = "hc(4000)", lineshape = BreitWigner(m = 3.987, Γ = 0.368)),
        # (; jp = jp"1+", name = "χc1(4010)", lineshape = BreitWignerSDwaves(m = 4.006, Γ = 0.067, γS = cos(0.47)^2, d = 3.0)),
        (; jp = jp"1+", name = "χc1(4010)", lineshape = BreitWigner(m = 4.006, Γ = 0.067)),
        (; jp = jp"1-", name = "ψ(4040)", lineshape = BreitWigner(4.039, 0.080)),
        # (; jp = jp"1+", name = "hc(4300)", lineshape = BreitWignerSDwaves(m = 4.306, Γ = 0.0372, γS = cos(1.02)^2, d = 3.0)),
        (; jp = jp"1+", name = "hc(4300)", lineshape = BreitWigner(m = 4.306, Γ = 0.0372)),
        # Tcbarsbar
        (; jp = jp"0+", name = "Tcs0(2870)", lineshape = BreitWigner(m = 2.866, Γ = 0.057, ma = mD, mb = mK, l = 0, d = 3.0)),
        (; jp = jp"1-", name = "Tcs1(2900)", lineshape = BreitWigner(m = 2.904, Γ = 0.110, ma = mD, mb = mK, l = 1, d = 3.0)),
        # NR
        (; jp = jp"1-", name = "NR(1--)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0--)", lineshape = ConstantLineshape),
        (; jp = jp"1+", name = "NR(1++)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0-+)", lineshape = NRexp(αβ = -0.11261265753519498 + 0.3362692520521029im, m0 = 4.35))
    ] |> DataFrame

decay_chains = [
    (k = 2, resonance_name = "EFF(1++)", l = 0),
    (k = 2, resonance_name = "EFF(1++)", l = 2),
    # 
    (k = 2, resonance_name = "ηc(3945)"),
    (k = 2, resonance_name = "χc2(3930)"),
    (k = 2, resonance_name = "hc(4000)", l = 0),
    (k = 2, resonance_name = "hc(4000)", l = 2),
    (k = 2, resonance_name = "χc1(4010)", l = 0),
    (k = 2, resonance_name = "χc1(4010)", l = 2),
    (k = 2, resonance_name = "ψ(4040)"),
    (k = 2, resonance_name = "hc(4300)", l = 0),
    (k = 2, resonance_name = "hc(4300)", l = 2),
    # 
    (k = 2, resonance_name = "NR(1--)"),
    (k = 2, resonance_name = "NR(0--)"),
    (k = 2, resonance_name = "NR(1++)", l = 0),
    (k = 2, resonance_name = "NR(0-+)"),
    # 
    (k = 3, resonance_name = "Tcs0(2870)"),
    (k = 3, resonance_name = "Tcs1(2900)", L = 0),
    (k = 3, resonance_name = "Tcs1(2900)", L = 1),
    (k = 3, resonance_name = "Tcs1(2900)", L = 2),
];

chains = let
    resonance_dict = LittleDict(
        resonances.name .=> NamedTuple.(eachrow(resonances)))
    #
    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k) # takes into account l and L from named tuple
        @unpack L, S, l, s = comprete_data
        # 
        two_j = _jp.two_j
        # 
        d = 3.0
        Xlineshape = resonance_dict[resonance_name].lineshape
        # 
        HRk = VertexFunction(RecouplingLS((L, S) .|> x2), BlattWeisskopf{div(x2(L), 2)}(d))
        Hij = VertexFunction(RecouplingLS((l, s) .|> x2), BlattWeisskopf{div(x2(l), 2)}(d))
        # 
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* [ch.Hij.h.two_ls[1] |> d2 for ch in chains]
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains)), chains))
end;

using ThreeBodyDecays.StaticArrays
new_couplings = SVector{19}([
    
])
new_couplings[1] = 1.0 + 0.0im
new_model = @set model_pure.couplings = new_couplings


println(new_model.couplings)


# compute full amplitude

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



# testing



Random.seed!(1234)
σs0 = randomPoint(masses(model_pure))
test_point = DalitzAndDecay(σs0, 0.3, 0.2)
amplitude(model_pure, test_point)

# number of chains
length(model_pure.names)

println("## Names of the chains")
for (i, name) in enumerate(model_pure.names)
    println("$i. $name")
end


# calling amplitude on one chain only

model_with_one_chain = model_pure[3]
amplitude(model_with_one_chain, test_point)






println("test_point =")
dump(test_point)

# compute 4-vectors for the three final particles in the B rest frame
function lambda(a, b, c)
    max(0.0, a^2 + b^2 + c^2 - 2*(a*b + b*c + c*a))
end

function fourmomenta_from_mandelstam(dd::DalitzAndDecay)
    σs = dd.σs
    # extract the first three numeric fields from the MandelstamTuple (robust to field names)
    fn = fieldnames(typeof(σs))
    vals = getfield.(Ref(σs), fn)
    nums = [Float64(v) for v in vals if isa(v, Real)]
    @assert length(nums) >= 3 "Could not extract three Mandelstam invariants from σs"
    s12, s13, s23 = nums[1:3]

    # final-state masses (order matches masses(model_pure))
    ms = masses(model_pure)
    m1, m2, m3 = Float64(ms[1]), Float64(ms[2]), Float64(ms[3])
    m0 = Float64(mB)

    # kinematics: put particle 3 along +z in B rest frame
    m12 = sqrt(max(0.0, s12))
    E3 = (m0^2 - s12 + m3^2) / (2m0)
    p3 = sqrt(max(0.0, E3^2 - m3^2))

    # 4-vector of particle 3 in B rest frame
    p3_4 = (E3, 0.0, 0.0, p3)

    # in the 12-rest frame compute particle 1/2 energies and momentum magnitude
    pstar = sqrt(lambda(s12, m1^2, m2^2)) / (2m12)
    E1_star = (s12 + m1^2 - m2^2) / (2m12)
    E2_star = (s12 + m2^2 - m1^2) / (2m12)

    # orientation of particle 1 in 12 rest frame from dd.cosθ and dd.ϕ
    cosθ = Float64(dd.cosθ); ϕ = Float64(dd.ϕ)
    sinθ = sqrt(max(0.0, 1.0 - cosθ^2))
    px1 = pstar * sinθ * cos(ϕ)
    py1 = pstar * sinθ * sin(ϕ)
    pz1 = pstar * cosθ
    p1_star = (E1_star, px1, py1, pz1)
    p2_star = (E2_star, -px1, -py1, -pz1)

    # boost from 12-rest to B-rest: β along z (12-system recoils opposite to particle 3)
    E12 = m0 - E3
    βz = -p3 / E12
    γ = E12 / m12
    function boost_z(v::NTuple{4,Float64})
        E, px, py, pz = v
        E_lab  = γ*(E + βz*pz)
        pz_lab = γ*(pz + βz*E)
        (E_lab, px, py, pz_lab)
    end

    p1_4 = boost_z(p1_star)
    p2_4 = boost_z(p2_star)

    return (p1 = p1_4, p2 = p2_4, p3 = p3_4)
end

# compute and display for test_point
fourvecs = fourmomenta_from_mandelstam(test_point)
println("(E, px, py, pz) in B-system:")
println("p1 = ", fourvecs.p1)
println("p2 = ", fourvecs.p2)
println("p3 = ", fourvecs.p3)

# 4-vector sum check
p1, p2, p3 = fourvecs.p1, fourvecs.p2, fourvecs.p3

p_sum = (p1[1] + p2[1] + p3[1],
         p1[2] + p2[2] + p3[2],
         p1[3] + p2[3] + p3[3],
         p1[4] + p2[4] + p3[4])

println("Sum = ", p_sum)
println("mB = ", mB)






# Berechne Mandelstam-Variablen s12, s13, s23 aus drei Viererimpulsen
# Jeder Viererimpuls kann ein NTuple{4,Real} oder ein indexierbares Objekt mit
# Elementen (E, px, py, pz) sein.
function mandelstam_from_fourvectors(p1, p2, p3)
    # Hilfsfunktion: Minkowski-Quadrat eines Summen-Viererimpulses (c=1, metric = diag(1,-1,-1,-1))
    sqsum(a, b) = begin
        E = float(a[1]) + float(b[1])
        px = float(a[2]) + float(b[2])
        py = float(a[3]) + float(b[3])
        pz = float(a[4]) + float(b[4])
        E^2 - px^2 - py^2 - pz^2
    end

    s12 = sqsum(p1, p2)
    s13 = sqsum(p1, p3)
    s23 = sqsum(p2, p3)

    # Rückgabe als NamedTuple (Feldreihenfolge bleibt erhalten und ist kompatibel mit späterem Zugriff)
    (s12 = s12, s13 = s13, s23 = s23)
end

# Beispiel (sofern p1,p2,p3 im Gültigkeitsbereich definiert sind):
# mandelstam = mandelstam_from_fourvectors(p1, p2, p3)

P1 = (2.0, 0.5, 0.3, 1.8)
P2 = (1.0, -0.5, 0.2, 0.8)
P3 = (0.5, 0.0, -0.1, 0.3)

mandelstam_test = mandelstam_from_fourvectors(P1, P2, P3)

# compute cos(theta) and phi from three four-vectors (in the B-rest frame)
function angles_from_fourvectors(p1, p2, p3)
    E1, p1x, p1y, p1z = Float64.(p1)
    E2, p2x, p2y, p2z = Float64.(p2)
    E3, p3x, p3y, p3z = Float64.(p3)

    # 12-system four-momentum and velocity β12 = P12/E12
    E12 = E1 + E2
    Px12 = p1x + p2x; Py12 = p1y + p2y; Pz12 = p1z + p2z
    βx = Px12 / E12; βy = Py12 / E12; βz = Pz12 / E12

    # boost velocity to go to 12-rest is -βvec
    bx, by, bz = -βx, -βy, -βz
    b2 = bx^2 + by^2 + bz^2
    @assert b2 < 1 "Invalid kinematics: |β| >= 1"
    γ = b2 == 0.0 ? 1.0 : 1.0 / sqrt(1.0 - b2)

    # Lorentz boost by velocity (bx,by,bz)
    function boost(v::NTuple{4,Float64})
        E, px, py, pz = v
        bdotp = bx*px + by*py + bz*pz
        Eprime = γ*(E - bdotp)
        if b2 == 0.0
            return (Eprime, px, py, pz)
        end
        k = (γ - 1.0) * (bdotp) / b2 - γ * E
        pxp = px + k * bx
        pyp = py + k * by
        pzp = pz + k * bz
        (Eprime, pxp, pyp, pzp)
    end

    # p1 in the 12-rest frame
    p1star = boost((E1, p1x, p1y, p1z))
    pxs, pys, pzs = p1star[2], p1star[3], p1star[4]
    pmag = sqrt(max(0.0, pxs^2 + pys^2 + pzs^2))

    cosθ = pmag == 0.0 ? 1.0 : pzs / pmag
    ϕ = atan(pys, pxs)  # atan(y,x) -> atan2(y,x)

    return (cosθ = cosθ, ϕ = ϕ)
end

# compute angles for the example four-vectors and build a DalitzAndDecay
angles = angles_from_fourvectors(P1, P2, P3)
println("cosθ = ", angles.cosθ, ", ϕ = ", angles.ϕ)

println("fourmomenta_from_mandelstam(dd) ->")
println(fourmomenta_from_mandelstam(dd))

fourmomenta_from_mandelstam(mandelstam_test)