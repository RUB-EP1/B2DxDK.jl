import Pkg
Pkg.activate(joinpath(@__DIR__))

using FourVectors
using JSON
using DataFrames

# -------------------------------------------------------------------------
# Read four-vectors and DPD kinematics from JSON file
# -------------------------------------------------------------------------
json_path = joinpath(@__DIR__, "..", "..", "data", "crosscheck_event.json")
event = JSON.parsefile(json_path)

fv  = event["four_vectors"]
dpd = event["dpd_kinematics"]

pD_vec  = FourVector(fv["D"]["px"],  fv["D"]["py"],  fv["D"]["pz"];  E = fv["D"]["E"])   # 1
pK_vec  = FourVector(fv["K"]["px"],  fv["K"]["py"],  fv["K"]["pz"];  E = fv["K"]["E"])   # 2
ppi_vec = FourVector(fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]; E = fv["pi"]["E"]) # 3
pD0_vec = FourVector(fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]; E = fv["D0"]["E"]) # 4

set_of_vectors = [pD_vec, pK_vec, ppi_vec, pD0_vec]

# Helper: invariant mass squared using explicit components
mass2(v) = v.E^2 - spatial_magnitude(v)^2

# -------------------------------------------------------------------------
# 1) Explicit construction via DDx chain, D0 in (D0, π) system
# -------------------------------------------------------------------------
θ_D0_Dx_chain1, cosθ_D0_Dx_chain1 = let
    # 1. Transform to DDx rest frame
    DDx_vec = pD0_vec + ppi_vec + pD_vec
    set_of_vectors_in_DDx_rest_frame = transform_to_cmf.(set_of_vectors, DDx_vec |> Ref)

    @assert spatial_magnitude(set_of_vectors_in_DDx_rest_frame[[1, 3, 4]] |> sum) < 1e-14

    # 2. Transform to Dx rest frame
    Dx_vec = set_of_vectors_in_DDx_rest_frame[[3, 4]] |> sum
    set_of_vectors_in_Dx_rest_frame = transform_to_cmf.(set_of_vectors_in_DDx_rest_frame, Dx_vec |> Ref)

    @assert spatial_magnitude(set_of_vectors_in_Dx_rest_frame[[3, 4]] |> sum) < 1e-15

    # 3. Calculate angles of D0
    D0_vec = set_of_vectors_in_Dx_rest_frame[4]

    polar_angle(D0_vec), cos_theta(D0_vec)
end

## Other decay chain via (DK) + Dx

θ_D0_Dx_chain2, cosθ_D0_Dx_chain2, ϕ_D0_Dx_chain2 = let
    # 1. orient to go to DK rest frame
    DK_vec = set_of_vectors[[1, 2]] |> sum
    set_of_vectors_DK_plane = rotate_to_plane.(set_of_vectors, DK_vec |> Ref, pD_vec |> Ref)
    @assert set_of_vectors_DK_plane[[1, 2]] |> sum |> transverse_momentum < 1e-14
    @assert set_of_vectors_DK_plane[1][2] < 1e-14

    set_of_vectors_DK_plane_z2Dx = set_of_vectors_DK_plane .|> Ry(π |> Float64)
    Dx_vec = set_of_vectors_DK_plane_z2Dx[[3, 4]] |> sum
    Dx_vec |> transverse_momentum # < 1e-14

    # 2. go to Dx rest frame
    γ_Dx = boost_gamma(Dx_vec)
    set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame = set_of_vectors_DK_plane_z2Dx .|> Bz(-γ_Dx)

    set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame[[3, 4]] |> sum |> spatial_magnitude
    D0_vec = set_of_vectors_DK_plane_z2Dx_in_Dx_rest_frame[4]

    θ  = polar_angle(D0_vec)
    cθ = cos_theta(D0_vec)
    ϕ  = atan(D0_vec[2], D0_vec[1])  # azimuthal angle in the Dx helicity frame

    θ, cθ, ϕ
end

# -------------------------------------------------------------------------
# Compare explicit invariants and angles to DPD json values (as in with_LDA)
# -------------------------------------------------------------------------

msq_KDx = mass2(pK_vec + ppi_vec + pD0_vec)
msq_DxD = mass2(pD_vec + ppi_vec + pD0_vec)
msq_DK  = mass2(pD_vec + pK_vec)

const ϵ = 1e-10

df_m = let
    names = ["msq_KDx", "msq_DxD", "msq_DK", "cos_theta_D_in_Dx", "phi_D_in_Dx"]
    DataFrame(
        variable = names,
        json     = [dpd[n] for n in names],
        julia    = [msq_KDx, msq_DxD, msq_DK, cosθ_D0_Dx_chain2, ϕ_D0_Dx_chain2],
    )
end

transform!(df_m, [:json, :julia] => ByRow() do x, y
    abs(x - y) < ϵ
end => :check)

println("\nCross check of DPD variables in the json file (explicit construction):")
println(df_m)

println("Difference comes from the initial state, it's exactly in the B frame, $(sum(set_of_vectors))")
