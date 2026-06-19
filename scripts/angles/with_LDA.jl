import Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using FourVectors
using LazyDecayAngles
using JSON
using DataFrames

# -------------------------------------------------------------------------
# Read four-vectors and kinematics from JSON file
# -------------------------------------------------------------------------
json_path = joinpath(@__DIR__, "..", "..", "data", "crosscheck_event.json")
event = JSON.parsefile(json_path)

fv   = event["four_vectors"]
dpd  = event["dpd_kinematics"]
tfpw = event["tf_pwa_kinematics"]

# -------------------------------------------------------------------------
# LazyDecayAngles program for angles in (D0, π) system
# Indices: 1 = D, 2 = K, 3 = π, 4 = D0
# -------------------------------------------------------------------------
pD  = FourVector(fv["D"]["px"],  fv["D"]["py"],  fv["D"]["pz"];  E = fv["D"]["E"])   # 1
pK  = FourVector(fv["K"]["px"],  fv["K"]["py"],  fv["K"]["pz"];  E = fv["K"]["E"])   # 2
ppi = FourVector(fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]; E = fv["pi"]["E"])  # 3
pD0 = FourVector(fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]; E = fv["D0"]["E"])  # 4

objs = (pD, pK, ppi, pD0)

# -------------------------------------------------------------------------
# Program reproducing tf_pwa-style angles (D0 in (D0, π) from B frame)
# -------------------------------------------------------------------------
program_D0pi = (
    # 1. Go to the rest frame of all (B frame)
    ToHelicityFrame((1, 2, 3, 4)),

    # 2. Measure (D0, π) system
    MeasureMassCosThetaPhi(:D0pi_vars, (4, 3)),

    # 3. Go to (D0, π) frame
    ToHelicityFrame((4, 3)),

    # 4. Measure D0 angles in (D0, π) frame
    MeasureCosThetaPhi(:vars_D0, 4),
)

(_, res_D0pi) = execute_decay_program(objs, program_D0pi)

# -------------------------------------------------------------------------
# Program reproducing DPD-style construction (masses + angles)
# -------------------------------------------------------------------------
program_dpd = (
    # 1. go to the rest frame of all
    ToHelicityFrame((1, 2, 3, 4)),

    # Invariants for mass-squared checks
    MeasureInvariant(:msq_KDx, (2, 3, 4)),
    MeasureInvariant(:msq_DxD, (1, 3, 4)),
    MeasureInvariant(:msq_DK,  (1, 2)),

    # 2. measure angles of 4,3,1 in Total Rest Frame
    MeasureMassCosThetaPhi(:vars_431, (4, 3, 1)),

    # 3. go to (4,3,1)
    ToHelicityFrame((4, 3, 1)),

    # 4. measure angles of 4,3 in (4,3,1) frame
    MeasureMassCosThetaPhi(:vars_431, (4, 3)),

    # 5. go to (4,3)
    ToHelicityFrame((4, 3)),

    # 6. measure angles of 4 in (4,3) frame
    MeasureCosThetaPhi(:vars_Dx_helicity, 4),
)

(_, res_dpd) = execute_decay_program(objs, program_dpd)



const ϵ = 1e-10

# --- 1) Mass-squared: json vs julia (DPD program) -------------------------
df_m = let
    names = ["msq_KDx", "msq_DxD", "msq_DK", "cos_theta_D_in_Dx", "phi_D_in_Dx"]
    DataFrame(
        variable = names,
        json     = [dpd[n] for n in names],
        julia    = [res_dpd.msq_KDx, res_dpd.msq_DxD, res_dpd.msq_DK, res_dpd.vars_Dx_helicity.cosθ, res_dpd.vars_Dx_helicity.ϕ],
    )
end

transform!(df_m, [:json, :julia] => ByRow() do x, y
    abs(x - y) < ϵ
end => :check)

println("\nCross check of DPD variables in the json file:")
println(df_m)

# --- 2) Angles: json vs tf_pwa vs julia -----------------------------------

df_a = let
    variable = ["cos_theta_D_in_Dx_from_B", "phi_D_in_Dx_from_B"]
    DataFrame(;
        variable,
        json_tf_pwa = [tfpw[n] for n in variable],
        julia    = [res_D0pi.vars_D0.cosθ, res_D0pi.vars_D0.ϕ],
    )
end

transform!(df_a, [:json_tf_pwa, :julia] => ByRow() do x, y
    abs(x - y) < ϵ
end => :check)

println("\nCross check of tf_pwa variables in the json file:")
println(df_a)
