using OrderedCollections
using LinearAlgebra
using DataFrames
using JSON
using CSV


t = CSV.read(joinpath(@__DIR__, "fit_frac1_pw.csv"), DataFrame)
# 
nWaves = size(t, 1)

mn = let
    m = t[:, 2:end] |> Matrix
    [m[i, 1:i] for i in 1:nWaves]
end

# 
δt = CSV.read(joinpath(@__DIR__, "fit_frac1_pw_err.csv"), DataFrame)
# 
δmn = let
    m = δt[:, 2:end] |> Matrix
    [m[i, 1:i] for i in 1:nWaves]
end


name_replacements = [
    "X(3872)/0" => "EFF(1++)_l0",
    "X(3872)/1" => "EFF(1++)_l2",
    "X(3915)(0-)/0" => "ηc(3945)_l1",
    "chi(c2)(3930)/0" => "χc2(3930)_l2",
    "X(3940)(1+)/0" => "hc(4000)_l0",
    "X(3940)(1+)/1" => "hc(4000)_l2",
    "X(3993)/0" => "χc1(4010)_l0",
    "X(3993)/1" => "χc1(4010)_l2",
    "Psi(4040)/0" => "ψ(4040)_l1",
    "X(4300)/0" => "hc(4300)_l0",
    "X(4300)/1" => "hc(4300)_l2",
    "NR(0-)SPp/0" => "NR(0-+)_l1",
    "NR(1+)PSp/0" => "NR(1++)_l0",
    "NR(0-)SPm/0" => "NR(0--)_l1",
    "NR(1-)PPm/0" => "NR(1--)_l1",
    "X0(2900)/0" => "Tcs0(2870)_l0_L1",
    "X1(2900)/0" => "Tcs1(2900)_l1_L0",
    "X1(2900)/1" => "Tcs1(2900)_l1_L1",
    "X1(2900)/2" => "Tcs1(2900)_l1_L2",
]

d = LittleDict(
    "waves" => replace(t[:, 1], name_replacements...),
    "matrix" => mn,
    "uncertainty" => δmn)

open(joinpath(@__DIR__, "interference_tf.json"), "w") do io
    JSON.print(io, d)
end
