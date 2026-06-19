# Notebook computes helicity angles for three different configurations
#  Dx D K,
#  D K Dx, and
#  K D Dx
# the phi angle of D in Dx->Dpi decays does not come the same in three configuration.
# it's understood why

using Markdown
using InteractiveUtils

begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec("ThreeBodyDecays")
        Pkg.PackageSpec("Plots")
        Pkg.PackageSpec(url="https://github.com/mmikhasenko/InstructionalDecayTrees.jl.git")
        Pkg.PackageSpec(url="https://github.com/mmikhasenko/FourVectors.jl.git")
        Pkg.PackageSpec("PartialWaveFunctions")
        Pkg.PackageSpec("HadronicLineshapes")
        Pkg.PackageSpec("JSON")
    ])
    # 
    using JSON
    using Plots
    using FourVectors
    using ThreeBodyDecays
    using HadronicLineshapes
    using PartialWaveFunctions
    using InstructionalDecayTrees
end

FourVectors.FourVector(p::JSON.Object) = FourVector(p["px"], p["py"], p["pz"]; E=p["E"])


objs = let
    json_path = joinpath(@__DIR__, "..", "..", "..", "data", "crosscheck_event.json")
    event = JSON.parsefile(json_path)
    four_vectors_json = event["four_vectors"]
    # 
    pD = FourVector(four_vectors_json["D"])
    pD0 = FourVector(four_vectors_json["D0"])
    pK = FourVector(four_vectors_json["K"])
    pπ = FourVector(four_vectors_json["pi"])
    # 
    (pD0, pπ, pD, pK)
end;


comment_minus_K = "the one that gives correct result -- Dx is in xz plane"
program_minus_K = (
    ToHelicityFrame((1, 2, 3, 4)),
    PlaneAlign((-4), (1, 2)),
    MeasureSpherical(:theta_ξ, :phi_ξ, (1, 2)),
    # 
    ToHelicityFrame((1, 2)),
    MeasureSpherical(:theta_D, :phi_D, 1),
);

comment_minus_D = "opposite rotation -- plane flip"
program_minus_D = (
    ToHelicityFrame((1, 2, 3, 4)),
    PlaneAlign((-3), (4,)),
    MeasureSpherical(:theta_ξ, :phi_ξ, (4,)),
    # 
    ToHelicityFrame((1, 2)),
    MeasureSpherical(:theta_D, :phi_D, 1),
);

comment_minus_Dx = "yx plan is not well defined in Dx helicty frame. z <---o Dx "
program_minus_Dx = (
    ToHelicityFrame((1, 2, 3, 4)),
    PlaneAlign((-1, -2), (3,)),
    MeasureSpherical(:theta_ξ, :phi_ξ, (3,)),
    # 
    ToHelicityFrame((1, 2)),
    MeasureSpherical(:theta_D, :phi_D, 1),
);


_, results_minus_K = apply_decay_instruction(program_minus_K, objs);
_, results_minus_D = apply_decay_instruction(program_minus_D, objs);
_, results_minus_Dx = apply_decay_instruction(program_minus_Dx, objs);

println("COMMENT: ", comment_minus_K)
@show results_minus_K

println("COMMENT: ", comment_minus_D)
@show results_minus_D

println("COMMENT: ", comment_minus_Dx)
@show results_minus_Dx
