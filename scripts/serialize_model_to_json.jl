# Write the B2DxDK amplitude model in the amplitude-serialization format:
#   data/b2dxdk_amplitude_model_Cminus.json   (19 chains)
#   data/b2dxdk_amplitude_model_Cplus.json    (15 chains, X(2900) absent)
# CascadeDecaysIO generates three validation points per file; test/runtests.jl checks
# the files.
#
# Run from the repo root:  julia --project=. scripts/serialize_model_to_json.jl
using CascadeDecaysIO

include("model.jl")

# final states 1..4 and the initial state, per TF-PWA charge configuration `c`
const PARTICLE_LABELS = Dict(
    :Cminus => ("D0", "pi+", "D-", "K+", "B+"),        # c = -1: B+ -> D*+ D- K+, D*+ -> D0 pi+
    :Cplus => ("anti-D0", "pi-", "D+", "K+", "B+"),    # c = +1: B+ -> D*- D+ K+, D*- -> anti-D0 pi-
)
const DISTRIBUTION = Dict(
    :Cminus => "Bplus_to_D0_piplus_Dminus_Kplus_Cminus",
    :Cplus => "Bplus_to_antiD0_piminus_Dplus_Kplus_Cplus",
)
const MASSES = SystemMasses(
    nominal_mass["D0"], nominal_mass["pi"], nominal_mass["D"], nominal_mass["K"];
    m0 = nominal_mass["Bp"],
)

# Validation points, nodes of the reference topology (((1, 2), 3), 4): m(D0 pi) = m(D*)
# (TF-PWA fixes the D* mass); the third point at m(D* D) = 3.885 GeV, 5 MeV above the
# D* D threshold, where X(3872) dominates the intensity
const DSTAR_ON_SHELL = Dict((1, 2) => nominal_mass["Dst"])
const NEAR_THRESHOLD = merge(DSTAR_ON_SHELL, Dict(((1, 2), 3) => 3.885))

for charge in (:Cminus, :Cplus)
    cascade = build_all_resonance_cascade(; charge)
    name = DISTRIBUTION[charge]
    events = [validationEvents(cascade, MASSES; n = 2, node_masses = DSTAR_ON_SHELL);
              validationEvents(cascade, MASSES; n = 1, node_masses = NEAR_THRESHOLD)]
    path = writeJson(joinpath(data_dir, "b2dxdk_amplitude_model_$(charge).json"), cascade, MASSES;
        name, particle_labels = PARTICLE_LABELS[charge],
        validation = validationSection(cascade, events; distribution = name))
    println("wrote ", relpath(path, repo_root))
end
