module B2DxDK

using CascadeDecays
import CascadeDecays.ThreeBodyDecays
using CascadeDecays.ThreeBodyDecays: Recoupling, RecouplingLS
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON
using StaticArrays

include("recoupling.jl")

export data_dir,
    repo_root,
    all_resonance_names
include("parameters.jl")

include("lineshapes.jl")
include("resonance_table.jl")
include("matching.jl")

export event_point
include("kinematics.jl")

export build_all_resonance_cascade,
    build_resonance_cascade,
    resonance_chain_names,
    resonance_chains_df
include("model.jl")

end # module B2DxDK
