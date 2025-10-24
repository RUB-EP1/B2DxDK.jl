
using ThreeBodyDecaysIO


# how to do serialization for custom lineshapes

function ThreeBodyDecaysIO.serializeToDict(obj::NamedArgFunc{<:NRexp})
    type = "NRexp"
    @unpack f, variable_names = obj
    @unpack αβ, m0 = f
    x = first(variable_names)
    appendix = Dict()
    dict = LittleDict{String,Any}(
        "type" => type,
        "αβ_real" => real(αβ),
        "αβ_imag" => imag(αβ),
        "m0" => m0,
    )
    return (dict, appendix)
end

function ThreeBodyDecaysIO.serializeToDict(obj::NamedArgFunc{<:ConstantLineshape})
    type = "ConstantLineshape"
    @unpack f, variable_names = obj
    x = first(variable_names)
    appendix = Dict()
    dict = LittleDict{String,Any}(
        "type" => type,
    )
    return (dict, appendix)
end


function ThreeBodyDecaysIO.dict2instance(::Type{ConstantLineshape}, dict)
    return NamedArgFunc(ConstantLineshape(), ["x"])
end

function ThreeBodyDecaysIO.dict2instance(::Type{NRexp}, dict)
    @unpack αβ_real, αβ_imag, m0 = dict
    return NamedArgFunc(NRexp(αβ_real + αβ_imag * 1im, m0), ["x"])
end

# Once
# model_pure is defined


# let # save to json
#     decay_description, appendix = serializeToDict(model_pure)
#     dict = add_hs3_fields(decay_description, appendix, "default-model")
#     open(joinpath(@__DIR__, "..", "data", "model_pure.json"), "w") do io
#         JSON.print(io, dict, 4)
#     end
# end




json_content = open(joinpath(@__DIR__, "..", "data", "model_pure.json")) do io
    JSON.parse(io)
end
rm("test.json")

@unpack decay_description = json_content["distributions"][1]

let
    @show length(decay_description["chains"]) == 3
    @unpack chains = decay_description
    map(chains) do chain
        length(chain["vertices"]) != 2 && error("Vertices length is not 2 but $(length(chain["vertices"]))")
        length(chain["propagators"]) != 1 && error("Propagators length is not 1")
    end
end


input = copy(json_content)

let
    @unpack decay_description = input["distributions"][1]
    @unpack functions = input
    workspace = Dict{String,Any}()
    for fn in functions
        @unpack name, type = fn
        instance_type = eval(Symbol(type))
        workspace[name] = dict2instance(instance_type, fn)
    end
    dict2instance(ThreeBodyDecay, decay_description; workspace) isa ThreeBodyDecay
end
