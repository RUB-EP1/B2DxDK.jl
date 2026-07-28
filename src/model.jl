function enrich_resonance_chains_df!(df)
    df.coupling_value = resolve_coupling_keys.(df.coupling_keys)
    df.matching_factor = chain_matching_factor.(eachrow(df))
    return df
end

function build_dxd_chain(lineshape, two_j, production::Recoupling, decay::Recoupling)
    return DecayChain(
        dxd_topology;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => Propagator(two_j, lineshape),
        ),
        vertices=(
            (((1, 2), 3), 4) => Vertex(production, BlattWeisskopf{vertex_blatt_l(production)}(WELL_SIZE)),
            ((1, 2), 3) => Vertex(decay, BlattWeisskopf{vertex_blatt_l(decay)}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function build_dk_chain(lineshape, two_j, root::Recoupling, daughter::Recoupling)
    return DecayChain(
        dk_topology;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => Propagator(two_j, lineshape),
        ),
        vertices=(
            ((1, 2), (3, 4)) => Vertex(root, BlattWeisskopf{vertex_blatt_l(root)}(WELL_SIZE)),
            (3, 4) => Vertex(daughter, BlattWeisskopf{vertex_blatt_l(daughter)}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

const resonance_chains_df = enrich_resonance_chains_df!(copy(resonance_chains_df_raw))

function resonance_chain_rows(name::String)
    resonance_chains_df[resonance_chains_df.resonance_name.==name, :]
end

function build_chain_from_row(row)
    lineshape = build_chain_lineshape(row)
    root = root_recoupling(row.root_two_ls, row.root_recoupling)
    daughter = RecouplingLS(row.daughter_two_ls)
    if row.topology == :dk
        return build_dk_chain(lineshape, row.propagator_two_j, root, daughter)
    end
    return build_dxd_chain(lineshape, row.propagator_two_j, root, daughter)
end

function chain_name(resonance_name::String, row)
    "$(resonance_name)_L$(vertex_l(row.root_two_ls))_d$(vertex_l(row.daughter_two_ls))"
end

function resonance_chain_names(resonance_name::String)
    [chain_name(resonance_name, row) for row in eachrow(resonance_chain_rows(resonance_name))]
end

function build_resonance_cascade(resonance_name::String)
    rows = collect(eachrow(resonance_chain_rows(resonance_name)))
    chains = Tuple(build_chain_from_row(row) for row in rows)
    effective_couplings = Tuple(
        rows[i].coupling_value * rows[i].matching_factor
        for i in eachindex(rows)
    )
    names = Tuple(chain_name(resonance_name, row) for row in rows)
    return CascadeDecay(
        chains,
        standard_system,
        dxd_topology;
        couplings=effective_couplings,
        names=names,
    )
end

function build_all_resonance_cascade(names=all_resonance_names)
    merge([build_resonance_cascade(name) for name in names]...)
end
