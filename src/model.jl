function enrich_resonance_chains_df!(df)
    df.coupling_value = resolve_coupling_keys.(df.coupling_keys)
    df.matching_factor = chain_matching_factor.(eachrow(df))
    info_enrich_resonance_chains_df!(df)
    return df
end

function build_dxd_chain(lineshape, two_j, root_two_ls, decay_two_ls, root_l, decay_l)
    return DecayChain(
        dxd_topology;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => Propagator(two_j, lineshape),
        ),
        vertices=(
            (((1, 2), 3), 4) => Vertex(RecouplingLS(root_two_ls), BlattWeisskopf{root_l}(WELL_SIZE)),
            ((1, 2), 3) => Vertex(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function build_dk_chain(
    lineshape,
    two_j,
    root_two_ls,
    dk_two_ls,
    root_l,
    dk_l;
    remove_root_particle2_phase=false,
)
    root_recoupling = remove_root_particle2_phase ? BuggyParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    return DecayChain(
        dk_topology;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => Propagator(two_j, lineshape),
        ),
        vertices=(
            ((1, 2), (3, 4)) => Vertex(root_recoupling, BlattWeisskopf{root_l}(WELL_SIZE)),
            (3, 4) => Vertex(RecouplingLS(dk_two_ls), BlattWeisskopf{dk_l}(WELL_SIZE)),
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
    root_l = vertex_l(row.root_two_ls)
    daughter_l = vertex_l(row.daughter_two_ls)
    if row.topology == :dk
        return build_dk_chain(
            lineshape,
            row.propagator_two_j,
            row.root_two_ls,
            row.daughter_two_ls,
            root_l,
            daughter_l;
            remove_root_particle2_phase=row.root_remove_particle2_phase,
        )
    end
    return build_dxd_chain(
        lineshape,
        row.propagator_two_j,
        row.root_two_ls,
        row.daughter_two_ls,
        root_l,
        daughter_l,
    )
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
