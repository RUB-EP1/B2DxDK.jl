function production_coupling_key(name::String)
    name == "X(3940)(1.)" &&
        return "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0"
    return "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
end

function build_resonance_chains_df()
    rows = NamedTuple[]
    total_x3872 = ("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0",)
    push!(rows, (
        resonance_name="X(3872)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        coupling_keys=total_x3872, lineshape=:adhoc_q0_bwr_ls_l0,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="X(3872)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
        coupling_keys=(total_x3872..., "X(3872)->Dst.D_g_ls_1"),
        lineshape=:adhoc_q0_bwr_ls_l2,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="X(3915)(0-)", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="chi(c2)(3930)", topology=:DxD,
        propagator_two_j=4, root_two_ls=(4, 4), daughter_two_ls=(4, 2),
        coupling_keys=("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l2,
        root_recoupling=:standard,
    ))
    for name in ("X(3940)(1.)", "X(3993)", "X(4300)")
        total = (production_coupling_key(name),)
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
            coupling_keys=total, lineshape=:bwr_ls_l0,
            root_recoupling=:standard,
        ))
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
            coupling_keys=(total..., "$(name)->Dst.D_g_ls_1"),
            lineshape=:bwr_ls_l2,
            root_recoupling=:standard,
        ))
    end
    push!(rows, (
        resonance_name="Psi(4040)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPp", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0",),
        lineshape=:nr_exp,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="NR(1.)PSp", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        coupling_keys=("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPm", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="NR(1-)PPm", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="X0(2900)", topology=:dk,
        propagator_two_j=0, root_two_ls=(2, 2), daughter_two_ls=(0, 0),
        coupling_keys=("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0",),
        lineshape=:x2900_bwr_l0,
        root_recoupling=:standard,
    ))
    # Issue A (TF-PWA omits the Jacob–Wick particle-2 phase): set
    # root_recoupling=:missing_particle2 when the dk root LS coupling is nonzero
    # at λ₂=0 (X1 L=0 and L=2). See docs/note-phase-two.md.
    total_x1 = ("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0",)
    daughter_x1 = (2, 0)
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(0, 0), daughter_two_ls=daughter_x1,
        coupling_keys=total_x1, lineshape=:x2900_bwr_l1,
        root_recoupling=:missing_particle2,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_1"),
        lineshape=:x2900_bwr_l1,
        root_recoupling=:standard,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(4, 4), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_2"),
        lineshape=:x2900_bwr_l1,
        root_recoupling=:missing_particle2,
    ))
    return DataFrame(rows)
end

const resonance_chains_df_raw = build_resonance_chains_df()
