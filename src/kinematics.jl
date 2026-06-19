const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const dxd_topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))
const kinematic_task = KinematicTask((dxd_topology, dk_topology))
const standard_system = CascadeSystem(
    external_spins,
    SystemMasses(
        nominal_mass["D0"],
        nominal_mass["pi"],
        nominal_mass["D"],
        nominal_mass["K"];
        m0=nominal_mass["Bp"],
    ),
)

function event_point(row)
    pDminus = FourVector(row.Dm_px, row.Dm_py, row.Dm_pz; E=row.Dm_E)
    pD0 = FourVector(row.D0_px, row.D0_py, row.D0_pz; E=row.D0_E)
    pKplus = FourVector(row.Kp_px, row.Kp_py, row.Kp_pz; E=row.Kp_E)
    piplus = FourVector(row.pip_px, row.pip_py, row.pip_pz; E=row.pip_E)
    return KinematicPoint(kinematic_task, (pD0, piplus, pDminus, pKplus))
end
