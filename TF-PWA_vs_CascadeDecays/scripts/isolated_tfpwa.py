"""
Conversion-Corrected Isolated TF-PWA Amplitude Evaluator
========================================================
Standalone Python implementation of TF-PWA's complete 13-channel coherent amplitude model
for B+ -> D*+ D- K+ -> (D0 pi+) D- K+.

Features:
- Pure double-precision (Float64) evaluation matching Julia conventions.
- Fast evaluation (~10 seconds for 50,000 events in NumPy).
- Dynamic reconstructed daughter masses matching TF-PWA's off-shell phase space evaluation.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import yaml
from sympy.physics.quantum.cg import CG

EPS = 1e-8

def dot3(a, b):
    return np.sum(a * b, axis=-1)

def norm3(a):
    return np.linalg.norm(a, axis=-1)

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / n

def cross_unit(a, b):
    cro = np.cross(a, b)
    norm = np.linalg.norm(cro, axis=-1, keepdims=True)
    mask = norm < EPS
    bias_other = np.ones_like(norm) + b
    cro = np.where(mask, np.cross(a, bias_other), cro)
    return unit(cro)

def angle_from(v, x_axis, y_axis):
    return np.arctan2(dot3(v, y_axis), dot3(v, x_axis))

def angle_zx_z_getx(z1, x1, z2):
    u_z1 = unit(z1)
    u_z2 = unit(z2)
    u_y1 = cross_unit(z1, x1)
    u_x1 = cross_unit(u_y1, z1)
    u_yr = cross_unit(z1, z2)
    u_xr = cross_unit(u_yr, z1)
    alpha = angle_from(u_xr, u_x1, u_y1)
    beta = angle_from(u_z2, u_z1, u_xr)
    gamma = np.zeros_like(beta)
    u_x2 = cross_unit(u_yr, u_z2)
    return {"alpha": alpha, "beta": beta, "gamma": gamma}, u_x2

def invariant_mass(p4):
    m2 = p4[..., 0] ** 2 - np.sum(p4[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.abs(m2))

def boost_vector(p4):
    return p4[..., 1:] / p4[..., 0:1]

def boost(p4, beta):
    beta2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    bp = np.sum(beta * p4[..., 1:], axis=-1)
    gamma2 = np.where(beta2 > EPS, (gamma - 1.0) / beta2, 0.0)
    spatial = p4[..., 1:]
    spatial = spatial + gamma2[..., None] * bp[..., None] * beta
    spatial = spatial + gamma[..., None] * p4[..., 0:1] * beta
    energy = (gamma * (p4[..., 0] + bp))[..., None]
    return np.concatenate([energy, spatial], axis=-1)

def rest_vector(core_p4, other_p4):
    return boost(other_p4, -boost_vector(core_p4))

def get_relative_p2(m0, m1, m2):
    return ((m0 * m0 - (m1 + m2) ** 2) * (m0 * m0 - (m1 - m2) ** 2)) / (4 * m0 * m0)

def bprime_polynomial(l, z):
    coeff = {
        0: [1.0],
        1: [1.0, 1.0],
        2: [1.0, 3.0, 9.0],
        3: [1.0, 6.0, 45.0, 225.0],
    }
    return np.polyval(coeff[int(l)], z)

def bprime_q2(l, q2, q02, d=3.0):
    z0 = q02 * d**2
    z = q2 * d**2
    ratio = bprime_polynomial(l, z0) / bprime_polynomial(l, z)
    return np.sqrt(np.where(ratio > 0, ratio, 1.0))

def barrier_factor2(l, mass, q2, q02, d=3.0, barrier_factor_norm=True):
    tmp = q2 ** (l / 2) * bprime_q2(l, q2, q02, d)
    if barrier_factor_norm:
        tmp = tmp / np.abs(q02) ** (l / 2)
    return tmp.reshape(-1, 1)

def gamma_running(m, gamma0, q, q0, l, m0, d=3.0):
    qq0 = np.where(q0 > 1e-15, (q / q0) ** (2 * l + 1), 1.0)
    mm0 = m0 / m
    bp = (np.sqrt(bprime_polynomial(l, (q0 * d) ** 2)) / np.sqrt(bprime_polynomial(l, (q * d) ** 2))) ** 2
    return gamma0 * qq0 * mm0 * bp

def bwr(m, m0, gamma0, q, q0, l, d=3.0):
    gamma_m = gamma_running(m, gamma0, q, q0, l, m0, d)
    x = m0 * m0 - m * m
    y = m0 * gamma_m
    denom = x * x + y * y
    return x / denom + 1j * y / denom

def small_d_weight(j2):
    ret = np.zeros((j2 + 1, j2 + 1, j2 + 1))
    def half_factorial(x):
        return math.factorial(x >> 1)
    for m in range(-j2, j2 + 1, 2):
        for n in range(-j2, j2 + 1, 2):
            for k in range(max(0, n - m), min(j2 - m, j2 + n) + 1, 2):
                ell = (2 * k + (m - n)) // 2
                sign = (-1) ** ((k + m - n) // 2)
                val = sign * math.sqrt(
                    half_factorial(j2 + m)
                    * half_factorial(j2 - m)
                    * half_factorial(j2 + n)
                    * half_factorial(j2 - n)
                )
                val /= (
                    half_factorial(j2 - m - k)
                    * half_factorial(j2 + n - k)
                    * half_factorial(k + m - n)
                    * half_factorial(k)
                )
                ret[ell][(m + j2) // 2][(n + j2) // 2] = val
    return ret

_small_d_weights_cache = {}
def get_cached_small_d_weight(j2):
    if j2 not in _small_d_weights_cache:
        _small_d_weights_cache[j2] = small_d_weight(j2)
    return _small_d_weights_cache[j2]

def small_d_matrix(theta, j2):
    theta = np.asarray(theta)
    powers = np.arange(0, j2 + 1).reshape(1, -1)
    half_theta = 0.5 * theta.reshape(-1, 1)
    sc = (np.sin(half_theta) ** powers) * (np.cos(half_theta) ** (j2 - powers))
    weights = get_cached_small_d_weight(j2).reshape(j2 + 1, (j2 + 1) * (j2 + 1))
    return (sc @ weights).reshape(-1, j2 + 1, j2 + 1)

def d_matrix_conj(alpha, beta, gamma, j2):
    m = np.arange(-j2 / 2, j2 / 2 + 1, 1).reshape(1, -1)
    d_small = small_d_matrix(beta, j2)
    exp_alpha = np.exp(1j * alpha.reshape(-1, 1) * m).reshape(-1, j2 + 1, 1)
    exp_gamma = np.exp(1j * gamma.reshape(-1, 1) * m).reshape(-1, 1, j2 + 1)
    return exp_alpha * exp_gamma * d_small.astype(complex)

def dfun_delta_v2(d, ja, la, lb, lc=(0,)):
    ln = int(2 * ja + 1 + 0.1)
    idx = []
    max_idx = ln * ln
    for la_i in la:
        for lb_i in lb:
            for lc_i in lc:
                delta = lb_i - lc_i
                if abs(delta) <= ja:
                    idx.append(int((la_i + ja) * ln + delta + ja + 0.1))
                else:
                    idx.append(max_idx)
    flat = d.reshape(-1, ln * ln)
    padded = np.pad(flat, ((0, 0), (0, 1)))
    return padded[:, idx].reshape(-1, len(la), len(lb), len(lc))

def get_d_matrix_lambda(angle, ja, la, lb, lc=None):
    d = d_matrix_conj(angle["alpha"], angle["beta"], angle.get("gamma", np.zeros_like(angle["beta"])), int(2 * ja + 0.1))
    if lc is None:
        return dfun_delta_v2(d, ja, la, lb, (0,)).reshape(-1, len(la), len(lb))
    return dfun_delta_v2(d, ja, la, lb, lc)

_cg_cache = {}
def cg_coef(j1, j2, m1, m2, j, m):
    key = (j1, j2, m1, m2, j, m)
    if key not in _cg_cache:
        _cg_cache[key] = float(CG(j1, m1, j2, m2, j, m).doit().evalf())
    return _cg_cache[key]

_cg_matrix_cache = {}
def cg_matrix(ja, jb, jc, ls_list, out_spins):
    key = (ja, jb, jc, tuple(ls_list), tuple(out_spins[0]), tuple(out_spins[1]))
    if key in _cg_matrix_cache:
        return _cg_matrix_cache[key]
    ret = np.zeros((len(ls_list), len(out_spins[0]), len(out_spins[1])))
    for i, (ell, spin) in enumerate(ls_list):
        for ib, lambda_b in enumerate(out_spins[0]):
            for ic, lambda_c in enumerate(out_spins[1]):
                ret[i, ib, ic] = (
                    math.sqrt(2 * ell + 1)
                    / math.sqrt(2 * ja + 1)
                    * cg_coef(jb, jc, lambda_b, -lambda_c, spin, lambda_b - lambda_c)
                    * cg_coef(ell, spin, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
                )
    _cg_matrix_cache[key] = ret
    return ret

def compute_chain_boosts(particle_p4, chain):
    particle_set = {name for _, outs in chain for name in outs}
    core_decay_map = {}
    part_data = {}
    pending = list(chain)
    while pending:
        extra = []
        for core, outs in pending:
            if core == "Bp":
                p_rest = particle_p4[core]
                part_data[core] = {"rest_p": {}}
                for out in outs:
                    core_decay_map[out] = core
                    part_data[core]["rest_p"][out] = rest_vector(p_rest, particle_p4[out])
                    particle_set.discard(out)
                for other in list(particle_set):
                    part_data[core]["rest_p"][other] = rest_vector(p_rest, particle_p4[other])
            elif core in core_decay_map:
                parent = core_decay_map[core]
                p_rest = part_data[parent]["rest_p"][core]
                part_data[core] = {"rest_p": {}}
                for out in outs:
                    core_decay_map[out] = core
                    part_data[core]["rest_p"][out] = rest_vector(p_rest, part_data[parent]["rest_p"][out])
                    particle_set.discard(out)
                for other in list(particle_set):
                    part_data[core]["rest_p"][other] = rest_vector(p_rest, part_data[parent]["rest_p"][other])
            else:
                extra.append((core, outs))
        pending = extra
    return part_data

def calculate_helicity_angles(particle_p4, chain):
    part_data = compute_chain_boosts(particle_p4, chain)
    n_events = particle_p4["Bp"].shape[0]
    set_x = {"Bp": np.broadcast_to(np.array([1.0, 0.0, 0.0]), (n_events, 3))}
    set_z = {"Bp": np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n_events, 3))}
    angles = {}
    for core, outs in chain:
        angles[core] = {}
        bias = -np.pi
        for out in outs:
            z2 = part_data[core]["rest_p"][out][..., 1:]
            ang, x_axis = angle_zx_z_getx(set_z[core], set_x[core], z2)
            set_x[out] = x_axis
            set_z[out] = z2
            ang["alpha"] = (ang["alpha"] - bias) % (2 * np.pi) + bias
            bias -= np.pi
            angles[core][out] = ang
    return angles

def spin_values(j):
    return tuple(np.arange(-j, j + 1, 1, dtype=float))

def ad_hoc_mass(m0, m_max, m_min):
    k = (m_max - m_min) / 2.0
    return k * (1.0 + np.tanh((2.0 * m0 - (m_max + m_min)) / k / 4.0)) + m_min

def event_relative_p2(m0, m1, m2):
    mass_sum = m1 + m2
    mass_difference = m1 - m2
    product = (m0 - mass_sum) * (m0 + mass_sum)
    product *= (m0 - mass_difference) * (m0 + mass_difference)
    return product / (4.0 * m0 * m0)

def nominal_relative_p2(m0, m1, m2):
    mass_sum = m1 + m2
    mass_difference = m1 - m2
    product = (m0 - mass_sum) * (m0 + mass_sum)
    product *= (m0 - mass_difference) * (m0 + mass_difference)
    return product / (2.0 * m0) ** 2

def standard_vertex_mdep(ls_list, g_ls, mass_core, q2, q02, has_barrier=True):
    g_ls = np.asarray(g_ls, dtype=complex)
    n_events = np.asarray(mass_core).reshape(-1).shape[0]
    if not has_barrier:
        return np.broadcast_to(g_ls.reshape(1, -1), (n_events, len(g_ls))).copy()
    factors = [
        barrier_factor2(ell, mass_core, q2, q02, d=3.0, barrier_factor_norm=True).reshape(-1)
        for ell, _ in ls_list
    ]
    return np.stack(factors, axis=-1).astype(complex) * g_ls.reshape(1, -1)

def bwr_ls_vertex_mdep(m, m0, width, theta0, ls_list, q2, q02, g_ls, sign):
    fractions = np.asarray([np.cos(theta0), np.sin(theta0)], dtype=float)
    partial = []
    for fraction, (ell, _) in zip(fractions, ls_list):
        bf = np.sqrt(q2 / q02) ** ell * bprime_q2(ell, q2, q02, d=3.0)
        partial.append(fraction * bf)
    partial = np.stack(partial, axis=-1)
    a = m0 * m0 - m * m
    b = m0 * width * np.sqrt(q2 / q02) * np.sum(partial * partial, axis=-1) * m / m0
    denominator = a - 1j * b
    return sign * partial / denominator[:, None] * np.asarray(g_ls, dtype=complex).reshape(1, -1)

def helicity_decay_amp_from_mdep(core_j, out_js, core_spins, out_spins, ls_list, m_dep, angle):
    m_dep = np.asarray(m_dep, dtype=complex)
    cg = cg_matrix(core_j, out_js[0], out_js[1], ls_list, out_spins).astype(complex)
    h = np.sum(m_dep[:, :, None, None] * cg[None, :, :, :], axis=1)
    d_conj = get_d_matrix_lambda(angle, core_j, core_spins, out_spins[0], out_spins[1])
    return h[:, None, :, :] * d_conj.reshape(-1, len(core_spins), len(out_spins[0]), len(out_spins[1]))

def create_model(params, config_yml, charge="Cminus"):
    nominal_mass = {
        "Bp": float(config_yml["particle"]["$top"]["Bp"]["mass"]),
        "D": float(config_yml["particle"]["$finals"]["D"]["mass"]),
        "K": float(config_yml["particle"]["$finals"]["K"]["mass"]),
        "D0": float(config_yml["particle"]["$finals"]["D0"]["mass"]),
        "pi": float(config_yml["particle"]["$finals"]["pi"]["mass"]),
        "Dst": float(config_yml["particle"]["Dst"]["mass"]),
    }

    def param_complex(base):
        return float(params[base + "r"]) * np.exp(1j * float(params[base + "i"]))

    def fitted_ls(prefix, count):
        return [param_complex(f"{prefix}_g_ls_{index}") for index in range(count)]

    charge_val = 1 if str(charge).lower() in ["cplus", "+1", "1"] else -1
    c_sign = 1.0 if charge_val > 0 else -1.0
    c_scale_c2 = 0.0 if charge_val > 0 else 1.0

    component_specs = [
        dict(name="X(3872)", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((0, 1), (2, 1)), model="BWR_LS", sign=c_sign, below=True),
        dict(name="X(3915)(0-)", topology="dstd", j=0, root_ls=((0, 0),), decay_ls=((1, 1),), model="BWR", sign=c_sign),
        dict(name="chi(c2)(3930)", topology="dstd", j=2, root_ls=((2, 2),), decay_ls=((2, 1),), model="BWR", sign=c_sign),
        dict(name="X(3940)(1.)", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((0, 1), (2, 1)), model="BWR_LS", sign=1.0),
        dict(name="X(3993)", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((0, 1), (2, 1)), model="BWR_LS", sign=c_sign),
        dict(name="Psi(4040)", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((1, 1),), model="BWR", sign=1.0),
        dict(name="X(4300)", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((0, 1), (2, 1)), model="BWR_LS", sign=1.0),
        dict(name="NR(0-)SPp", topology="dstd", j=0, root_ls=((0, 0),), decay_ls=((1, 1),), model="New", sign=c_sign),
        dict(name="NR(1.)PSp", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((0, 1),), model="one", sign=c_sign),
        dict(name="NR(0-)SPm", topology="dstd", j=0, root_ls=((0, 0),), decay_ls=((1, 1),), model="one", sign=1.0),
        dict(name="NR(1-)PPm", topology="dstd", j=1, root_ls=((1, 1),), decay_ls=((1, 1),), model="one", sign=1.0),
        dict(name="X0(2900)", topology="dk", j=0, root_ls=((1, 1),), decay_ls=((0, 0),), model="BWR", sign=c_scale_c2),
        dict(name="X1(2900)", topology="dk", j=1, root_ls=((0, 0), (1, 1), (2, 2)), decay_ls=((1, 0),), model="BWR", sign=c_scale_c2),
    ]

    for spec in component_specs:
        spec["mass"] = float(params.get(spec["name"] + "_mass", 4.35))
        if spec["model"] in ("BWR", "BWR_LS"):
            spec["width"] = float(params[spec["name"] + "_width"])

    def dstd_component_amplitude(final_p4, spec):
        name = spec["name"]
        p4 = {key: np.asarray(value) for key, value in final_p4.items()}
        p4["Dst"] = p4["D0"] + p4["pi"]
        p4["R"] = p4["Dst"] + p4["D"]
        p4["Bp"] = p4["R"] + p4["K"]
        masses = {key: invariant_mass(value) for key, value in p4.items()}
        chain = [("Bp", ["R", "K"]), ("R", ["Dst", "D"]), ("Dst", ["D0", "pi"])]
        angles = calculate_helicity_angles(p4, chain)

        q2_root = event_relative_p2(masses["Bp"], masses["R"], masses["K"])
        q02_root = nominal_relative_p2(nominal_mass["Bp"], spec["mass"], nominal_mass["K"])
        root_prefix = f"Bp->{name}.K"
        root_g = fitted_ls(root_prefix, len(spec["root_ls"]))
        root_mdep = standard_vertex_mdep(spec["root_ls"], root_g, masses["Bp"], q2_root, q02_root)

        q2_decay = event_relative_p2(masses["R"], masses["Dst"], masses["D"])
        q0_mass = spec["mass"]
        if spec.get("below", False):
            q0_mass = ad_hoc_mass(
                q0_mass,
                nominal_mass["Bp"] - nominal_mass["K"],
                nominal_mass["Dst"] + nominal_mass["D"],
            )
        q02_decay = nominal_relative_p2(q0_mass, nominal_mass["Dst"], nominal_mass["D"])
        decay_prefix = f"{name}->Dst.D"
        decay_g = fitted_ls(decay_prefix, len(spec["decay_ls"]))
        if spec["model"] == "BWR_LS":
            decay_mdep = bwr_ls_vertex_mdep(
                masses["R"], spec["mass"], spec["width"], float(params[name + "_theta0"]),
                spec["decay_ls"], q2_decay, q02_decay, decay_g, spec["sign"],
            )
            particle_factor = np.ones_like(masses["R"], dtype=complex)
        else:
            decay_mdep = standard_vertex_mdep(spec["decay_ls"], decay_g, masses["R"], q2_decay, q02_decay)
            if spec["model"] == "BWR":
                particle_factor = spec["sign"] * bwr(
                    masses["R"], spec["mass"], spec["width"],
                    np.sqrt(q2_decay), np.sqrt(q02_decay), spec["decay_ls"][0][0], d=3.0,
                )
            elif spec["model"] == "New":
                alpha = float(params[name + "_alpha"])
                beta = float(params[name + "_beta"])
                particle_factor = spec["sign"] * np.exp(-(alpha + 1j * beta) * (masses["R"] ** 2 - spec["mass"] ** 2))
            else:
                particle_factor = np.full_like(masses["R"], spec["sign"], dtype=complex)

        dst_mdep = standard_vertex_mdep(((1, 0),), [param_complex("Dst->D0.pi_g_ls_0")], masses["Dst"], np.zeros_like(masses["Dst"]), 0.0, has_barrier=False)
        root_amp = helicity_decay_amp_from_mdep(0, (spec["j"], 0), (0,), (spin_values(spec["j"]), (0,)), spec["root_ls"], root_mdep, angles["Bp"]["R"])
        decay_amp = helicity_decay_amp_from_mdep(spec["j"], (1, 0), spin_values(spec["j"]), (spin_values(1), (0,)), spec["decay_ls"], decay_mdep, angles["R"]["Dst"])
        dst_amp = helicity_decay_amp_from_mdep(1, (0, 0), spin_values(1), ((0,), (0,)), ((1, 0),), dst_mdep, angles["Dst"]["D0"])
        total = param_complex(f"Bp->{name}.K{name}->Dst.DDst->D0.pi_total_0") * particle_factor
        tensor = np.einsum("...agd,...gfb,...fce,...->...abcde", root_amp, decay_amp, dst_amp, total)
        return tensor.reshape(len(masses["Bp"]), -1)[:, 0]

    def dk_component_amplitude(final_p4, spec):
        if spec.get("sign", 1.0) == 0.0:
            return np.zeros(len(final_p4["D"]), dtype=complex)
        name = spec["name"]
        p4 = {key: np.asarray(value) for key, value in final_p4.items()}
        p4["Dst"] = p4["D0"] + p4["pi"]
        p4["R"] = p4["D"] + p4["K"]
        p4["Bp"] = p4["R"] + p4["Dst"]
        masses = {key: invariant_mass(value) for key, value in p4.items()}
        chain = [("Bp", ["R", "Dst"]), ("R", ["D", "K"]), ("Dst", ["D0", "pi"])]
        angles = calculate_helicity_angles(p4, chain)

        q2_root = event_relative_p2(masses["Bp"], masses["R"], masses["Dst"])
        q02_root = nominal_relative_p2(nominal_mass["Bp"], spec["mass"], nominal_mass["Dst"])
        root_g = fitted_ls(f"Bp->{name}.Dst", len(spec["root_ls"]))
        root_mdep = standard_vertex_mdep(spec["root_ls"], root_g, masses["Bp"], q2_root, q02_root)

        q2_decay = event_relative_p2(masses["R"], masses["D"], masses["K"])
        q02_decay = nominal_relative_p2(spec["mass"], nominal_mass["D"], nominal_mass["K"])
        decay_g = fitted_ls(f"{name}->D.K", len(spec["decay_ls"]))
        decay_mdep = standard_vertex_mdep(spec["decay_ls"], decay_g, masses["R"], q2_decay, q02_decay)
        particle_factor = bwr(
            masses["R"], spec["mass"], spec["width"],
            np.sqrt(q2_decay), np.sqrt(q02_decay), spec["decay_ls"][0][0], d=3.0,
        )
        dst_mdep = standard_vertex_mdep(((1, 0),), [param_complex("Dst->D0.pi_g_ls_0")], masses["Dst"], np.zeros_like(masses["Dst"]), 0.0, has_barrier=False)

        root_amp = helicity_decay_amp_from_mdep(0, (spec["j"], 1), (0,), (spin_values(spec["j"]), spin_values(1)), spec["root_ls"], root_mdep, angles["Bp"]["R"])
        decay_amp = helicity_decay_amp_from_mdep(spec["j"], (0, 0), spin_values(spec["j"]), ((0,), (0,)), spec["decay_ls"], decay_mdep, angles["R"]["D"])
        dst_amp = helicity_decay_amp_from_mdep(1, (0, 0), spin_values(1), ((0,), (0,)), ((1, 0),), dst_mdep, angles["Dst"]["D0"])
        total = param_complex(f"Bp->{name}.Dst{name}->D.KDst->D0.pi_total_0") * particle_factor
        tensor = np.einsum("...agf,...gde,...fbc,...->...abcde", root_amp, decay_amp, dst_amp, total)
        return tensor.reshape(len(masses["Bp"]), -1)[:, 0]

    def isolated_amplitude(final_p4):
        total_amp = None
        component_amps = {}
        for spec in component_specs:
            if spec["topology"] == "dstd":
                amp = dstd_component_amplitude(final_p4, spec)
            else:
                amp = dk_component_amplitude(final_p4, spec)
            component_amps[spec["name"]] = amp
            if total_amp is None:
                total_amp = np.zeros_like(amp)
            total_amp += amp
        return total_amp, component_amps

    return isolated_amplitude, component_specs

def main():
    parser = argparse.ArgumentParser(description="Evaluate Conversion-Corrected Isolated TF-PWA amplitudes.")
    parser.add_argument("--events", type=str, default=None, help="Path to sampled events JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path for text complex amplitudes")
    parser.add_argument("--analysis-dir", type=str, default=None, help="Path to Analysis directory containing params and config")
    parser.add_argument("--charge", type=str, default="Cminus", choices=["Cminus", "Cplus", "-1", "+1", "1"], help="Charge configuration (Cminus or Cplus)")
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    suite_dir = work_dir.parent
    charge_tag = "Cplus" if str(args.charge).lower() in ["cplus", "+1", "1"] else "Cminus"

    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
    else:
        analysis_candidates = [
            suite_dir.parent / "data",
            suite_dir.parent / "Analysis",
            suite_dir.parent / "archive" / "investigation" / "Analysis",
            suite_dir.parent / "B2DxDK.jl" / "data",
            suite_dir.parent / "B2DxDK.jl" / "Analysis",
            suite_dir.parent / "B2DxDK.jl" / "archive" / "investigation" / "Analysis",
            suite_dir.parent.parent / "data",
            suite_dir.parent.parent / "B2DxDK.jl" / "archive" / "investigation" / "Analysis",
        ]
        analysis_dir = next(
            (p for p in analysis_candidates if (p / "final_params_full.json").exists() and (p / "config_a.yml").exists()),
            None,
        )
        if not analysis_dir:
            raise FileNotFoundError("Could not find directory with final_params_full.json and config_a.yml. Please specify --analysis-dir.")

    with open(analysis_dir / "final_params_full.json", "r", encoding="utf-8") as f:
        params = json.load(f)["value"]
    with open(analysis_dir / "config_a.yml", "r", encoding="utf-8") as f:
        config_yml = yaml.safe_load(f)

    if args.events:
        events_path = Path(args.events)
    else:
        candidates = [
            suite_dir.parent / "data" / "sampled_events_tfpwa.json",
            suite_dir.parent / "B2DxDK.jl" / "data" / "sampled_events_tfpwa.json",
            suite_dir.parent.parent / "data" / "sampled_events_tfpwa.json",
            suite_dir.parent.parent / "B2DxDK.jl" / "data" / "sampled_events_tfpwa.json",
        ]
        events_path = next((p for p in candidates if p.exists()), None)
        if not events_path:
            raise FileNotFoundError("Could not find sampled_events_tfpwa.json. Please specify --events.")

    print(f"Loading events from: {events_path}")
    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p4_data = data["p4"]
    final_p4 = {name: np.asarray(p4_data[name], dtype=float) for name in ["D", "K", "D0", "pi"]}
    n_events = len(final_p4["D"])
    print(f"Loaded {n_events} events. Evaluating isolated TF-PWA model (charge={charge_tag})...")

    model_fn, specs = create_model(params, config_yml, charge=charge_tag)
    total_amp, comp_amps = model_fn(final_p4)

    output_path = Path(args.output) if args.output else suite_dir / "amp_data" / charge_tag / "isolated_tfpwa_amp.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.column_stack([np.real(total_amp), np.imag(total_amp)]), fmt="%.17e", header="real imag")
    print(f"Saved {n_events} complex amplitudes (ASCII text) to: {output_path}")

if __name__ == "__main__":
    main()
