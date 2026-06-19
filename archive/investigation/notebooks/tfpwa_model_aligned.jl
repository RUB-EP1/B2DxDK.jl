### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    # CODEX ALIGNMENT CHANGE:
    # The previous notebook used ThreeBodyDecays/FourVectors and a different
    # crosscheck event.  This notebook is now a self-contained Julia counterpart
    # of Analysis/psi4040_independent_amplitude_flow.ipynb.
    #
    # It intentionally reads only the same scalar model inputs from
    # Analysis/config_a.yml and Analysis/final_params_full.json, then uses the
    # same hardcoded four-vectors from tf_pwa_analysis_Gemini.py.
    using LinearAlgebra
    using Printf
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
md"""
# Isolated TF-PWA Psi(4040) Amplitude in Julia

This Pluto notebook mirrors the validated Python notebook
`Analysis/psi4040_independent_amplitude_flow.ipynb`.

The calculation reconstructs the isolated chain

```text
Bp -> Psi(4040) K
Psi(4040) -> Dst D
Dst -> D0 pi
```

and contracts the same helicity-decay tensors with the same running-width
Breit-Wigner particle factor used by TF-PWA for `Psi(4040)`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000003
begin
    # CODEX ALIGNMENT CHANGE:
    # Local parsers avoid extra Julia package installation while still taking
    # the parameters from the same YAML/JSON files used by the Python notebook.
    const investigation_root = normpath(joinpath(@__DIR__, ".."))
    analysis_dir = joinpath(investigation_root, "Analysis")
    config_text = read(joinpath(analysis_dir, "config_a.yml"), String)
    params_text = read(joinpath(analysis_dir, "final_params_full.json"), String)

    function parse_mass_from_yaml(text::String, label::String)
        pattern = Regex("(?m)^\\s*" * replace(label, "(" => "\\(", ")" => "\\)") *
                        ":\\s*\\n(?:.*\\n){0,8}?\\s*mass:\\s*([-+0-9.eE]+)")
        m = match(pattern, text)
        m === nothing && error("Could not find mass for " * label)
        return parse(Float64, m.captures[1])
    end

    function parse_json_number(text::String, key::String)
        escaped_key = replace(key, "(" => "\\(", ")" => "\\)")
        m = match(Regex("\"" * escaped_key * "\"\\s*:\\s*([-+0-9.eE]+)"), text)
        m === nothing && error("Could not find JSON key " * key)
        return parse(Float64, m.captures[1])
    end

    # CODEX ALIGNMENT CHANGE:
    # Same nominal masses and Psi(4040) fit parameters as the isolated Python
    # implementation.  Event masses are still computed from the hardcoded event.
    nominal_mass = Dict(
        "Bp" => parse_mass_from_yaml(config_text, "Bp"),
        "D" => parse_mass_from_yaml(config_text, "D"),
        "K" => parse_mass_from_yaml(config_text, "K"),
        "D0" => parse_mass_from_yaml(config_text, "D0"),
        "pi" => parse_mass_from_yaml(config_text, "pi"),
        "Dst" => parse_mass_from_yaml(config_text, "Dst"),
        "Psi(4040)" => parse_json_number(params_text, "Psi(4040)_mass"),
    )
    psi_width = parse_json_number(params_text, "Psi(4040)_width")

    # CODEX ALIGNMENT CHANGE:
    # Same event four-vectors as tf_pwa_analysis_Gemini.py and the Python
    # notebook, in TF-PWA order [E, px, py, pz].
    p4 = Dict(
        "D" => [2.0452, -0.1467, 0.2235, -0.7847],
        "D0" => [2.2606, 0.2284, -0.3689, 1.2019],
        "K" => [0.7718, -0.0873, 0.1803, -0.5584],
        "pi" => [0.2017, 0.0056, -0.0349, 0.1413],
    )

    # TF-PWA's local C(...) selector is c=-1 here. For Psi(4040), C=-1 makes
    # the wrapper pass the BWR factor through unchanged in this selected probe.
    c_selector = -1.0

    # The Python probe overwrites all selected couplings to 1 + 0im.
    g_total = 1.0 + 0.0im
    g_bp_to_psi_k = 1.0 + 0.0im
    g_psi_to_dst_d = 1.0 + 0.0im
    g_dst_to_d0_pi = 1.0 + 0.0im
end

# ╔═╡ 00000000-0000-0000-0000-000000000004
begin
    const EPS = 1.0e-8

    dot3(a, b) = dot(a, b)
    norm3(a) = norm(a)
    unitvec(v) = v ./ norm(v)

    function cross_unit(a, b)
        c = cross(a, b)
        if norm(c) < EPS
            c = cross(a, ones(3) .+ b)
        end
        return unitvec(c)
    end

    angle_from(v, x_axis, y_axis) = atan(dot3(v, y_axis), dot3(v, x_axis))

    function angle_zx_z_getx(z1, x1, z2)
        # Physical meaning: construct the Euler rotation from an old frame
        # (z1,x1) to the helicity frame whose z-axis follows daughter z2.
        u_z1 = unitvec(z1)
        u_z2 = unitvec(z2)
        u_y1 = cross_unit(z1, x1)
        u_x1 = cross_unit(u_y1, z1)
        u_yr = cross_unit(z1, z2)
        u_xr = cross_unit(u_yr, z1)
        alpha = angle_from(u_xr, u_x1, u_y1)
        beta = angle_from(u_z2, u_z1, u_xr)
        gamma = 0.0
        u_x2 = cross_unit(u_yr, u_z2)
        return (alpha=alpha, beta=beta, gamma=gamma), u_x2
    end

    invariant_mass(v) = sqrt(abs(v[1]^2 - sum(abs2, v[2:4])))
    boost_vector(v) = v[2:4] ./ v[1]

    function boost(v, beta)
        beta2 = dot(beta, beta)
        gamma = 1.0 / sqrt(1.0 - beta2)
        bp = dot(beta, v[2:4])
        gamma2 = beta2 > EPS ? (gamma - 1.0) / beta2 : 0.0
        spatial = v[2:4] .+ gamma2 * bp .* beta .+ gamma * v[1] .* beta
        energy = gamma * (v[1] + bp)
        return [energy; spatial]
    end

    rest_vector(core, other) = boost(other, -boost_vector(core))

    get_relative_p2(m0, m1, m2) =
        ((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)

    function bprime_polynomial(l, z)
        coeff = Dict(
            0 => [1.0],
            1 => [1.0, 1.0],
            2 => [1.0, 3.0, 9.0],
            3 => [1.0, 6.0, 45.0, 225.0],
        )[l]
        y = 0.0
        for c in coeff
            y = y * z + c
        end
        return y
    end

    function bprime_q2(l, q2, q02; d=3.0)
        z0 = q02 * d^2
        z = q2 * d^2
        ratio = bprime_polynomial(l, z0) / bprime_polynomial(l, z)
        return sqrt(ratio > 0 ? ratio : 1.0)
    end

    function barrier_factor2(l, q2, q02; d=3.0, barrier_factor_norm=true)
        # Centrifugal barrier q^L B'_L(q,q0,d), normalized as TF-PWA does.
        tmp = q2^(l / 2) * bprime_q2(l, q2, q02; d)
        if barrier_factor_norm
            tmp /= abs(q02)^(l / 2)
        end
        return tmp
    end

    function gamma_running(m, gamma0, q, q0, l, m0; d=3.0)
        qq0 = q0 > 1.0e-15 ? (q / q0)^(2l + 1) : 1.0
        mm0 = m0 / m
        bp = (sqrt(bprime_polynomial(l, (q0 * d)^2)) /
              sqrt(bprime_polynomial(l, (q * d)^2)))^2
        return gamma0 * qq0 * mm0 * bp
    end

    function bwr(m, m0, gamma0, q, q0, l; d=3.0)
        # Relativistic Breit-Wigner propagator with running width.
        gamma_m = gamma_running(m, gamma0, q, q0, l, m0; d)
        x = m0^2 - m^2
        y = m0 * gamma_m
        denom = x^2 + y^2
        return x / denom + im * y / denom
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000005
begin
    # CODEX ALIGNMENT CHANGE:
    # Wigner-D and Clebsch-Gordan helpers are local Julia translations of the
    # Python notebook's TF-PWA-specialized helpers.  Only integer-spin cases
    # needed by this chain are required here.

    half_factorial(x::Int) = factorial(x ÷ 2)

    function small_d_weight(j2::Int)
        ret = zeros(Float64, j2 + 1, j2 + 1, j2 + 1)
        for m in -j2:2:j2
            for n in -j2:2:j2
                for k in max(0, n - m):2:min(j2 - m, j2 + n)
                    sign_power = (k + m - n) ÷ 2
                    sign = isodd(sign_power) ? -1.0 : 1.0
                    val = sign * sqrt(
                        half_factorial(j2 + m) *
                        half_factorial(j2 - m) *
                        half_factorial(j2 + n) *
                        half_factorial(j2 - n)
                    )
                    val /= (
                        half_factorial(j2 - m - k) *
                        half_factorial(j2 + n - k) *
                        half_factorial(k) *
                        half_factorial(k + m - n)
                    )
                    ell = (2k + (m - n)) ÷ 2
                    ret[ell + 1, (m + j2) ÷ 2 + 1, (n + j2) ÷ 2 + 1] = val
                end
            end
        end
        return ret
    end

    function small_d_matrix(theta, j2::Int)
        sc = [sin(theta / 2)^p * cos(theta / 2)^(j2 - p) for p in 0:j2]
        weights = small_d_weight(j2)
        out = zeros(Float64, j2 + 1, j2 + 1)
        for i in 1:j2+1, j in 1:j2+1, p in 1:j2+1
            out[i, j] += sc[p] * weights[p, i, j]
        end
        return out
    end

    function d_matrix_conj(alpha, beta, gamma, j2::Int)
        mvals = collect((-j2 / 2):1:(j2 / 2))
        dsmall = small_d_matrix(beta, j2)
        out = zeros(ComplexF64, j2 + 1, j2 + 1)
        for i in 1:j2+1, j in 1:j2+1
            out[i, j] = exp(im * alpha * mvals[i]) * exp(im * gamma * mvals[j]) * dsmall[i, j]
        end
        return out
    end

    function fact_nonneg(x)
        xi = round(Int, x)
        xi < 0 && return 0.0
        return Float64(factorial(big(xi)))
    end

    function phase_minus_one(n)
        ni = round(Int, n)
        return isodd(ni) ? -1.0 : 1.0
    end

    function wigner3j(j1, j2, j3, m1, m2, m3)
        abs(m1 + m2 + m3) > 1.0e-12 && return 0.0
        (abs(m1) > j1 || abs(m2) > j2 || abs(m3) > j3) && return 0.0
        (j3 > j1 + j2 || j3 < abs(j1 - j2)) && return 0.0

        delta = fact_nonneg(j1 + j2 - j3) *
                fact_nonneg(j1 - j2 + j3) *
                fact_nonneg(-j1 + j2 + j3) /
                fact_nonneg(j1 + j2 + j3 + 1)
        pref = phase_minus_one(j1 - j2 - m3) * sqrt(delta)
        pref *= sqrt(
            fact_nonneg(j1 + m1) * fact_nonneg(j1 - m1) *
            fact_nonneg(j2 + m2) * fact_nonneg(j2 - m2) *
            fact_nonneg(j3 + m3) * fact_nonneg(j3 - m3)
        )

        zmin = max(0, round(Int, j2 - j3 - m1), round(Int, j1 - j3 + m2))
        zmax = min(round(Int, j1 + j2 - j3), round(Int, j1 - m1), round(Int, j2 + m2))
        zmin > zmax && return 0.0

        total = 0.0
        for z in zmin:zmax
            denom =
                fact_nonneg(z) *
                fact_nonneg(j1 + j2 - j3 - z) *
                fact_nonneg(j1 - m1 - z) *
                fact_nonneg(j2 + m2 - z) *
                fact_nonneg(j3 - j2 + m1 + z) *
                fact_nonneg(j3 - j1 - m2 + z)
            total += phase_minus_one(z) / denom
        end
        return pref * total
    end

    function cg_coef(j1, j2, m1, m2, j, m)
        abs(m1 + m2 - m) > 1.0e-12 && return 0.0
        return phase_minus_one(j1 - j2 + m) * sqrt(2j + 1) *
               wigner3j(j1, j2, j, m1, m2, -m)
    end

    function cg_factor(ja, jb, jc, lambda_b, lambda_c, ell, spin)
        delta = lambda_b - lambda_c
        return sqrt(2ell + 1) / sqrt(2ja + 1) *
               cg_coef(jb, jc, lambda_b, -lambda_c, spin, delta) *
               cg_coef(ell, spin, 0, delta, ja, delta)
    end

    function d_lookup(dmat, ja, lambda_a, lambda_b, lambda_c)
        delta = lambda_b - lambda_c
        abs(delta) > ja && return 0.0 + 0.0im
        ia = round(Int, lambda_a + ja) + 1
        idelta = round(Int, delta + ja) + 1
        return dmat[ia, idelta]
    end

    function helicity_decay_amp(; core_j, out_js, core_spins, out_spins,
                                ls_list, g_ls, angle, q2, q02,
                                has_barrier=true, barrier_norm=true)
        # Tensor amplitude for one two-body vertex:
        # D^J*(angles) times LS-to-helicity recoupling times form factor.
        bf = 1.0
        if has_barrier
            ell = ls_list[1][1]
            bf = barrier_factor2(ell, q2, q02; d=3.0, barrier_factor_norm=barrier_norm)
        end
        dmat = d_matrix_conj(angle.alpha, angle.beta, angle.gamma, round(Int, 2core_j))
        amp = zeros(ComplexF64, length(core_spins), length(out_spins[1]), length(out_spins[2]))
        for (ia, lambda_a) in enumerate(core_spins)
            for (ib, lambda_b) in enumerate(out_spins[1])
                for (ic, lambda_c) in enumerate(out_spins[2])
                    rec = 0.0
                    for (ell, spin) in ls_list
                        rec += cg_factor(core_j, out_js[1], out_js[2], lambda_b, lambda_c, ell, spin)
                    end
                    amp[ia, ib, ic] = g_ls * bf * rec *
                                      d_lookup(dmat, core_j, lambda_a, lambda_b, lambda_c)
                end
            end
        end
        return amp
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000006
begin
    function compute_chain_boosts(particle_p4, chain)
        particle_set = Set{String}()
        for (_, outs) in chain
            foreach(x -> push!(particle_set, x), outs)
        end

        core_decay_map = Dict{String,String}()
        part_data = Dict{String,Dict{String,Vector{Float64}}}()
        pending = copy(chain)
        while !isempty(pending)
            extra = []
            for (core, outs) in pending
                if core == "Bp"
                    p_rest = particle_p4[core]
                    part_data[core] = Dict{String,Vector{Float64}}()
                    for out in outs
                        core_decay_map[out] = core
                        part_data[core][out] = rest_vector(p_rest, particle_p4[out])
                        delete!(particle_set, out)
                    end
                    for other in collect(particle_set)
                        part_data[core][other] = rest_vector(p_rest, particle_p4[other])
                    end
                elseif haskey(core_decay_map, core)
                    parent = core_decay_map[core]
                    p_rest = part_data[parent][core]
                    part_data[core] = Dict{String,Vector{Float64}}()
                    for out in outs
                        core_decay_map[out] = core
                        part_data[core][out] = rest_vector(p_rest, part_data[parent][out])
                        delete!(particle_set, out)
                    end
                    for other in collect(particle_set)
                        part_data[core][other] = rest_vector(p_rest, part_data[parent][other])
                    end
                else
                    push!(extra, (core, outs))
                end
            end
            pending = extra
        end
        return part_data
    end

    function calculate_helicity_angles(particle_p4, chain)
        # Construct the same helicity frames and alpha wrapping convention used
        # by TF-PWA's cal_helicity_angle path.
        part_data = compute_chain_boosts(particle_p4, chain)
        set_x = Dict("Bp" => [1.0, 0.0, 0.0])
        set_z = Dict("Bp" => [0.0, 0.0, 1.0])
        angles = Dict{String,Dict{String,NamedTuple}}()
        for (core, outs) in chain
            angles[core] = Dict{String,NamedTuple}()
            bias = -pi
            for out in outs
                z2 = part_data[core][out][2:4]
                ang, x_axis = angle_zx_z_getx(set_z[core], set_x[core], z2)
                set_x[out] = x_axis
                set_z[out] = z2
                alpha = mod(ang.alpha - bias, 2pi) + bias
                bias -= pi
                angles[core][out] = (alpha=alpha, beta=ang.beta, gamma=ang.gamma)
            end
        end
        return angles
    end

    function format_complex(z)
        sign = imag(z) >= 0 ? "+" : "-"
        return @sprintf("%.16g %s %.16gj", real(z), sign, abs(imag(z)))
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000007
begin
    # CODEX ALIGNMENT CHANGE:
    # Step-by-step execution mirrors the Python notebook, including all print
    # statements needed to compare intermediate physics quantities.
    println("Step 1: Reconstruct intermediate four-vectors by summing daughters.")
    p4["Dst"] = p4["D0"] .+ p4["pi"]
    p4["Psi(4040)"] = p4["Dst"] .+ p4["D"]
    p4["Bp"] = p4["Psi(4040)"] .+ p4["K"]
    for name in ["Dst", "Psi(4040)", "Bp"]
        println(@sprintf("  %9s p4 = %s", name, string(p4[name])))
    end

    println("\nStep 2: Compute invariant masses from the event kinematics.")
    event_mass = Dict(name => invariant_mass(vec) for (name, vec) in p4)
    for name in ["Bp", "Psi(4040)", "Dst", "D", "K", "D0", "pi"]
        println(@sprintf("  m(%s) = %.12f GeV", name, event_mass[name]))
    end

    println("\nStep 3: Compute helicity Euler angles for the sequential decay chain.")
    chain = [
        ("Bp", ["Psi(4040)", "K"]),
        ("Psi(4040)", ["Dst", "D"]),
        ("Dst", ["D0", "pi"]),
    ]
    angles = calculate_helicity_angles(p4, chain)
    for (core, outs) in chain
        for out in outs
            a = angles[core][out]
            println(@sprintf("  %9s -> %-9s: alpha=% .12f, beta=% .12f, gamma=% .12f",
                             core, out, a.alpha, a.beta, a.gamma))
        end
    end

    println("\nStep 4: Compute breakup momenta q^2 and nominal q0^2 for barrier factors.")
    q2_bp = get_relative_p2(event_mass["Bp"], event_mass["Psi(4040)"], event_mass["K"])
    q02_bp = get_relative_p2(nominal_mass["Bp"], nominal_mass["Psi(4040)"], nominal_mass["K"])
    q2_psi = get_relative_p2(event_mass["Psi(4040)"], event_mass["Dst"], event_mass["D"])
    q02_psi = get_relative_p2(nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])
    q2_dst = get_relative_p2(event_mass["Dst"], event_mass["D0"], event_mass["pi"])
    println(@sprintf("  Bp -> Psi K:      q2=%.12f, q02=%.12f", q2_bp, q02_bp))
    println(@sprintf("  Psi -> Dst D:     q2=%.12f, q02=%.12f", q2_psi, q02_psi))
    println(@sprintf("  Dst -> D0 pi:     q2=%.12f; no barrier factor in this config", q2_dst))

    println("\nStep 5: Build the three helicity-decay tensors.")
    spin0 = [0]
    spin1 = [-1, 0, 1]

    amp_bp = helicity_decay_amp(
        core_j=0, out_js=(1, 0), core_spins=spin0, out_spins=(spin1, spin0),
        ls_list=[(1, 1)], g_ls=g_bp_to_psi_k, angle=angles["Bp"]["Psi(4040)"],
        q2=q2_bp, q02=q02_bp, has_barrier=true, barrier_norm=true,
    )
    println("  Bp -> Psi K tensor size $(size(amp_bp)); flat = $(vec(amp_bp))")

    amp_psi = helicity_decay_amp(
        core_j=1, out_js=(1, 0), core_spins=spin1, out_spins=(spin1, spin0),
        ls_list=[(1, 1)], g_ls=g_psi_to_dst_d, angle=angles["Psi(4040)"]["Dst"],
        q2=q2_psi, q02=q02_psi, has_barrier=true, barrier_norm=true,
    )
    println("  Psi -> Dst D tensor size $(size(amp_psi)); flat = $(vec(amp_psi))")

    amp_dst = helicity_decay_amp(
        core_j=1, out_js=(0, 0), core_spins=spin1, out_spins=(spin0, spin0),
        ls_list=[(1, 0)], g_ls=g_dst_to_d0_pi, angle=angles["Dst"]["D0"],
        q2=q2_dst, q02=0.0, has_barrier=false, barrier_norm=false,
    )
    println("  Dst -> D0 pi tensor size $(size(amp_dst)); flat = $(vec(amp_dst))")

    println("\nStep 6: Compute particle factors.")
    q_psi = sqrt(q2_psi)
    q0_psi = sqrt(q02_psi)
    psi_factor = bwr(event_mass["Psi(4040)"], nominal_mass["Psi(4040)"],
                     psi_width, q_psi, q0_psi, 1; d=3.0)
    dst_factor = 1.0 + 0.0im
    println("  Psi(4040) C(BWR) factor = $(format_complex(psi_factor))")
    println("  Dst model-one factor     = $(format_complex(dst_factor))")

    println("\nStep 7: Contract the tensors exactly like DecayChain.get_amp.")
    particle_factor = g_total * psi_factor * dst_factor
    amplitude = let particle_factor = particle_factor
        total = zero(ComplexF64)
        for ig in eachindex(spin1), ifst in eachindex(spin1)
            total += amp_bp[1, ig, 1] * amp_psi[ig, ifst, 1] *
                     amp_dst[ifst, 1, 1] * particle_factor
        end
        total
    end
    println("  independent Julia Psi(4040) amplitude = $(format_complex(amplitude))")

    println("\nStep 8: Numerical regression check against the validated Python/TF-PWA value.")
    tfpwa_reference = -0.0006049977354135942 - 0.0030870270680673287im
    delta = amplitude - tfpwa_reference
    println("  TF-PWA reference amplitude = $(format_complex(tfpwa_reference))")
    println(@sprintf("  difference                 = %.3e + %.3ej", real(delta), imag(delta)))
    @assert abs(delta) < 2.0e-12 "Julia implementation does not match TF-PWA reference closely enough."
    println("  PASS: Julia implementation matches isolated Python/TF-PWA within tolerance.")

    amplitude
end

# ╔═╡ Cell order:
# ╠═00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
# ╠═00000000-0000-0000-0000-000000000003
# ╠═00000000-0000-0000-0000-000000000004
# ╠═00000000-0000-0000-0000-000000000005
# ╠═00000000-0000-0000-0000-000000000006
# ╠═00000000-0000-0000-0000-000000000007
