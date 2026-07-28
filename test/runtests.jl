using Test
using Arrow
using B2DxDK
using CascadeDecays
using DataFrames

include("amplitude_crosscheck.jl")

const crosscheck_data_path = joinpath(data_dir, "crosscheck.arrow")
const amplitude_reference_path = joinpath(data_dir, "crosscheck_amplitudes_reference.txt")
const amplitude_tolerance = 1e-10

@testset "B2DxDK" begin
    @testset "model table" begin
        @test nrow(resonance_chains_df) == 19
        @test length(all_resonance_names) == 13
        @test Set(resonance_chains_df.resonance_name) == Set(all_resonance_names)
        @test all(!isempty, resonance_chain_names.(all_resonance_names))
    end

    @testset "cascade construction" begin
        cascade = build_all_resonance_cascade()
        @test cascade !== nothing
        for name in all_resonance_names
            @test !isempty(resonance_chain_names(name))
            @test build_resonance_cascade(name) !== nothing
        end
        @test merge([build_resonance_cascade(name) for name in all_resonance_names]...) !== nothing
    end

    @testset "particle-2 convention" begin
        x1_rows = collect(eachrow(B2DxDK.resonance_chain_rows("X1(2900)")))
        @test length(x1_rows) == 3
        @test [row.root_recoupling for row in x1_rows] == [:missing_particle2, :standard, :missing_particle2]
        @test first(eachrow(B2DxDK.resonance_chain_rows("X0(2900)"))).root_recoupling == :standard
        dk_rows = resonance_chains_df[resonance_chains_df.topology.==:dk, :]
        @test all(row -> row.root_recoupling in (:standard, :missing_particle2), eachrow(dk_rows))
    end

    @testset "amplitude crosscheck" begin
        @test isfile(crosscheck_data_path)
        @test isfile(amplitude_reference_path)

        df = DataFrame(Arrow.Table(crosscheck_data_path))
        cascade = build_all_resonance_cascade()
        resonance_amps_by_event = compute_resonance_amplitudes(df, cascade)

        @test length(resonance_amps_by_event) == nrow(df) == 100

        max_abs_delta, max_rel_delta, ref_header = max_amplitude_delta(
            amplitude_reference_path,
            df,
            resonance_amps_by_event,
        )
        @test max_abs_delta <= amplitude_tolerance
        @test max_rel_delta <= amplitude_tolerance

        expected_columns = 2 + 2 * length(all_resonance_names) + 2  # event, weight, components, total
        @test length(ref_header) == expected_columns
    end

    @testset "amplitude totals" begin
        df = DataFrame(Arrow.Table(crosscheck_data_path))
        cascade = build_all_resonance_cascade()
        resonance_amps_by_event = compute_resonance_amplitudes(df, cascade)

        for (idx, component_amps) in pairs(resonance_amps_by_event)
            total = sum(component_amps)
            row = amplitude_row(idx, df, component_amps)
            @test row[end - 1] ≈ real(total)
            @test row[end] ≈ imag(total)
        end
    end

    @testset "first-event spot check" begin
        df = DataFrame(Arrow.Table(crosscheck_data_path))
        cascade = build_all_resonance_cascade()
        amps = only(compute_resonance_amplitudes(df[1:1, :], cascade))

        _, ref_rows = load_amplitude_reference(amplitude_reference_path)
        ref = ref_rows[1]
        @test ref[1] ≈ 1.0
        @test ref[2] ≈ df.weight[1]

        for (i, name) in pairs(all_resonance_names)
            re_idx = 2 + 2(i - 1) + 1
            im_idx = re_idx + 1
            @test ref[re_idx] ≈ real(amps[i]) atol=amplitude_tolerance
            @test ref[im_idx] ≈ imag(amps[i]) atol=amplitude_tolerance
        end
    end
end
