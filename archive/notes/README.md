# Convention notes

Working notes on helicity/LS conventions written while the model was being matched
to TF-PWA. Kept for the reasoning behind signs that survive in the production code
(chiefly `MAGIC_SIGNS` in [`src/matching.jl`](../../src/matching.jl)); not needed to
run or read the model.

| file | what it covers |
|---|---|
| [`note-cascadedecays-v040.md`](note-cascadedecays-v040.md) | The `CascadeDecays v0.1.0 → v0.4.0` upgrade: which angles the child-aware helicity-frame fix moves, why the amplitude is not invariant under it, the measured effect on TF-PWA agreement, and the API port. Derivation behind `MAGIC_SIGNS["X1(2900)"] = -1`. |
| [`note-phase-two.md`](note-phase-two.md) | Partial-wave algebra of the Jacob–Wick particle-2 phase at the dk root, as it stood on `v0.1.0`. The algebra holds; its conclusion that a helicity-dependent recoupling is unavoidable was superseded by the above. |

The active TF-PWA review note, which is a deliverable rather than a working note,
stays at [`docs/tfpwa_review/`](../../docs/tfpwa_review/).
