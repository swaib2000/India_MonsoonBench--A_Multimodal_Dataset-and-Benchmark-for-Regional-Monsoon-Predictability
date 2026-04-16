# Result Summaries

This directory contains lightweight benchmark result summaries that are safe to version-control.

Included artifacts:

- `metadata/`: processed regional dataset summaries, state patch counts, and IMD LPA normal metadata.
- `regional_benchmark_summaries/`: compact CSV summaries for regional monthly autoregressive neural baselines and ordinal metrics.
- `simple_baselines/`: persistence and seasonal climatology summaries, plus neural-vs-simple comparison.
- `transition_analysis/`: best model by monsoon transition phase.

Excluded artifacts:

- full raster stacks,
- NumPy patch arrays,
- model checkpoints,
- per-sample predictions,
- training logs,
- full `baseline_runs/` directories.

These larger artifacts should be distributed via a separate data archive rather than GitHub.
