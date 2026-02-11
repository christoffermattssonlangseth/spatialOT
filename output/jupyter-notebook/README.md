# Spatial-IT / RRMap OT Notebook

Notebook: `spatial-it-rrmap-ot-analysis.ipynb`

## What it does

Runs optimal transport (OT) transitions between consecutive course labels using `rrmap_ot.py`, then:

- builds state-to-state transition matrices
- reports top-k destination states per source state
- generates QC summaries (entropy, unbalanced mass diagnostics)
- exports CSV tables

## Required setup

- Open this notebook from the `spatial-OT` repo.
- Use Jupyter kernel `sc_py312` (the default in notebook metadata).
- Set `ADATA_PATH` in the configuration cell to your `.h5ad` file.

## Model-aware execution (recommended)

The notebook is configured to run one model at a time via:

- `MODELS`
- `MODEL_NAME` (`"Chronic"` or `"RR"`)

This prevents cross-model transitions (for example, `chronic long -> PLP CFA`) that can happen in a full-dataset chain.

## Course label cleanup

`STRIP_COURSE_LABELS = True` trims whitespace in course labels before filtering/order matching.  
This handles labels like `"onset II "` vs `"onset II"`.

## Outputs

Exported tables are written to:

- `output/ot_transition_tables/Chronic/` when `MODEL_NAME="Chronic"`
- `output/ot_transition_tables/RR/` when `MODEL_NAME="RR"`
- `output/ot_transition_tables/full_dataset/` when `MODEL_NAME=None`

Files include:

- `transition_<course_t>__to__<course_tp1>.csv`
- `topk_<course_t>__to__<course_tp1>.csv`

## Typical run

1. Update `ADATA_PATH`.
2. Choose `MODEL_NAME`.
3. Run all cells from top to bottom.
