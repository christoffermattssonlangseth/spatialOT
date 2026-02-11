# spatial-OT

Utilities and notebook workflow for optimal transport (OT) transition analysis on RRMap / Spatial-IT `AnnData`.

## Main files

- `rrmap_ot.py`: OT transition utilities (balanced/unbalanced, centroid/cell modes).
- `output/jupyter-notebook/spatial-it-rrmap-ot-analysis.ipynb`: analysis notebook.

## Notebook workflow

The notebook computes OT transitions between consecutive course labels, then:

- builds state-to-state transition matrices
- reports top-k destination states per source state
- generates QC summaries (entropy, unbalanced mass diagnostics)
- exports CSV tables

## Recommended setup

- Open notebook from this repo (`spatial-OT`) root.
- Use Jupyter kernel `sc_py312`.
- Set `ADATA_PATH` in notebook config to your `.h5ad` file.

## Model-aware runs

The notebook supports model-specific ordering to avoid cross-model transitions:

- `MODELS`
- `MODEL_NAME` (`"Chronic"` or `"RR"`)
- `STRIP_COURSE_LABELS = True` to normalize labels like `"onset II "` vs `"onset II"`

## Outputs

Generated tables are written under:

- `output/ot_transition_tables/Chronic/`
- `output/ot_transition_tables/RR/`
- `output/ot_transition_tables/full_dataset/`
