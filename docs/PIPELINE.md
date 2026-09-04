# Pipeline execution order and module map

All scripts live flat in `code/` (they import each other), self-locate via
`Path(__file__)`, and write under `code/output/`. Run stages in order.

## Stage 1 — data acquisition

| Script | Input | Output |
|---|---|---|
| `sos_download.py` | NASA Earthdata login (`EARTHDATA_USER` / `EARTHDATA_PASSWORD` env vars) | SWOT SoS L4 per-continent files → `data/sos/` |
| `extract_sos_reaches.py` | SoS L4 files | global reach list + mean-discharge reference (`output/全球reach清单.csv`) |
| `hydrocron_fetch.py` | reach list; public Hydrocron API | per-reach WSE/width time series → `output/ts_shards_global/` (resumable) |
| `download_grand.py` | NASA Earthdata GIS service | GRanD/GDAT dam points → `output/human_activity/` |

## Stage 2 — feature matrix

| Script | Role |
|---|---|
| `features.py` | **shared module**: noise-robust feature suite f1–f7 (series cleaning, harmonic fit, ρv² proxy, width bias correction) |
| `build_features_v1.py` | vectorised feature computation over ~139k reaches → `output/features_v1_parts/` |
| `assemble_features_v1.py` | merge parts, add f1 ranks, f7 coupling, metadata, QC flags → `output/特征矩阵_v1.parquet` + `特征矩阵_v1_qc.parquet` (97,566 reaches) |
| `make_slope_fill.py` | neighbourhood fill of missing slopes → `output/slope_fill.parquet` |
| `f2_recompute_depth.py` | **f2 elevation-confound fix**: depth-relative variability (P90−P10)/mean of h_rel = WSE − P5(WSE); patches the matrix |
| `f5_recompute_masked.py` + `patch_f5_matrix.py` | masked-harmonic refit of f5 (gating-bug fix); writes columns back |

## Stage 3 — regime space and continuum evidence

| Script | Role |
|---|---|
| `regime_space.py` | low-dimensional embedding, Hartigan dip test, scale-space mode persistence → `output/regime_space/` |
| `gmm_converged.py` | GMM BIC vs k with n_init=10 (supersedes the k-selection part of `cluster.py`; shows no elbow, nested overlapping components) |
| `cluster.py` | multi-method clustering + cross-method ARI (0.14–0.24) |
| `regime_map.py` | maps the continuum back to geographic space (Robinson projection helpers used by figures) |
| `modality_robustness.py` | D1: construction-independent unimodality (feature ablations incl. slope-free set) |

## Stage 4 — validation modules

| Script | Role |
|---|---|
| `d2_snapshot_fidelity.py` | D2: short-record fidelity — SWOT halves vs GSIM long records |
| `d3_noise_attenuation.py` | D3: measurement-noise attenuation correction |
| `check_phase_wse_vs_q.py` | WSE-vs-discharge seasonal phase cross-check |
| `c_supplementary.py` | C1 QC-threshold sensitivity + C2 end-member subgroup profiles |

## Stage 5 — drivers and human fingerprint

| Script | Role |
|---|---|
| `driver_attribution.py` | circularity-corrected driver attribution (climate / terrain / human) |
| `tgd_fingerprint.py` | Three Gorges Dam displacement in feature space (needs Yangtze gauge records — not redistributable) |
| `tgd_net_effect.py` | TGD net-effect recompute (year-by-year) |
| `dam_contrast.py` | global dam vs free-flowing reach contrast + coverage audit |
| `dam_contrast_slope.py` | dam-siting selection correction via slope matching (median effect −0.3 to −1.2%) |
| `power_mde.py` | statistical power / minimum detectable effect for the dam null |

## Stage 6 — discharge-regime comparison and scaling

| Script | Role |
|---|---|
| `compare_q_regime.py` | GSIM discharge regimes vs SWOT hydrodynamic regimes (weak coupling) |
| `r5_hydraulic_geometry.py` | hydraulic-geometry exponent β global map and scaling law |

## Stage 7 — publication figures and supplement

| Script | Product |
|---|---|
| `fig1_publication.py` | Fig. 1 global hydrodynamic-regime map |
| `fig2_publication.py` | Fig. 2 continuum evidence (4-panel) |
| `fig3_publication.py` | Fig. 3 driver attribution |
| `fig4_publication.py` | Fig. 4 human fingerprint / dam displacement |
| `fig5_publication.py` | Fig. 5 discharge vs hydrodynamic regimes |
| `manuscript/assemble_supplementary.py` | assembles Supplementary S1–S5 (docx + md) from `code/output/regime_space/` |

## Pilot provenance

`feature_matrix_v0.py` — 369-reach pilot builder from the feasibility phase,
kept for provenance; superseded by Stage 2.
