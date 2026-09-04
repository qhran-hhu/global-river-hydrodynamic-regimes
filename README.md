# A global continuum of river hydrodynamic regimes — analysis code

Analysis code for the manuscript **"A global continuum of river hydrodynamic regimes"**
(Qihua Ran, Hohai University). The study uses 3.4 years of SWOT (Surface Water and
Ocean Topography) observations over 97,566 quality-controlled river reaches to build
the first global map of river *hydrodynamic* regimes — the dynamics of the streamwise
momentum flux density ρv² — and shows that this regime space is a unimodal continuum
ordered primarily by macro-climate, with visible but local human fingerprints.

A companion methodology paper, *"Beyond Water Volume: A Parallel Hydrodynamic Framework
for River Basin Characterization"* (in preparation for *Water Resources Research*),
defines and validates the feature set on Chinese gauge records.

## Repository layout

```
code/            analysis pipeline (flat layout; scripts import each other)
  features.py    noise-robust dynamical feature suite f1–f7 (shared module)
  plotstyle.py   shared matplotlib/seaborn setup
docs/PIPELINE.md execution order and module map
data/README.md   how to obtain the input datasets
manuscript/      supplementary-material assembly script
```

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Quick start

The pipeline runs in numbered stages (see `docs/PIPELINE.md` for the full DAG):

```bash
cd code
# 1. data acquisition (requires a free NASA Earthdata account for SoS L4)
set EARTHDATA_USER=...   & set EARTHDATA_PASSWORD=...   # Windows
python sos_download.py --dest ../data/sos
python extract_sos_reaches.py
python hydrocron_fetch.py            # SWOT reach time series via the public Hydrocron API
python download_grand.py             # GRanD/GDAT global dam points

# 2. feature matrix
python build_features_v1.py
python assemble_features_v1.py
python make_slope_fill.py
python f2_recompute_depth.py         # depth-relative water-level variability
python f5_recompute_masked.py && python patch_f5_matrix.py

# 3. regime space, statistics, drivers, human fingerprint, figures
python regime_space.py
python gmm_converged.py
...
python fig1_publication.py           # … fig5_publication.py
```

All scripts locate their own directory and write under `code/output/`; no
absolute paths are required. Input data are not shipped — see `data/README.md`.

## Data availability

* **SWOT reach time series** — public via the PO.DAAC **Hydrocron** API
  (SWOT River Database, SWOT_L2_HR_RiverSP).
* **SoS L4** (SWOT prior river database, discharge priors) — NASA Earthdata login.
* **GRanD/GDAT** global dam database — public.
* **GSIM** streamflow indices — public (doi:10.1594/PANGAEA.887477).
* **Yangtze gauge records** (Three Gorges ground truth) — courtesy of the
  Changjiang Water Resources Commission; restrictions apply, not redistributable.
* The **derived reach-level feature matrix** (`特征矩阵_v1_qc.parquet`, 97,566 reaches)
  will be deposited on Zenodo with a DOI upon formal publication.

## Known pre-release notes

* Output *directory* names are anglicised (`output/human_activity/`,
  `output/feature_matrix_v0/`); some output *file* labels remain in Chinese and
  will be anglicised in the final release pass before Zenodo deposit.
* `feature_matrix_v0.py` is the pilot-stage (369-reach) builder kept for provenance.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

## License

MIT — see `LICENSE`.

## Contact

Qihua Ran, Hohai University, Nanjing, China.
