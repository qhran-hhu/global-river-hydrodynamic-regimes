# Input data — how to obtain

No raw data are shipped in this repository. Obtain the inputs as follows:

| Dataset | Source | Access |
|---|---|---|
| SWOT L2 RiverSP reach time series | PO.DAAC **Hydrocron** API — https://podaac.github.io/hydrocron | public, no login |
| SWOT SoS L4 (discharge priors, per continent) | PO.DAAC via NASA Earthdata | free Earthdata account; set `EARTHDATA_USER` / `EARTHDATA_PASSWORD` env vars |
| GRanD / GDAT global dam points | NASA Earthdata GIS service (see `code/download_grand.py`) | public |
| GSIM streamflow indices | https://doi.org/10.1594/PANGAEA.887477 | public |
| Yangtze gauge records (Three Gorges ground truth) | Changjiang Water Resources Commission | restricted — not redistributable; scripts expecting these records (`tgd_fingerprint.py`, `tgd_net_effect.py`) exit with a clear message if the file is absent |

The derived reach-level feature matrix produced by Stage 2
(97,566 quality-controlled reaches) will be deposited on Zenodo with a DOI
upon formal publication of the manuscript.
