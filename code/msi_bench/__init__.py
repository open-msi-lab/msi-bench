"""MSI-Bench: a semi-synthetic benchmark and reference framework for
credibility assessment and fusion of multi-source Maritime Safety Information.

The package is organised into independent, composable layers:

* ``msi_bench.config``    — configuration, paths, seeds, .env loading
* ``msi_bench.schema``    — the MSI-Model record type and enumerations (Section 3.2)
* ``msi_bench.geo``       — coordinate parsing and geodesy
* ``msi_bench.temporal``  — DTG (date-time-group) parsing
* ``msi_bench.parsing``   — the parsing pipeline over the real NAVAREA archive
* ``msi_bench.credibility`` — the MCAM (Section 3.3)
* ``msi_bench.similarity``  — composite similarity (Section 3.4)
* ``msi_bench.clustering``  — STS-DBSCAN (Section 3.4)
* ``msi_bench.conflict``    — conflict taxonomy + partitioning (Section 3.5)
* ``msi_bench.fusion``      — credibility-weighted D-S fusion (Section 3.6)
* ``msi_bench.generation``  — the synthetic multi-source overlay (Section 4.3)
* ``msi_bench.experiments`` — experiment runners E0–E7

Only the parsing pipeline and Experiment E0 are needed to reproduce the
real-data characterisation; they have no LLM/API dependency.
"""

__version__ = "0.1.0"
