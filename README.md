# MSI-Bench: data and code

Data and reference implementation for

> Hongze Liu, Zhengjiang Liu, Muzhuang Guo. *A credibility assessment and conflict-aware
> evidence fusion framework for multi-source maritime safety information.*
> Navigation College, Dalian Maritime University, Dalian 116026, China.
> Corresponding author: Muzhuang Guo, gmzdlmu@dlmu.edu.cn

This release contains everything needed to (a) inspect the benchmark and (b) re-run
the framework and recompute the numbers reported in the paper. It is CPU-only and
needs no API key.

* Repository: <https://github.com/open-msi-lab/msi-bench>
* Version: 1.0.0
* Licences: data CC BY 4.0 for the synthetic
  overlay and the derived files, public domain for the NGA archive; code MIT.
  See `LICENSE-DATA` and `LICENSE-CODE`.

## 1. Contents

```
README.md  LICENSE-CODE  LICENSE-DATA  CITATION.cff  DATASHEET.md
MANIFEST.sha256        SHA-256 of every file listed below (BUILD-INFO.json excluded)
BUILD-INFO.json        build metadata: sources, source commit, counts, sanitization hits
code/                  the msi_bench package, tests/, requirements.txt,
                       config.example.yaml, .env.example
data/                  NAVAREA IV archive: 2012.csv ... 2024.csv and sample.csv
benchmark/             MSI-Bench, schema-aligned variant (the primary benchmark)
benchmark_degraded/    MSI-Bench, schema-degraded variant (raw text only)
results/               frozen outputs of the experiments reported in the paper
```

Integrity check (from this directory): `shasum -a 256 -c MANIFEST.sha256`
(macOS) or `sha256sum -c MANIFEST.sha256` (Linux).

## 2. The data

### 2.1 Real backbone: NGA navigational warnings (`data/`)

The backbone is the navigational-warning archive of the U.S. National
Geospatial-Intelligence Agency (NGA), Maritime Safety Information:
<https://msi.nga.mil>. The warnings are United States government works and are in the
public domain; they are redistributed here in full and unmodified. All text is English.

* One file per year, 2012-2024, plus `sample.csv` (25 rows, the same twelve columns).
* Columns: `msgYear, msgNumber, navArea, subregion, text, status, issueDate,
  authority, cancelDate, cancelNavArea, cancelMsgYear, cancelMsgNumber`
  (the yearly files carry one further, empty column).
* The files hold several NGA warning series (field `navArea`). The paper analyses
  NAVAREA IV (`navArea == 4`): 14,274 messages over 2012-2024. The loader in
  `code/msi_bench/parsing/` selects them.

Partitions used in the paper: 2012-2021 for archive statistics and the fitted
vocabulary, 2022-2023 as the primary benchmark (70/30 development/test split at the
event level), 2024 as a single-shot temporal holdout.

### 2.2 MSI-Bench (`benchmark/`, `benchmark_degraded/`)

MSI-Bench couples the 1,118 real NAVAREA IV events of 2022-2023 with a simulated
multi-source and adversarial overlay. Language models wrote narrative text only;
coordinates, timestamps, device/network/submitter metadata, tampering perturbations
and forged provenance come from code, seeded with 20260613 (reference time
2024-06-13). `DATASHEET.md` documents motivation, composition and intended use.

| File | Content |
|---|---|
| `records.json` | 5,228 records: official, genuine crowd, genuine sensor, and the five attack classes (T1 fabrication, T2 replay, T3 tampering, T4 impersonation, T5 Sybil). Fields: `record_id, event_type, provider_type, provider_id, authority_norm, raw_text, structured_data, geometry_kind, coordinates, centroid, spatial_uncertainty_nm, issue_time, valid_start, valid_end, device_class, auth_level, network_origin, submission_position, observation_type`, ground truth `gt_category, gt_event_id, gt_quality_tier`, and `provenance` (split, generator, attack metadata) |
| `events.json` | the 1,118 event identities (lifecycle chains of the archive), event type, centroid, split |
| `histories.json` | simulated submission histories of the contributor population (4,794 identities) |
| `attack_reserve.json` | 3,762 additional attack records (disjoint namespace) used for the adversarial-pressure sweep |
| `meta.json` | counts, seed, reference time, generator identifiers |
| `DATASHEET.md` | datasheet |

`benchmark_degraded/` has the same files; its crowd records carry no extracted
structured attributes (the pipeline re-derives type, position and time from text).

Generator models (pinned in `meta.json`): generator A `gpt-5.2-codex`
(provider key `openai_responses`), generator B `claude-sonnet-4-6` (provider key
`anthropic_proxy`), judge C `gpt-5.3-codex`. The generation prompts are in
`code/msi_bench/generation/generators.py`. The provider keys `anthropic_proxy` and
`openai_responses` recorded in `meta.json` (and the alias `openai_proxy`) select the
standard Anthropic / OpenAI SDK clients; an optional `base_url` in the provider block of
`config.yaml`, or the environment variable `LLM_BASE_URL`, points them at any compatible
gateway, and the official endpoint is used otherwise. Tokens are read from the
environment variable named by `api_key_env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
or `LLM_GATEWAY_API_TOKEN`; see `code/.env.example`). Because hosted model versions are retired,
the generated text is shipped (`raw_text` in `records.json`); it is never regenerated
by the experiments.

## 3. Installation

Python 3.9 or newer (the verification of this release was run with Python 3.12). From this directory:

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, scipy, scikit-learn, shapely, geographiclib,
pyyaml, matplotlib. `sentence-transformers` and `torch` provide the sentence encoder
(all-MiniLM-L6-v2, downloaded from the Hugging Face hub on first use; without it the
code falls back to TF-IDF and the clustering numbers change). `anthropic`, `openai`
and `google-genai` are only needed to generate new benchmark text; `xgboost` is used
by one baseline. The released code contains no downloader or crawler.

Check the installation:

```bash
python3 tests/test_framework.py        # or: pytest tests/    (31 tests, mock mode)
```

## 4. Recomputing the results

The experiments read `data/` and the benchmark directories, and write to
`<output_dir>/<experiment>/`. Default `output_dir` is `results/`, which holds the
frozen outputs. To compare without overwriting them, create a local configuration
first (run from `code/`):

```bash
sed 's|^  output_dir: results .*|  output_dir: results_recomputed|' config.example.yaml > config.yaml
```

`config.example.yaml` carries the settings of the paper (sentence encoder on, gated
temporal mode, `theta_ndup` 0.75, attack reserve 5.0). With `llm.enabled: false`
(the default) a deterministic mock stands in for every language model, so no key or
network access is needed.

```bash
python3 -m msi_bench.experiments.run e0                              # archive characterisation
python3 -m msi_bench.experiments.run e3r                             # real lifecycle-chain recovery
python3 -m msi_bench.experiments.run e1 --benchmark ../benchmark     # credibility
python3 -m msi_bench.experiments.run e2 --benchmark ../benchmark     # dimension ablation
python3 -m msi_bench.experiments.run e3 --benchmark ../benchmark     # clustering on the overlay
python3 -m msi_bench.experiments.run e4 --benchmark ../benchmark     # fusion
python3 -m msi_bench.experiments.run e5 --benchmark ../benchmark     # sensitivity
python3 -m msi_bench.experiments.run e6 --benchmark ../benchmark     # end to end
python3 -m msi_bench.experiments.run e7 --benchmark ../benchmark     # alternative paradigms
python3 -m msi_bench.experiments.run fidelity --benchmark ../benchmark
```

The schema-degraded bracket is obtained by running `e1`, `e3` and `e4` with
`--benchmark ../benchmark_degraded` (and a different `output_dir`).

### 4.1 Manuscript table and figure to result file

All paths are relative to this directory; "keys" are JSON keys in the named file.

| Manuscript item | Frozen result (keys) | Command |
|---|---|---|
| Section 4, archive counts (14,274 messages; 10,816 / 2,933 / 525 by period) | `results/e0/e0_summary.json` (`messages_by_year_function`, `event_chains_benchmark_years`) | `e0` |
| Table 2, benchmark composition | `benchmark/meta.json` (`category_counts`, `reserve_counts`, `attack_prevalence_pct`); counts of `gt_category` and `provenance.split` in `benchmark/records.json` | see 4.2 |
| Table 3 and Fig. 3(a), credibility assessment | `results/e1/e1_results.json` (`auc_roc_primary`, `auc_pr_primary`, `macro_auc_test`, `weighting_methods_primary`, `per_dimension_auc_primary`, `primary_auc_multiseed`); `results/e2/e2_results.json` (single-dimension intervals); `results/e7/e7_results.json` (`methods`, `mcam_vs_baselines`) | `e1`, `e2`, `e7` |
| Fig. 3(b) and the operating points of Section 4.1 | `results/e1/e1_results.json` (`detection_far_curve_test`, `per_attack_operating_test`, `operating_threshold_dev`, `per_attack_auc_test`) | `e1` |
| Section 4.1, leave-one-out ablation | `results/e2/e2_results.json` (`variants`, `delta_from_full`, `full_vs_ablation_significance`) | `e2` |
| Table 4 and Fig. 4, clustering (rows: spatial DBSCAN, weighted-sum ST DBSCAN, agglomerative, no type constraint, STS-DBSCAN; noise rates) | `results/e3r/e3r_results.json` (`methods`, `constraint_effect`); `results/e3/e3_results.json` (`methods`, `noise_classification`) | `e3r`, `e3` |
| Table 5 and Fig. 5(a), fusion | `results/e4/e4_results.json` (`methods_test`, `q_sweep_test`, `conflict_validation`, `zadeh_stress_test`, `conflict_management`) | `e4` |
| Table 6 and Fig. 5(b), Sybil-majority records | `results/e4/e4_results.json` (`independence_discount_test.discount_off` / `discount_on`, `sybil_majority_by_tier`); `results/e6/e6_results.json` (`sybil_majority_fused`) | `e4`, `e6` |
| Table 7 and Fig. 6, end to end, attack prevalence, 2024 holdout | `results/e6/e6_results.json` (`pipeline_ablation` with `*_test` keys, `adversarial_pressure`, `temporal_holdout`) | `e6` |
| Table A.1, parameter values | `code/config.example.yaml`; `results/e5/e5_results.json` (sensitivity) | `e5` |

In the result files the code's vocabulary is used: "independence discount" is the
effective-source folding of the paper, "Standard D-S" is Dempster's rule,
"conflict" covers the content contradictions of the paper, and `J` is a composite
index that the paper does not report.

Some values in the paper are counts or supplementary analyses derived from the
pipeline output and are not stored in the frozen result files: the two-radius
ST-DBSCAN rows, the two similarity ablations and the bootstrap interval of Table 4;
the two-source-rule column of Table 6; the breakdown of undetected events in
Section 4.4; and Table 8 (the run with every official record withheld). They can be
recomputed with the released code from `data/` and `benchmark/`: Table 8, for
example, is `msi_bench.experiments.common.run_pipeline` applied to the benchmark
with the records of `gt_category == "OFFICIAL"` removed.

### 4.2 Table 2 counts and the paraphrase-gate audit

```bash
python3 - <<'EOF'
import json, collections
rec = json.load(open("../benchmark/records.json"))
res = json.load(open("../benchmark/attack_reserve.json"))
print(collections.Counter(r["gt_category"] for r in rec))
para = lambda rs: [r["provenance"]["containment_vs_campaign"] for r in rs
                   if r["provenance"].get("identity_tier") == "paraphrased"]
print(len(para(rec)), len(para(res)), max(para(rec) + para(res)))
EOF
```

### 4.3 What does not recompute offline, and known deviations

Four reported quantities involve fresh language-model calls and cannot be regenerated
from this release: the five-seed regeneration of the overlay (`primary_auc_multiseed`
in `e1_results.json`), the 2024 holdout (`temporal_holdout` in `e6_results.json`; it
builds a new overlay), the zero-shot language-model judge row of Table 3
(`e7_results.json`) and the schema bracket as published (`brackets_results.json`,
which rebuilds both variants). Offline, the runners substitute the deterministic mock
for these, so a re-run produces different numbers for exactly these entries; use the
frozen files.

Everything else is computed from `data/` and the shipped benchmark. With Python 3.12
and current library versions, `e0`, `e1` (except the multi-seed block), `e2`, `e3`,
`e3r`, `e4`, `e5`, `e6` (except the holdout and wall-clock `latency_s`) and `fidelity`
reproduced the frozen JSON values exactly. In `e7` the rule-based, Dawid-Skene and
MCAM rows are exact; the gradient-boosted row (XGBoost) differs in the third decimal
and depends on the XGBoost version (0.7991 against the frozen 0.7990).

On macOS, importing PyTorch and XGBoost in one process can crash with a segmentation
fault (duplicate OpenMP runtimes); run `e7` with `OMP_NUM_THREADS=1` in that case.

## 5. What is not included

* The language-model response cache and generation transcript (prompts and raw model
  responses). The narrative text it produced is in `records.json` and
  `attack_reserve.json`.
* API keys, `.env`, `config.yaml` and the other local configuration files.
* Derived caches (parsed-corpus and embedding caches, Python bytecode); the code
  rebuilds them on first use.
* Manuscript sources, working notes, and results of preliminary configurations.
* Any code to acquire warnings from the NGA website. The warnings are provided in
  `data/`.

## 6. How to cite

Cite the paper and this release (see `CITATION.cff`): <https://github.com/open-msi-lab/msi-bench>.
