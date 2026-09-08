# Artifact manifest

Every number, table and figure in the paper, mapped to the script that produces it and
the files that script reads and writes. Reviewers asked for this mapping explicitly.

## Protocol

All reported estimates use one protocol, implemented in `scripts/cr_final_tables.py`
and the shared helpers in `scripts/_cr_common.py`:

| | |
|---|---|
| Grouping | **source**, not record. CtrlPairs: normalized `(question, context, answer)` triple of the underlying NQ-Swap QA (1,200 cell-A rows → 555 groups). RAGTruth: normalized source passage (2,000 responses → 450 groups). `item_id` indexes a swap record / a single response and does **not** isolate sources. |
| Residualization | fitted inside each training fold, applied to that fold's test rows |
| Scoring | signed AUROC; polarity fixed on the training fold; no `max(AUC, 1-AUC)` anywhere |
| Folds | explicitly seeded group splitter (`seeded_group_folds`), mean ± sd over 10 splits |
| Intervals | cluster bootstrap over source groups, B = 2000 |
| Layer | fixed a priori at relative depth 0.5, never argmax-selected |

`sklearn.model_selection.GroupKFold` is deliberately not used: it assigns folds by
`np.argsort` over group sizes, and our group sizes are almost all tied, so the split —
and hence the AUROC — changes with the numpy version.

## Reproduction

```bash
pip install -r requirements.txt
bash reproduce.sh            # CPU analysis from cached features
bash reproduce.sh extract    # regenerate the features on one A100-40G
```

Stage B needs `runs/*/features.npz` (2.4 GB), which is too large for the repository and
is not redistributed; `bash reproduce.sh extract` regenerates it. `reproduce.sh` checks
for it and exits with instructions rather than failing part-way.

## Paper → producer

| Paper object | Producer | Reads | Writes |
|---|---|---|---|
| Table 1 (C1: supp `O=1` raw / +conf / cross-strata, CIs) | `scripts/cr_final_tables.py` | `runs/*_v2/features.npz`, `data/ctrlpairs_v2.jsonl` | `runs/cr_final_tables.json` → `table1` |
| Table 2 (C3: transported-to-native cosine, transfer AUROC, nulls) | `scripts/harden_analyses.py`, `scripts/heldout_procrustes.py` | `runs/*_v2/features.npz` | `runs/hardened.json` → `c3`, `runs/heldout_procrustes.json` |
| Table 2 nuisance rows (length / overlap / factuality / confidence) | `scripts/cr_band_and_pairs.py` | `runs/*_v2/features.npz` | `runs/cr_band_and_pairs.json` → `c3_per_pair` |
| Table 3 (C4 pooled: synth-d_S, conf, in-domain, NLI, overlap, CIs) | `scripts/cr_final_tables.py` | `runs/*_v2/features.npz`, `runs/*_rt/features.npz`, `data/ragtruth.jsonl` | `runs/cr_final_tables.json` → `table3` |
| Table 4 (C4 per task type) | `scripts/cr_final_tables.py` | as Table 3 | `runs/cr_final_tables.json` → `table4_pertask`, `prose.per_task_mean_over_models` |
| Table 4 Sent-NLI rows | `scripts/nli_sentence_ragtruth.py` | `data/ragtruth.jsonl` | `runs/nli_sentence_ragtruth.json` |
| Figure 1 (decoupling) | `scripts/make_figures.py::fig_decoupling` | `data/ctrlpairs_v2.jsonl`, `runs/v1_corr.json` | `paper/figs/fig_decoupling.pdf` |
| Figure 2 (C1 by layer) | `scripts/make_figures.py::fig_c1_layers` | `runs/*_v2/features.npz`, `runs/*_v2_nc/features.npz` | `paper/figs/fig_c1_layers.pdf`, `runs/fig_c1_layers_values.json` |
| Figure 3 (C3 transfer heatmaps) | `scripts/make_figures.py::fig_c3_heatmap` | `runs/*_v2/features.npz` | `paper/figs/fig_c3_heatmap.pdf`, `runs/fig_c3_heatmap_values.json` |
| Figure 4 (C4 RAGTruth) | `scripts/make_figures.py::fig_c4_ragtruth` | `runs/cr_final_tables.json` (same source as Table 3, so bars and table cannot diverge) | `paper/figs/fig_c4_ragtruth.pdf` |
| §3 construction counts, corr(S,O), donor collisions | `scripts/cr_dupleak_audit.py`, `scripts/harden_analyses.py` | `data/ctrlpairs_v2.jsonl` | `runs/cr_dupleak_audit.json`, `runs/hardened.json` → `cellC` |
| §3 NLI fooled rate on clean cell C | `scripts/harden_analyses.py` | `data/ctrlpairs_v2.jsonl` | `runs/hardened.json` → `cellC` |
| §5.1 no-context / shuffled ablation | `scripts/analyze_nocontext.py` | `runs/*_v2{,_nc,_shuf}/features.npz` | `runs/nocontext_ablation.json` |
| §5.2 confidence asymmetry, separability angles | `scripts/cr_c2_and_separability.py` | `runs/*_v2/features.npz` | `runs/cr_c2_and_separability.json` |
| §5.4 paired baseline tests | `scripts/cr_final_tables.py`, `scripts/cr_band_and_pairs.py` | `runs/*_rt/features.npz` | `runs/cr_final_tables.json` → `table3`, `runs/cr_band_and_pairs.json` → `c4_paired` |
| §5.4 in-domain with overlap residualized | `scripts/cr_joint_grouped_foldlocal.py` | `runs/*_rt/features.npz`, `data/ragtruth.jsonl` | `runs/cr_joint_grouped_foldlocal.json` |
| App. mid-layer band | `scripts/cr_band_and_pairs.py` | `runs/*_v2/features.npz` | `runs/cr_band_and_pairs.json` → `band` |
| App. fold-split sensitivity | `scripts/cr_estimator_domain_control.py` | `runs/*_rt/features.npz` | `runs/cr_estimator_domain_control.json` |
| App. parametric-belief rates | extraction stage | `runs/*_v2/parametric_belief.npz` | — |
| App. estimator × domain control | `scripts/cr_estimator_domain_control.py` | `runs/*_v2/features.npz`, `runs/*_rt/features.npz` | `runs/cr_estimator_domain_control.json` |

## Audit artifacts

Written in response to review, not cited as headline numbers, but released so the
checks can be repeated:

| Question | Script | Output |
|---|---|---|
| How much source-QA duplication is in CtrlPairs, and does it cross folds? | `scripts/cr_dupleak_audit.py` | `runs/cr_dupleak_audit.json` |
| Does regrouping by source QA move Table 1? | `scripts/cr_dupleak_table1.py` | `runs/cr_dupleak_table1.json` |
| What happens if duplicates are removed outright? | `scripts/cr_dupleak_dedup.py` | `runs/cr_dupleak_dedup.json` |
| Does passage-level grouping move the RAGTruth probe? | `scripts/cr_passage_leakage.py` | `runs/cr_passage_leakage.json` |
| Does the probe survive a within-passage-only comparison? | `scripts/cr_within_passage.py` | `runs/cr_within_passage.json` |
| Where did `max(AUC, 1-AUC)` actually bind? | `scripts/cr_protocol_audit_flip.py` | `runs/cr_protocol_audit_flip.json` |
| Signed RAGTruth AUROCs, pooled and per task | `scripts/cr_signed_auroc_ragtruth.py` | `runs/cr_signed_auroc_ragtruth.json` |

## Data

| File | Rows | Source |
|---|---|---|
| `data/ctrlpairs_v2.jsonl` | 3,593 (A 1,200 / C 1,200 / D 1,193) over 1,200 swap records and 555 source QAs | built from `pminervini/NQ-Swap` dev by `src/groundlm/data/build_ctrlpairs_v2.py` |
| `data/ragtruth.jsonl` | 2,000 responses over 450 source passages | strided subset of the 2,700-row `wandb/RAGTruth-processed` test split, by `src/groundlm/data/build_ragtruth.py` |
| `data/belief_items.jsonl` | 1,200 | closed-book parametric-belief check |
| `runs/*/features.npz` | cached residual states, `last_L` pooling only | `scripts/run_extract.py` (GPU) |
