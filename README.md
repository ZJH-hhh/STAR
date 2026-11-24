# STAR

**Structure and Text-Aware Relational meta-learning for few-shot molecular property prediction**

STAR uses assay text descriptions and molecular structure information for few-shot molecular property prediction. By default, the code enables all three STAR modules:

- Semantic task sampler
- L-TAFM task-adaptive feature modulation
- SK-Rel structure-knowledge relation module

<p align="center">
  <img src="figure/overview.jpg" width="85%" alt="STAR overview">
</p>

## Environment

Main dependencies:

```text
python 3.7
torch 1.7.1
torch_geometric 1.6.3
torch_scatter 2.0.7
learn2learn 0.1.6
rdkit 2022.9.3
scikit-learn
tensorboard
```

## Revision inputs

Place each benchmark CSV at `data/<physical_dataset>/<physical_dataset>.csv`. Use the same processed molecule ordering as the revision manifests. Pass the extracted revision artifact directory with `--revision_root`. The following paths are relative to that directory.

| Resource | Use |
| --- | --- |
| `eval_manifests/<dataset>/<K>shot/seed_<seed>.jsonl` | S9 support/query molecule IDs, including 20 repetitions per held-out assay |
| `text_embeddings/<dataset>/all_corpus/full.npz` | S28 frozen description embeddings, ordered by dataset-local task ID |
| Electronic Tables S26–S29 | Task splits, processed descriptions, encoder metadata and historical table-to-run index |
| Separate S30–S31 prediction index | Affected-compound subset under train-only fitting and corrected fingerprints |

## Assay text and matched controls

Use the original embeddings identified in S28 when replaying a historical checkpoint. To build new resources from the processed descriptions in S27 and check task IDs and splits against S26:

```bash
python scripts/precompute_property_embeddings.py --dataset all --fit_scope both \
  --text_table /path/to/Supplementary_Processed_Texts.csv \
  --split_table /path/to/Supplementary_Task_Splits.csv \
  --output_dir /path/to/new-revision/text_embeddings
```

The all-corpus implementation uses NumPy. The train-only implementation fits vocabulary, IDF and SVD on the meta-training assays and transforms held-out text with those fitted objects. The archived train-only environment used scikit-learn 1.7.1. Use a compatible Python environment for rebuilding those resources. Exact historical vectors, shuffled assignments and random controls require the archived matrices. Matching effective dimensions alone cannot verify vector values.

| Condition | Additional arguments to run.py |
| --- | --- |
| Full descriptions or names alone | `--text_variant full` or `--text_variant name` |
| Shuffled descriptions or fixed random vectors | `--text_variant shuffled` or `--text_variant random`; the resource follows the training seed |
| Fit text on meta-training tasks only | `--text_fit train_only` |
| Support-adapted task ID | `--conditioning support_adapted_id`; width comes from S28 or an explicit `--id_dim` |

The ID control learns training-assay identifiers, initializes a distinct identifier for each evaluation assay, and adapts it through the support loss. Its auxiliary draws are uniform and it uses no text input or semantic reward. For the text-conditioned runs, the diversity penalty is computed over the distinct selected assays in the training minibatch. The implementation evaluates their scale vectors before adaptation with FiLM dropout disabled, calculates the population variance across assays in each feature dimension, and averages over feature dimensions.

## Run the fixed-endpoint evaluation

```bash
python run.py --dataset tox21 --n_support 10 --random_seed 0 --gpu 0 \
  --revision_root /path/to/revision
```

The default trains for 2,000 episodes, saves `checkpoint.pt` before evaluation, and initializes each support repetition from that endpoint. The evaluator checks manifest identity, balanced supports and the complete remaining labeled query set against the local dataset. Use seeds 0, 1 and 2 separately. The baseline uses `--relation_backend baseline --use_knowledge_edges none --use_semantic_sampler 0 --use_ltafm 0`. Match the archived neighborhood setting for the intended configuration (9 for the reported 10-shot all-off reference).

**Auxiliary-assay sampling.** At meta-training time Sem-Samp mixes the softmax of assay-text cosine similarities (temperature 0.10) with a uniform distribution, giving each component weight 0.50. Sampling is without replacement. The default draw count follows the original code: all other meta-training assays. With a full-pool draw the weights affect draw order and every candidate is included. An optional smaller `--train_auxi_task_num` lets the weights affect which assays are included. The paper specifies the mixture and temperature; it gives no numerical draw count. At meta-test time auxiliary draws are uniform. New checkpoints and evaluation summaries record the resolved counts. Explicit missing-label edges are off by default, so only observed activities create property-label edges.

For paired inference, repeat the training command with the same architecture, seed, shot and manifest, adding `--eval_checkpoint <checkpoint.pt>`. Add `--mask_query_aux` to hide query auxiliary activities during both support adaptation and scoring. The default permits observed auxiliary activities. Both paths withhold query target labels from graph construction.

Each run writes per-episode and per-assay CSVs and a seed summary with the manifest SHA-256. AUROC and non-interpolated AP are percentages. Precision@100 is a fraction, and enrichment is the ratio to that assay/repetition's query prevalence. Ties at the screening cutoff use ascending molecule ID. Average repetitions within assay, then weight assays equally within seed. Combine ToxCast partitions by held-out assay counts before calculating the mean and sample SD over the three training seeds. The S29 index refers to the historical data archive. Newly generated files belong to a separate run.

Check that the historical S29 entries resolve inside the extracted release with `python scripts/verify_run_map.py /path/to/revision`. This checks the referenced files and compares AUROC and AP with each linked summary. Establishing run provenance requires the original experimental exports.

## Three-seed summaries, molecular predictions and attribution

To reproduce a displayed mean or paired interval from the S29 index, run:

```bash
python scripts/summarize_revision.py --run-map /path/to/Supplementary_Table_Run_Map.csv \
  --dataset muv --shot 10 --arm Full main star_full --arm Names main names_only --output muv.json
```

For new evaluations, omit `--run-map` and supply each arm's run directories as `--arm LABEL DIR0 DIR1 DIR2`. ToxCast requires all nine physical partitions for each seed. The script weights assays equally, checks manifest pairing and computes descriptive three-seed intervals and reference-minus-comparator contrasts.

Runs export `evaluation_protocol.json` and `per_episode_screening_top100.csv`. Add `--save_predictions` for compressed per-molecule probabilities, labels and empty-scaffold flags. For S30–S31, supply the corrected-cache, train-only, panel-visible PCBA 10-shot predictions under Full and Names in seed order:

```bash
python scripts/affected_subset.py --full FULL0.csv.gz FULL1.csv.gz FULL2.csv.gz \
  --names NAMES0.csv.gz NAMES1.csv.gz NAMES2.csv.gz --output affected.json
```

To evaluate S15, load each Tox21 checkpoint with `--eval_checkpoint` and add `--attribution_episodes contexts.jsonl`. Each of its three JSON lines identifies the task (1, 4 or 10), support seed, support IDs, query IDs and the same 32 selected molecule IDs in `task_id`, `support_seed`, `support_indices`, `query_indices` and `attribution_indices`. After support adaptation, the intervention changes final query-node embeddings with structural relations held fixed. Use the same contexts across all three seeds.

```bash
python scripts/summarize_attribution.py seed0/attribution.jsonl seed1/attribution.jsonl \
  seed2/attribution.jsonl --output attribution-summary.json
```

This implementation defaults to Spearman rank correlation, top-set intersection divided by the selected count, ten reproducible same-size random masks and 20,000 molecule-cluster bootstrap resamples. The optional `--overlap jaccard` and `--bootstrap-seed` select alternative documented conventions. Reproducing the reported numbers requires the original panel, episode inputs, checkpoints and analysis conventions.

## Whole-molecule fingerprints

The default computes 166 MACCS bits plus 2,048 radius-2 Morgan bits from every valid whole molecule. This includes acyclic molecules with an empty Bemis–Murcko scaffold. SK-Rel rebuilds the similarities from these features. The loader replaces unversioned fingerprints in old processed graphs on access and memoizes newly computed features by SMILES and mode. It preserves the molecule indices in those graphs. The source archive contains no benchmark fingerprint cache.

Use `--fingerprint_mode legacy` to reproduce the historical empty-scaffold rule. For an inference-only comparison, evaluate the same fixed checkpoint and manifest once in each mode. Training from scratch with corrected fingerprints constitutes a new run. Historical main results and the corrected-cache sensitivity results have distinct protocols.

Run the regression checks from the repository root:

```bash
python -m unittest discover -s tests
```
